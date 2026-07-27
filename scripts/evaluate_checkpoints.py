"""Automated checkpoint evaluation sweep for TRM models.

Fixes the two bugs found in run_eval_only.py:
  1. Wrong batch size: cfg_pretrain.yaml defaults to global_batch_size=768,
     which OOMs / mismatches a model trained with global_batch_size=1.
  2. Wrong config source: cfg_pretrain.yaml carries pretraining defaults
     (data_paths=data/arc-aug-1000, evaluators=[arc@ARC], L_cycles=6, ...)
     that silently override the actual fine-tuning config, so the model
     reconstructed for eval doesn't architecturally match the checkpoint.

Fix: for each checkpoint, load ITS OWN all_config.yaml (the exact config
Hydra composed at training time) instead of config/cfg_pretrain.yaml, and
only override the few fields that must change for evaluation: dataset
path, batch size, and evaluators. This guarantees the reconstructed model
always matches the checkpoint it's loading weights into.

Usage:
    python scripts/evaluate_checkpoints.py checkpoints/maze-custom \\
        --dataset data/maze-30x30-hard-1k

    # only steps you care about, no EMA, save under a custom dir
    python scripts/evaluate_checkpoints.py checkpoints/maze-custom \\
        --dataset data/maze-30x30-hard-1k --no-apply-ema --outdir results/sweep1

Layout assumptions (adjust find_config_for_checkpoint() if yours differs):
  - <root>/step_<N>              (weight file)  or  <root>/step_<N>/       (weight dir)
  - all_config.yaml lives either inside the step_<N> directory, or as a
    sibling file in <root> shared across all steps of that run. Both are
    searched for automatically.

Output:
  - <outdir>/checkpoint_results.csv   one row per checkpoint
  - <outdir>/loss_vs_step.png
  - <outdir>/accuracy_vs_step.png
  - <outdir>/exact_accuracy_vs_step.png
  - printed summary of the best checkpoint by exact_accuracy

Single-process only (no torchrun). For multi-GPU sweeps, adapt the
distributed init block from run_eval_only.py.
"""

import os
import re
import sys
import csv
import copy
import glob
import argparse
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trm.training import PretrainConfig, create_dataloader, init_train_state
from trm.evaluation import create_evaluators, evaluate
from trm.models.ema import EMAHelper

try:
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
except Exception:
    pass

STEP_RE = re.compile(r'step_(\d+)')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint_root', help='Directory containing step_* checkpoints for one training run')
    p.add_argument('--dataset', default=None, help='Dataset path to evaluate on. Overrides data_paths/data_paths_test '
                                                     'from all_config.yaml. If omitted, whatever the config already '
                                                     'points at is used as-is.')
    p.add_argument('--pattern', default='step_*', help='Glob pattern (relative to checkpoint_root) for discovering checkpoints')
    p.add_argument('--outdir', default=None, help='Where to write CSV/plots (default: <checkpoint_root>_evalresults, '
                                                    'as a sibling of checkpoint_root)')
    p.add_argument('--global-batch-size', type=int, default=1, help='Forced eval batch size (default: 1, matching single-sample training)')
    p.add_argument('--apply-ema', action='store_true', default=True)
    p.add_argument('--no-apply-ema', dest='apply_ema', action='store_false')
    p.add_argument('--bf16', action='store_true', default=True)
    p.add_argument('--no-bf16', dest='bf16', action='store_false')
    p.add_argument('--limit', type=int, default=None, help='Only evaluate the first N discovered checkpoints (sorted by step)')
    return p.parse_args()


def discover_checkpoints(root: str, pattern: str) -> List[str]:
    """Find checkpoint paths under root matching pattern, sorted by step number."""
    candidates = glob.glob(os.path.join(root, pattern))
    numbered = []
    for c in candidates:
        m = STEP_RE.search(os.path.basename(c))
        if m:
            numbered.append((int(m.group(1)), c))
    numbered.sort(key=lambda x: x[0])
    return [c for _, c in numbered]


def find_config_for_checkpoint(ckpt_path: str) -> Optional[str]:
    """Locate all_config.yaml for a given checkpoint, checking the checkpoint's
    own directory first, then its parent directory (shared run-level config)."""
    search_dirs = []
    if os.path.isdir(ckpt_path):
        search_dirs.append(ckpt_path)
        search_dirs.append(os.path.dirname(ckpt_path.rstrip('/')))
    else:
        search_dirs.append(os.path.dirname(ckpt_path))
        search_dirs.append(os.path.dirname(os.path.dirname(ckpt_path)))

    for d in search_dirs:
        candidate = os.path.join(d, 'all_config.yaml')
        if os.path.exists(candidate):
            return candidate
    return None


def load_config_for_checkpoint(ckpt_path: str, config_path: str, dataset: Optional[str], batch_size: int) -> PretrainConfig:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # ---------- Force evaluation config (this is the actual fix) ----------
    if dataset is not None:
        cfg['data_paths'] = [dataset]
        cfg['data_paths_test'] = [dataset]
    cfg['load_checkpoint'] = ckpt_path
    cfg['evaluators'] = []                 # drop stray evaluators (e.g. arc@ARC) not relevant to this eval
    cfg['global_batch_size'] = batch_size  # never inherit a pretraining batch size
    cfg['eval_save_outputs'] = []          # skip saving per-example arrays during a sweep
    cfg['checkpoint_every_eval'] = False

    return PretrainConfig(**cfg)


def extract_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Pull lm_loss/accuracy/exact_accuracy/q_halt_accuracy out of whichever
    eval-set dict the evaluate() call returned them under."""
    out = {'lm_loss': float('nan'), 'accuracy': float('nan'),
           'exact_accuracy': float('nan'), 'q_halt_accuracy': float('nan')}
    for _, m in metrics.items():
        if isinstance(m, dict) and 'exact_accuracy' in m:
            for k in out:
                if k in m:
                    out[k] = float(m[k])
            break
    return out


def run_one_checkpoint(ckpt_path: str, config_path: str, args) -> Dict[str, float]:
    config = load_config_for_checkpoint(ckpt_path, config_path, args.dataset, args.global_batch_size)

    torch.random.manual_seed(config.seed)
    try:
        cudnn.benchmark = True
    except Exception:
        pass

    eval_loader, eval_metadata = create_dataloader(
        config, 'test', rank=0, world_size=1,
        test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size
    )

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        evaluators = []

    train_state = init_train_state(config, eval_metadata, rank=0, world_size=1, is_eval=True)

    train_state_eval = train_state
    if args.apply_ema or config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)
        # No explicit shadow file in a sweep; assume checkpoint already holds
        # EMA weights if the run saved them (init_train_state loaded whatever
        # load_checkpoint pointed at).
        train_state_eval = copy.deepcopy(train_state)

    ts = copy.deepcopy(train_state_eval)
    ts.model.eval()

    use_cuda = torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (args.bf16 and use_cuda) else nullcontext()

    with torch.inference_mode(), amp_ctx:
        metrics = evaluate(
            config=config, train_state=ts, eval_loader=cast(Any, eval_loader),
            eval_metadata=eval_metadata, evaluators=evaluators,
            rank=0, world_size=1, cpu_group=None,
        )

    return extract_metrics(cast(dict, metrics))


def main():
    args = parse_args()
    os.environ.setdefault('DISABLE_COMPILE', '1')

    root = args.checkpoint_root.rstrip('/')
    outdir = args.outdir or f'{root}_evalresults'
    os.makedirs(outdir, exist_ok=True)

    ckpts = discover_checkpoints(args.checkpoint_root, args.pattern)
    if args.limit:
        ckpts = ckpts[:args.limit]
    if not ckpts:
        print(f"No checkpoints matching '{args.pattern}' found under {args.checkpoint_root}")
        return

    rows = []
    for ckpt_path in ckpts:
        step_match = STEP_RE.search(os.path.basename(ckpt_path))
        step = int(step_match.group(1)) if step_match else -1
        name = os.path.basename(ckpt_path.rstrip('/'))

        config_path = find_config_for_checkpoint(ckpt_path)
        if config_path is None:
            print(f"  [skip] {name}: no all_config.yaml found near this checkpoint")
            continue

        print(f"Evaluating {name} (config: {config_path}) ...")
        try:
            m = run_one_checkpoint(ckpt_path, config_path, args)
        except Exception as e:
            print(f"  [FAILED] {name}: {e}")
            continue

        row = {'checkpoint': name, 'step': step, **m}
        rows.append(row)
        print(f"  lm_loss={m['lm_loss']:.4f}  accuracy={m['accuracy']:.4f}  "
              f"exact_accuracy={m['exact_accuracy']:.4f}  q_halt_accuracy={m['q_halt_accuracy']:.4f}")

    if not rows:
        print("No checkpoints evaluated successfully.")
        return

    rows.sort(key=lambda r: r['step'])

    csv_path = os.path.join(outdir, 'checkpoint_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'step', 'lm_loss', 'accuracy', 'exact_accuracy', 'q_halt_accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    steps = [r['step'] for r in rows]

    def plot_metric(key: str, ylabel: str, fname: str):
        vals = [r[key] for r in rows]
        plt.figure(figsize=(7, 4.5))
        plt.plot(steps, vals, marker='o')
        plt.xlabel('Step')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Step')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Wrote {path}")

    plot_metric('lm_loss', 'Loss', 'loss_vs_step.png')
    plot_metric('accuracy', 'Accuracy', 'accuracy_vs_step.png')
    plot_metric('exact_accuracy', 'Exact Accuracy', 'exact_accuracy_vs_step.png')

    best = max(rows, key=lambda r: r['exact_accuracy'])
    print(f"\nBest checkpoint: {best['checkpoint']}")
    print(f"Exact accuracy: {best['exact_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()