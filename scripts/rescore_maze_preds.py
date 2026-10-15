"""Evaluate a maze checkpoint and classify every prediction with maze_checker.

Uses the training evaluate() path, so exact accuracy matches training evals.

    python scripts/rescore_maze_preds.py --checkpoint checkpoints/maze/maze_hard_step_32550 \\
        --config config/resolved/maze_hard_pretrained.yaml \\
        --dataset data/maze-30x30-hard-1k --outdir results/maze_hard_baseline

Writes preds.pt (token tensors), verdicts.jsonl (all mazes), failures.jsonl
(pred != label) and summary.json to --outdir.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
import torch

from trm.training import PretrainConfig, create_dataloader, init_train_state
from trm.evaluation import evaluate, MazePath
from trm.evaluation.maze_checker import check, summarize, Verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="resolved config yaml matching the checkpoint's architecture")
    p.add_argument("--dataset", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch-size", type=int, default=250)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.update(
        data_paths=[args.dataset], data_paths_test=[args.dataset],
        load_checkpoint=args.checkpoint, global_batch_size=args.batch_size,
        evaluators=[], eval_save_outputs=["inputs", "labels", "preds"],
        checkpoint_path=args.outdir,
    )
    config = PretrainConfig(**cfg)
    os.makedirs(args.outdir, exist_ok=True)
    torch.random.manual_seed(config.seed)

    loader, meta = create_dataloader(config, "test", rank=0, world_size=1, test_set_mode=True,
                                     epochs_per_iter=1, global_batch_size=config.global_batch_size)
    state = init_train_state(config, meta, rank=0, world_size=1, is_eval=True)
    state.model.eval()

    metrics = evaluate(config, state, loader, meta, [], rank=0, world_size=1, cpu_group=None)
    set_metrics = next(iter(metrics.values()))

    # evaluate() saved step_0_all_preds.0; drop padding rows and rename.
    raw_path = os.path.join(args.outdir, "step_0_all_preds.0")
    raw = torch.load(raw_path)
    keep = (raw["inputs"] != 0).any(dim=-1)
    data = {k: raw[k][keep] for k in ("inputs", "labels", "preds")}
    torch.save(data, os.path.join(args.outdir, "preds.pt"))
    os.remove(raw_path)

    inputs, labels, preds = (data[k].numpy() for k in ("inputs", "labels", "preds"))
    verdicts = []
    with open(os.path.join(args.outdir, "verdicts.jsonl"), "w") as fa, \
         open(os.path.join(args.outdir, "failures.jsonl"), "w") as ff:
        for i in range(len(inputs)):
            v = check(inputs[i], preds[i], labels[i])
            verdicts.append(v)
            line = json.dumps({"index": i, **v.to_dict()}) + "\n"
            fa.write(line)
            if not v.exact:
                ff.write(line)

    failures = [v for v in verdicts if not v.exact]
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n": len(verdicts),
        "exact_accuracy_model_head": float(set_metrics["exact_accuracy"]),
        "all": summarize(verdicts),
        "failures_only": summarize(failures),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    n, nf = len(verdicts), len(failures)
    print(f"\n{n} mazes, exact {n - nf}/{n} ({(n - nf) / n:.2%}); model-head exact_accuracy {summary['exact_accuracy_model_head']:.4f}")
    print(f"{nf} non-exact predictions re-scored:")
    for cat in ("valid_shortest", "valid_longer", "broken"):
        k = sum(v.category == cat for v in failures)
        print(f"  {cat:15s} {k:4d}  ({k / max(nf, 1):.1%} of failures)")
    for code, frac in sorted(summary["failures_only"].items(), key=lambda kv: -kv[1]):
        if code.startswith("broken_primary/") and frac:
            print(f"    broken, first reason {code.split('/')[1]:18s} {round(frac * nf):4d}")


if __name__ == "__main__":
    main()
