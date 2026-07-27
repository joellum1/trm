"""Run inference with a fine-tuned TRM checkpoint on held-out mazes and save the
raw inputs/labels/preds tensors in exactly the format visualise_maze.py expects:

    torch.save({"inputs": <N, seq_len>, "labels": <N, seq_len>, "preds": <N, seq_len>}, output)

visualise_maze.py then does data["inputs"].numpy().reshape(-1, GRID_SIZE, GRID_SIZE)
etc., so no vocab/token knowledge lives in this script at all -- it just moves the
model's raw token ids to disk. Whatever tile types your build_maze_dataset.py
CHARSET defines (including your custom checkpoint/reward/penalty tiles) pass
through untouched.

Usage:
    python scripts/generate_predictions.py \\
        --checkpoint checkpoints/maze-custom/step_12000 \\
        --dataset data/maze-custom \\
        --num-samples 50 \\
        --output results/maze/step_12000_preds.pt

    python scripts/visualise_maze.py \\
        --preds results/maze/step_12000_preds.pt \\
        --num_samples 10 --output_dir results/maze
"""

import os
import sys
import argparse
from contextlib import nullcontext
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch

from trm.training import PretrainConfig, create_dataloader, init_train_state

try:
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to a single step_N checkpoint, e.g. checkpoints/maze-custom/step_12000')
    p.add_argument('--dataset', required=True, help="Dataset root (e.g. data/maze-custom); reads the 'test' split under it")
    p.add_argument('--num-samples', type=int, default=50,
                    help="How many test mazes to run inference on and save (default: 50). "
                         "All go through the model in a single batch, so keep this within "
                         "what fits on your GPU -- visualise_maze.py's own --num_samples/"
                         "--random_seed then picks a subset of these to actually plot.")
    p.add_argument('--output', required=True, help='Where to write the preds .pt file, e.g. results/maze/step_12000_preds.pt')
    p.add_argument('--bf16', action='store_true', default=True)
    p.add_argument('--no-bf16', dest='bf16', action='store_false')
    return p.parse_args()


def find_config_for_checkpoint(ckpt_path: str) -> Optional[str]:
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


def load_config_for_checkpoint(ckpt_path: str, config_path: str, dataset: str, batch_size: int) -> PretrainConfig:
    """Same fix as evaluate_checkpoints.py: load THIS checkpoint's own
    all_config.yaml (not config/cfg_pretrain.yaml) so the reconstructed model
    always matches the weights being loaded."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    cfg['data_paths'] = [dataset]
    cfg['data_paths_test'] = [dataset]
    cfg['load_checkpoint'] = ckpt_path
    cfg['evaluators'] = []
    cfg['global_batch_size'] = batch_size
    cfg['eval_save_outputs'] = []
    cfg['checkpoint_every_eval'] = False

    return PretrainConfig(**cfg)


def main():
    args = parse_args()
    os.environ.setdefault('DISABLE_COMPILE', '1')

    ckpt_path = args.checkpoint.rstrip('/')

    config_path = find_config_for_checkpoint(ckpt_path)
    if config_path is None:
        print(f"No all_config.yaml found near {ckpt_path}. Can't reconstruct a matching model.")
        return

    print(f"Checkpoint: {ckpt_path}")
    print(f"Config:     {config_path}")

    config = load_config_for_checkpoint(ckpt_path, config_path, args.dataset, args.num_samples)

    torch.random.manual_seed(config.seed)

    eval_loader, eval_metadata = create_dataloader(
        config, 'test', rank=0, world_size=1,
        test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size
    )

    train_state = init_train_state(config, eval_metadata, rank=0, world_size=1, is_eval=True)
    train_state.model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (args.bf16 and device == 'cuda') else nullcontext()

    set_name, batch, actual_count = next(iter(eval_loader))
    n = min(args.num_samples, actual_count)
    print(f"Running inference on {n} sample(s) from '{set_name}' split ...")

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode(), amp_ctx:
        with torch.device(device):
            carry = train_state.model.initial_carry(batch)
        steps = 0
        while True:
            carry, loss, metrics, outputs, all_finish = train_state.model(
                carry=carry, batch=batch, return_keys={'preds'}
            )
            steps += 1
            if all_finish:
                break
    print(f"  Completed inference in {steps} ACT step(s)")

    save_data = {
        'inputs': batch['inputs'][:n].cpu(),
        'labels': batch['labels'][:n].cpu(),
        'preds': outputs['preds'][:n].cpu(),
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(save_data, args.output)
    print(f"Wrote {args.output}  (inputs/labels/preds each shaped {tuple(save_data['inputs'].shape)})")


if __name__ == '__main__':
    main()