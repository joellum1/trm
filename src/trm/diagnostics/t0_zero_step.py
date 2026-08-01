"""T0: zero-step evaluation of a vocab-transplanted checkpoint.

Question answered: does the pretrained maze checkpoint still score its
baseline exact-accuracy AFTER the 6 -> 11 vocabulary transplant, with no
fine-tuning at all?

  ~baseline  -> transplant is clean; the fine-tune failure is downstream
  near zero  -> the transplant itself is broken; fix before anything else

Run it twice to separate the transplant from everything else:

  # control: no transplant, vocab 6, should reproduce the published number
  python -m trm.diagnostics.t0_zero_step \
      --ckpt alphaXiv/trm-model-maze/maze_hard_step_32550 \
      --config checkpoints/maze-custom/all_config.yaml \
      --data data/maze-30x30-hard-1k \
      --no-transplant \
      --out results/t0_control.json

  # treatment: transplanted to vocab 11, same data
  python -m trm.diagnostics.t0_zero_step \
      --ckpt alphaXiv/trm-model-maze/maze_hard_step_32550 \
      --config checkpoints/maze-custom/all_config.yaml \
      --data data/maze-30x30-hard-1k \
      --out results/t0_transplant.json
"""

import os
import sys
import json
import copy
import argparse
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import yaml
import torch

from trm.training import PretrainConfig, create_dataloader, init_train_state
from trm.evaluation import create_evaluators, evaluate
from trm.diagnostics.transplant import (
    assert_charset_prefix, transplant_vocab, find_vocab_keys)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Local path, or HF 'user/repo/filename'")
    p.add_argument("--config", required=True,
                   help="all_config.yaml to base the architecture on")
    p.add_argument("--data", required=True,
                   help="ORIGINAL maze dataset (vocab 6)")
    p.add_argument("--out", required=True)
    p.add_argument("--v-old", type=int, default=6)
    p.add_argument("--v-new", type=int, default=11)
    p.add_argument("--transplant", action="store_true", default=True)
    p.add_argument("--no-transplant", dest="transplant", action="store_false")
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    return p.parse_args()


def resolve_ckpt(path: str) -> str:
    """Download from HF if the path isn't local."""
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    parts = path.split("/", 2)
    assert len(parts) == 3, f"HF path must be user/repo/filename, got {path!r}"
    local = hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename=parts[2])
    print(f"downloaded {path} -> {local}")
    return local


def main():
    a = parse_args()
    os.environ.setdefault("DISABLE_COMPILE", "1")

    # --- charset guard: cheap, and it fails loudly if ordering ever breaks ---
    from trm.data.custom import CHARSET
    assert_charset_prefix("# SGo", CHARSET)

    # --- load config, force eval settings ---
    with open(a.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_paths"] = [a.data]
    cfg["data_paths_test"] = [a.data]
    cfg["evaluators"] = []
    cfg["eval_save_outputs"] = []
    cfg["checkpoint_every_eval"] = False
    cfg["global_batch_size"] = a.global_batch_size
    cfg["load_checkpoint"] = None          # set below, after transplant

    # --- checkpoint: load, inspect, optionally transplant ---
    ckpt_path = resolve_ckpt(a.ckpt)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)

    vocab_keys = find_vocab_keys(sd, a.v_old)
    print("vocab-dependent tensors:")
    for k in vocab_keys:
        print(f"  {k}  {tuple(sd[k].shape)}")
    assert len(vocab_keys) == 2, f"expected embed + lm_head, got {vocab_keys}"

    target_vocab = a.v_old
    if a.transplant:
        sd = transplant_vocab(sd, vocab_keys, a.v_old, a.v_new)
        target_vocab = a.v_new
        print(f"transplanted vocab {a.v_old} -> {a.v_new}")
    else:
        print("no transplant (control run)")

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(sd, tmp.name)
    cfg["load_checkpoint"] = tmp.name

    config = PretrainConfig(**cfg)
    torch.random.manual_seed(config.seed)

    # --- data (vocab 6) but model built at target_vocab ---
    eval_loader, eval_metadata = create_dataloader(
        config, "test", rank=0, world_size=1, test_set_mode=True,
        epochs_per_iter=1, global_batch_size=config.global_batch_size)

    print(f"dataset vocab_size={eval_metadata.vocab_size} "
          f"seq_len={eval_metadata.seq_len} "
          f"puzzles={eval_metadata.num_puzzle_identifiers}")
    assert eval_metadata.vocab_size == a.v_old, (
        f"expected the ORIGINAL dataset (vocab {a.v_old}), "
        f"got vocab {eval_metadata.vocab_size} -- wrong --data path?")

    eval_metadata = copy.deepcopy(eval_metadata)
    eval_metadata.vocab_size = target_vocab   # build model at extended vocab

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        evaluators = []

    train_state = init_train_state(
        config, eval_metadata, rank=0, world_size=1, is_eval=True)
    train_state.model.eval()

    use_cuda = torch.cuda.is_available()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (a.bf16 and use_cuda) else nullcontext())

    with torch.inference_mode(), amp:
        metrics = evaluate(
            config=config, train_state=train_state,
            eval_loader=cast(Any, eval_loader), eval_metadata=eval_metadata,
            evaluators=evaluators, rank=0, world_size=1, cpu_group=None)

    flat = {}
    for _, m in cast(dict, metrics).items():
        if isinstance(m, dict) and "exact_accuracy" in m:
            flat = {k: float(v) for k, v in m.items()
                    if isinstance(v, (int, float))}
            break

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"

    res = {
        "exact_accuracy": flat.get("exact_accuracy"),
        "accuracy": flat.get("accuracy"),
        "lm_loss": flat.get("lm_loss"),
        "q_halt_accuracy": flat.get("q_halt_accuracy"),
        "steps": flat.get("steps"),
        "transplanted": a.transplant,
        "model_vocab": target_vocab,
        "data_vocab": a.v_old,
        "vocab_keys": vocab_keys,
        "ckpt": a.ckpt,
        "data": a.data,
        "git_sha": sha,
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()