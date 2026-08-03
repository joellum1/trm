"""Main training script for TRM models.

This script launches distributed training using Hydra for configuration.
Run with: python scripts/train.py [hydra overrides]
"""

import sys
import os

# Compatibility shim: restore a few deprecated numpy aliases that some
# third-party libraries (e.g. networkx, older packages) still reference
# (like `np.int`, `np.bool`). This avoids AttributeError on import when
# running with newer numpy versions that may have removed those names.
#
# CRITICAL: only fill in names numpy does NOT already define. numpy 2.x
# reintroduced `np.bool` (== np.bool_) and `np.long`, and overwriting them
# with the Python builtins corrupts numpy.ma: its `nomask` sentinel is
# built from `np.bool`, so it becomes a plain Python `False` and every
# masked-array operation dies with
#     AttributeError: 'bool' object has no attribute 'view'
# numpy.ma is imported lazily (pandas -> wandb), so the failure surfaces
# far from here, typically inside wandb.log().
try:
    import numpy as np

    _np_compat_aliases = {
        'bool': bool,
        'int': int,
        'float': float,
        'object': object,
        'str': str,
        'long': int,
        'unicode': str,
        'bytes': bytes,
    }

    for _name, _val in _np_compat_aliases.items():
        if hasattr(np, _name):
            # numpy defines this itself -- leave it alone.
            continue
        try:
            setattr(np, _name, _val)
        except Exception:
            # Best-effort: ignore if we cannot set the attribute.
            pass
except Exception:
    # If numpy isn't available or something else goes wrong, continue
    # without blocking imports — downstream imports may still error.
    print("Couldn't load numpy. Pls check the required version and/or if it's installed correctly.")

import torch
from typing import Optional, Any, Sequence, List, cast
from dataclasses import dataclass
import os
import math
import json
import yaml
import shutil
import copy
import csv

import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

# Make adam_atan2 optional so evaluation-only runs don't crash when the
# optimizer package isn't installed. If unavailable, AdamATan2 will be
# set to None and optimizer construction should be skipped for eval.
try:
    from trm.training.optimizers import AdamATan2
except Exception:
    AdamATan2 = None

from trm.data import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from trm.utils import load_model_class, get_model_source_path
from trm.models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from trm.models.ema import EMAHelper
from trm.training import (
    PretrainConfig, TrainState, ArchConfig, LossConfig, EvaluatorConfig,
    create_dataloader, create_model, init_train_state, train_batch,
    save_train_state, load_checkpoint, compute_lr,
)
from trm.evaluation import evaluate, create_evaluators


def _flatten_metrics(metrics) -> dict:
    """evaluate() returns {set_name: {metric: value}}; evaluators merge their
    own flat keys (e.g. 'path/valid') in at the top level. Flatten both into
    one dict of plain floats."""
    out = {}
    if not isinstance(metrics, dict):
        return out
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                try:
                    out[f"{k}/{kk}"] = float(vv)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _eval_set_metrics(metrics) -> dict:
    """Pull out the per-set metric dict that carries exact_accuracy."""
    if not isinstance(metrics, dict):
        return {}
    for v in metrics.values():
        if isinstance(v, dict) and "exact_accuracy" in v:
            return v
    return {}


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml. Written unconditionally: this file is the record of
    # what actually ran (load_checkpoint, batch size, lr, ...) and is the first
    # thing you want when a run misbehaves. It used to be skipped whenever
    # wandb.run was None, i.e. under WANDB_MODE=disabled.
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    if wandb.run is not None:
        try:
            wandb.run.log_code(config.checkpoint_path)
        except Exception as e:
            print(f"wandb.log_code skipped: {e}")


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="../config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except Exception as e:
        print(f"NO EVAL DATA FOUND: {e}")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        print(f"No evaluator found: {e}")
        evaluators = []

    if RANK == 0:
        print(f"[data] train examples={train_metadata.total_groups} "
              f"vocab={train_metadata.vocab_size} seq_len={train_metadata.seq_len}")
        print(f"[run ] batch={config.global_batch_size} epochs={config.epochs} "
              f"eval_interval={config.eval_interval} lr={config.lr} "
              f"load_checkpoint={config.load_checkpoint}")
        print(f"[eval] evaluators={[type(e).__name__ for e in evaluators]}")

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    metrics_csv = None
    metrics_writer = None
    if RANK == 0:
        print(f"[run ] total_steps={train_state.total_steps} "
              f"(~{train_state.total_steps / max(total_iters, 1):.0f} per eval interval)")
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)

        # Bug fix: config.checkpoint_path isn't created on disk until
        # save_code_and_config() runs its os.makedirs() call below -- and
        # that happens *after* this block used to try opening the metrics
        # CSV inside it, which raised FileNotFoundError on a fresh run.
        # Ensure the directory exists first.
        os.makedirs(config.checkpoint_path, exist_ok=True)

        metrics_csv = open(
            os.path.join(config.checkpoint_path, "training_metrics.csv"),
            "w",
            newline=""
        )

        metrics_writer = csv.writer(metrics_csv)
        metrics_writer.writerow([
            "step",
            "epoch",
            "train_loss",
            "eval_loss",
            "accuracy",
            "exact_accuracy",
            "q_halt_accuracy",
            "q_halt_loss",
            "learning_rate"
        ])

        # Full eval metrics (including evaluator keys like path/*) go to their
        # own file -- the fixed-column CSV above cannot represent them.
        eval_jsonl = open(
            os.path.join(config.checkpoint_path, "eval_metrics.jsonl"),
            "w", encoding="utf-8"
        )

        # Bug fix: this used to call wandb.log(metrics, ...) here, but
        # `metrics` doesn't exist until the training loop below assigns it --
        # that was a guaranteed NameError on every run. Nothing valid to log
        # yet at this point, so the call is just removed.

        save_code_and_config(config)
    else:
        eval_jsonl = None

    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                flat = _flatten_metrics(metrics)

                # Print BEFORE wandb.log: if wandb raises, the numbers survive
                # in the log rather than being lost with the run.
                if train_state.step % 20 == 0 or train_state.step <= 5:
                    print("TRAIN_METRICS step=%d " % train_state.step +
                          "  ".join(f"{k}={v:.5f}" for k, v in sorted(flat.items())),
                          flush=True)

                if metrics_writer is not None:
                    loss = next((v for k, v in flat.items() if k.endswith("loss")), "")
                    lr = next((v for k, v in flat.items() if k.endswith("lr")), "")
                    metrics_writer.writerow([
                        train_state.step,
                        _iter_id * train_epochs_per_iter,
                        loss,
                        "", "", "", "", "",
                        lr,
                    ])
                    metrics_csv.flush()

                try:
                    wandb.log(metrics, step=train_state.step)
                except Exception as e:
                    print(f"wandb.log failed (continuing): {type(e).__name__}: {e}")

                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                flat = _flatten_metrics(metrics)

                # Print BEFORE wandb.log, for the same reason as above.
                print("EVAL_METRICS step=%d " % train_state.step +
                      json.dumps(flat, sort_keys=True), flush=True)

                if eval_jsonl is not None:
                    eval_jsonl.write(json.dumps(
                        {"step": train_state.step,
                         "epoch": (_iter_id + 1) * train_epochs_per_iter,
                         **flat}) + "\n")
                    eval_jsonl.flush()

                if metrics_writer is not None:
                    _m = _eval_set_metrics(metrics)
                    metrics_writer.writerow([
                        train_state.step,
                        (_iter_id + 1) * train_epochs_per_iter,
                        "",
                        _m.get("loss", ""),
                        _m.get("accuracy", ""),
                        _m.get("exact_accuracy", ""),
                        _m.get("q_halt_accuracy", ""),
                        _m.get("q_halt_loss", ""),
                        ""
                    ])
                    metrics_csv.flush()

                try:
                    wandb.log(metrics, step=train_state.step)
                except Exception as e:
                    print(f"wandb.log failed (continuing): {type(e).__name__}: {e}")

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        wandb.finish()
    except Exception:
        pass
    if metrics_csv is not None:
        metrics_csv.close()
    if eval_jsonl is not None:
        eval_jsonl.close()


if __name__ == "__main__":
    launch()