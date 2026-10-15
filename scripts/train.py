"""Main training script for TRM models.

This script launches distributed training using Hydra for configuration.
Run with: python scripts/train.py [hydra overrides]
"""

import sys
import os

# Restore deprecated numpy aliases (np.int, ...) that some third-party libraries
# still use. Only fill in names numpy lacks: numpy 2.x defines np.bool/np.long,
# and overwriting np.bool breaks numpy.ma (surfaces later inside wandb.log()).
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
            continue
        try:
            setattr(np, _name, _val)
        except Exception:
            pass
except Exception:
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

# Optional so eval-only runs work without the compiled optimizer.
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
from trm.training.sync import CheckpointSync
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


def ema_state_dict(model: nn.Module, ema_helper: Optional[EMAHelper]) -> dict:
    """state_dict with EMA weights swapped in (no model deep-copy)."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if ema_helper is not None:
        for name, shadow in ema_helper.shadow.items():
            sd[name] = shadow.detach().clone()
    return sd


def save_resume_state(path: str, train_state: TrainState, ema_helper: Optional[EMAHelper]):
    """Weights, optimizer and EMA state, step. Written atomically."""
    tmp = path + ".tmp"
    torch.save({
        "step": train_state.step,
        "model": train_state.model.state_dict(),
        "optimizers": [o.state_dict() for o in train_state.optimizers],
        "ema": ema_helper.state_dict() if ema_helper is not None else None,
    }, tmp)
    os.replace(tmp, path)


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

    # Always written (also under WANDB_MODE=disabled): the record of what ran.
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

    step_mode = bool(config.eval_every_steps or config.checkpoint_every_steps)
    resume_path = os.path.join(config.checkpoint_path, "last.pt")
    resuming = config.resume and os.path.exists(resume_path)

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

        os.makedirs(config.checkpoint_path, exist_ok=True)

        metrics_csv = open(
            os.path.join(config.checkpoint_path, "training_metrics.csv"),
            "a" if resuming else "w",
            newline=""
        )

        metrics_writer = csv.writer(metrics_csv)
        if not resuming:
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

        # All eval metrics (incl. evaluator keys); the CSV has fixed columns.
        eval_jsonl = open(
            os.path.join(config.checkpoint_path, "eval_metrics.jsonl"),
            "a" if resuming else "w", encoding="utf-8"
        )

        save_code_and_config(config)
    else:
        eval_jsonl = None

    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Resume: restore state, then skip the batches already consumed (the loader is seeded).
    skip_batches = 0
    if resuming:
        state = torch.load(resume_path, map_location="cuda")
        train_state.model.load_state_dict(state["model"])
        for optim, sd in zip(train_state.optimizers, state["optimizers"]):
            optim.load_state_dict(sd)
        if ema_helper is not None and state["ema"] is not None:
            ema_helper.load_state_dict(state["ema"])
        train_state.step = skip_batches = state["step"]
        del state
        if RANK == 0:
            print(f"[run ] resumed from {resume_path} at step {train_state.step}")
            progress_bar.update(train_state.step)

    sync = None
    if RANK == 0:
        sync = CheckpointSync(config.hf_repo, config.drive_remote,
                              project_name=config.project_name, run_name=config.run_name)
        sync.push(os.path.join(config.checkpoint_path, "all_config.yaml"))

    def run_eval():
        """Evaluate (EMA weights if enabled) and log. Leaves the model in train mode."""
        if RANK == 0:
            print("EVALUATE")
        if config.ema:
            print("SWITCH TO EMA")
            eval_model = ema_helper.ema_copy(train_state.model)
        else:
            eval_model = train_state.model
        train_state_eval = TrainState(model=eval_model, optimizers=[], optimizer_lrs=[], carry=None,
                                      step=train_state.step, total_steps=train_state.total_steps)
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

            print("EVAL_METRICS step=%d " % train_state.step +
                  json.dumps(flat, sort_keys=True), flush=True)

            if eval_jsonl is not None:
                eval_jsonl.write(json.dumps(
                    {"step": train_state.step,
                     "epoch": train_state.step * config.global_batch_size / train_metadata.total_groups,
                     **flat}) + "\n")
                eval_jsonl.flush()

            if metrics_writer is not None:
                _m = _eval_set_metrics(metrics)
                metrics_writer.writerow([
                    train_state.step,
                    train_state.step * config.global_batch_size / train_metadata.total_groups,
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

        del train_state_eval, eval_model
        train_state.model.train()

    def save_checkpoint():
        """step_N (EMA weights, loadable by the eval scripts) + last.pt (resume)."""
        if RANK != 0 or config.checkpoint_path is None:
            return
        print(f"SAVE CHECKPOINT step={train_state.step}")
        os.makedirs(config.checkpoint_path, exist_ok=True)
        step_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}")
        torch.save(ema_state_dict(train_state.model, ema_helper), step_path)
        save_resume_state(resume_path, train_state, ema_helper)
        sync.push(step_path,
                  os.path.join(config.checkpoint_path, "eval_metrics.jsonl"),
                  os.path.join(config.checkpoint_path, "training_metrics.csv"))

    last_eval_step = last_ckpt_step = train_state.step if resuming else -1

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            if skip_batches > 0:
                skip_batches -= 1
                continue

            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                flat = _flatten_metrics(metrics)

                # Print before wandb.log so the numbers survive a wandb failure.
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

            if step_mode and train_state.step <= train_state.total_steps:
                if config.eval_every_steps and train_state.step % config.eval_every_steps == 0:
                    run_eval()
                    last_eval_step = train_state.step
                if config.checkpoint_every_steps and train_state.step % config.checkpoint_every_steps == 0:
                    save_checkpoint()
                    last_ckpt_step = train_state.step

        if step_mode:
            continue

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

    if step_mode:
        train_state.step = min(train_state.step, train_state.total_steps)
        if train_state.step != last_eval_step:
            run_eval()
        if train_state.step != last_ckpt_step:
            save_checkpoint()

    if sync is not None:
        sync.close()

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