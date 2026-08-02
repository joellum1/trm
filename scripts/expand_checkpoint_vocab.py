"""Transplant weights from an old checkpoint into a new (differently-shaped)
architecture -- specifically for the case where fine-tuning added new maze
tile types (checkpoint/reward/penalty), so the token embedding and output
head are a different size than the pretrained checkpoint.

For every parameter:
  - identical shape  -> copied over exactly
  - shape differs only in size (e.g. vocab dim grew from 6 -> 11) -> the
    overlapping region is copied (old tile types keep their learned weights),
    the rest is left at its fresh random initialization (new tile types start
    from scratch, same as any newly-added layer would)
  - key only exists in the new model -> left as fresh random init
  - key only exists in the old checkpoint -> FATAL (the two configs are not
    the same architecture family)

The overlap copy is only meaningful if the original token ids did not move,
i.e. NEW_CHARSET starts with OLD_CHARSET. That is asserted up front: without
it this script will happily copy learned embeddings into positions that now
mean something else and report success.

Usage:
    python scripts/expand_checkpoint_vocab.py \\
        --old-checkpoint checkpoints/maze/maze_hard_step_32550 \\
        --new-config checkpoints/maze-custom/all_config.yaml \\
        --dataset data/maze-custom \\
        --output checkpoints/maze/maze_hard_step_32550_expanded.pt

Then launch training with:
    +load_checkpoint=checkpoints/maze/maze_hard_step_32550_expanded.pt

Validate the result before trusting it (expect exact_accuracy ~0.8510):
    python -m trm.diagnostics.t0_zero_step \\
        --ckpt checkpoints/maze/maze_hard_step_32550_expanded.pt \\
        --config config/t0_zero_step.yaml \\
        --data data/maze-30x30-hard-1k \\
        --v-old 11 --no-transplant \\
        --out results/t0_expanded.json
"""

import os
import sys
import json
import argparse
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch

from trm.training import PretrainConfig, create_dataloader, init_train_state

# Frozen literal: the vocabulary the pretrained maze checkpoint was trained
# under. This is a historical fact and must never be imported from a module
# that could change alongside it.
PRETRAINED_CHARSET = "# SGo"

# Exactly these should change shape when growing the vocabulary. Anything
# else changing shape means something unexpected is vocab-dependent.
EXPECTED_PARTIAL = {
    'model.inner.embed_tokens.embedding_weight',
    'model.inner.lm_head.weight',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--old-checkpoint', required=True,
                   help='Path to the pretrained checkpoint (smaller vocab)')
    p.add_argument('--new-config', required=True,
                   help='all_config.yaml describing the TARGET architecture')
    p.add_argument('--dataset', required=True,
                   help='Dataset root used to build the target model, e.g. data/maze-custom')
    p.add_argument('--output', required=True, help='Where to write the merged checkpoint')
    p.add_argument('--allow-unexpected-partial', action='store_true',
                   help='Permit tensors outside EXPECTED_PARTIAL to change shape')
    p.add_argument('--skip-charset-check', action='store_true',
                   help='Skip the CHARSET prefix guard (you almost never want this)')
    return p.parse_args()


def check_charset():
    """The overlap copy assumes old token ids kept their meaning."""
    try:
        from trm.data.custom import CHARSET
    except ImportError as e:
        raise SystemExit(
            f"Could not import CHARSET from trm.data.custom ({e}).\n"
            "Ensure src/trm/data/custom/__init__.py exists and re-run `pip install -e .`"
        )

    prefix = CHARSET[:len(PRETRAINED_CHARSET)]
    if prefix != PRETRAINED_CHARSET:
        raise SystemExit(
            "\nFATAL: CHARSET was reordered -- original token ids have moved.\n"
            f"  pretrained charset : {PRETRAINED_CHARSET!r}\n"
            f"  current prefix     : {prefix!r}\n"
            f"  full CHARSET       : {CHARSET!r}\n\n"
            "New tile types must be APPENDED, not inserted. Copying the old\n"
            "embedding rows now would attach learned weights to symbols that\n"
            "mean something else. Fix CHARSET ordering and rebuild the dataset."
        )
    if len(set(CHARSET)) != len(CHARSET):
        raise SystemExit(f"FATAL: duplicate symbol in CHARSET: {CHARSET!r}")

    v_old = len(PRETRAINED_CHARSET) + 1   # +1 for PAD
    v_new = len(CHARSET) + 1
    print(f"CHARSET check OK: {PRETRAINED_CHARSET!r} -> {CHARSET!r}")
    print(f"  new symbols: {CHARSET[len(PRETRAINED_CHARSET):]!r}")
    print(f"  vocab {v_old} -> {v_new}\n")
    return v_old, v_new, CHARSET


def build_fresh_model(config_path: str, dataset: str):
    """Instantiate the target architecture with fresh random weights (no
    load_checkpoint), using the exact same construction path init_train_state
    already uses elsewhere -- guarantees matching key names/prefixes."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if 'defaults' in cfg or 'hydra' in cfg:
        raise SystemExit(
            f"{config_path} looks like a Hydra INPUT config (has 'defaults'/'hydra').\n"
            "Pass a resolved all_config.yaml from a checkpoint directory instead."
        )

    cfg['data_paths'] = [dataset]
    cfg['data_paths_test'] = [dataset]
    cfg['load_checkpoint'] = None
    cfg['evaluators'] = []
    cfg['global_batch_size'] = 1
    cfg['eval_save_outputs'] = []
    cfg['checkpoint_every_eval'] = False

    config = PretrainConfig(**cfg)
    _, metadata = create_dataloader(
        config, 'test', rank=0, world_size=1,
        test_set_mode=True, epochs_per_iter=1, global_batch_size=1
    )
    print(f"target dataset: vocab_size={metadata.vocab_size} seq_len={metadata.seq_len} "
          f"puzzles={metadata.num_puzzle_identifiers}")
    train_state = init_train_state(config, metadata, rank=0, world_size=1, is_eval=True)
    return train_state.model, metadata


def main():
    args = parse_args()
    os.environ.setdefault('DISABLE_COMPILE', '1')

    if args.skip_charset_check:
        print("WARNING: --skip-charset-check -- overlap copy is unverified\n")
        v_old = v_new = None
        charset = None
    else:
        v_old, v_new, charset = check_charset()

    print(f"Building target (new-vocab) architecture from {args.new_config} ...")
    new_model, metadata = build_fresh_model(args.new_config, args.dataset)
    new_state = new_model.state_dict()

    if v_new is not None and metadata.vocab_size != v_new:
        raise SystemExit(
            f"\nFATAL: dataset vocab_size={metadata.vocab_size} but CHARSET implies {v_new}.\n"
            f"  --dataset {args.dataset} was not built with the current CHARSET.\n"
            "  Rebuild it with trm.data.custom.build_maze_dataset."
        )

    print(f"Loading old checkpoint: {args.old_checkpoint}")
    old_state = torch.load(args.old_checkpoint, map_location='cpu', weights_only=False)
    old_state = old_state.get('model', old_state)

    merged: Dict[str, torch.Tensor] = {}
    exact, partial, new_random, ndim_skip, shrunk = [], [], [], [], []

    for key, new_tensor in new_state.items():
        if key not in old_state:
            merged[key] = new_tensor
            new_random.append((key, tuple(new_tensor.shape)))
            continue

        old_tensor = old_state[key]

        if tuple(old_tensor.shape) == tuple(new_tensor.shape):
            merged[key] = old_tensor.clone()
            exact.append(key)
        elif old_tensor.dim() == new_tensor.dim():
            # Truncation would silently discard learned weights -- refuse.
            for axis, (o, n) in enumerate(zip(old_tensor.shape, new_tensor.shape)):
                if n < o:
                    shrunk.append((key, axis, o, n))

            merged_tensor = new_tensor.clone()
            slices = tuple(slice(0, min(o, n))
                           for o, n in zip(old_tensor.shape, new_tensor.shape))
            merged_tensor[slices] = old_tensor[slices]
            merged[key] = merged_tensor
            partial.append((key, tuple(old_tensor.shape), tuple(new_tensor.shape)))
        else:
            merged[key] = new_tensor
            ndim_skip.append((key, tuple(old_tensor.shape), tuple(new_tensor.shape)))

    unmatched_old = [k for k in old_state if k not in new_state]

    # ---------------------------------------------------------------- report
    print(f"\n{len(exact)} parameter(s) copied exactly (identical shape).")

    print(f"\n{len(partial)} parameter(s) partially transplanted:")
    for key, old_shape, new_shape in partial:
        print(f"  {key}: {old_shape} -> {new_shape}  (overlap copied, rest fresh init)")

    if new_random:
        print(f"\n{len(new_random)} parameter(s) only in the new model -- fresh random init:")
        for key, shape in new_random:
            print(f"  {key}: {shape}")

    # ------------------------------------------------------------ hard fails
    problems = []

    if unmatched_old:
        problems.append(
            f"{len(unmatched_old)} key(s) in the old checkpoint have no match in the new "
            f"model -- the two configs are not the same architecture family:\n    "
            + "\n    ".join(unmatched_old))

    if ndim_skip:
        problems.append(
            "dimension-count mismatch (not a simple size change):\n    "
            + "\n    ".join(f"{k}: old {o} vs new {n}" for k, o, n in ndim_skip))

    if shrunk:
        problems.append(
            "a dimension SHRANK -- learned weights would be silently discarded:\n    "
            + "\n    ".join(f"{k} axis {a}: {o} -> {n}" for k, a, o, n in shrunk))

    got_partial = {k for k, _, _ in partial}
    if not args.allow_unexpected_partial and got_partial != EXPECTED_PARTIAL:
        problems.append(
            f"unexpected set of resized tensors.\n"
            f"    expected: {sorted(EXPECTED_PARTIAL)}\n"
            f"    got     : {sorted(got_partial)}\n"
            f"    (pass --allow-unexpected-partial if this is intentional)")

    if not partial and not ndim_skip:
        problems.append(
            "no shape mismatches at all -- the checkpoint already matches this "
            "architecture. Use --old-checkpoint directly as load_checkpoint and "
            "skip this script.")

    if problems:
        print("\n" + "=" * 70)
        for p in problems:
            print("FATAL: " + p)
        print("=" * 70)
        raise SystemExit(1)

    # ------------------------------------------------------------- verify
    print("\nVerifying transplanted tensors:")
    for key, old_shape, new_shape in partial:
        o, m = old_state[key], merged[key]
        n_old = o.shape[0]
        assert torch.equal(m[:n_old], o), f"{key}: old rows were modified"
        old_max = m[:n_old].abs().max().item()
        new_max = m[n_old:].abs().max().item()
        print(f"  {key}")
        print(f"    old rows preserved exactly  (|max| = {old_max:.5f})")
        print(f"    new rows fresh init          |max| = {new_max:.5f} "
              f"({new_max / old_max:.1%} of old)")
        if new_max == 0:
            print("    WARNING: new rows are all zero -- new tile types are "
                  "indistinguishable on the input side")
        if new_max > old_max:
            print("    WARNING: new rows exceed old rows in magnitude -- they may "
                  "dominate logits; check T0 before fine-tuning")

    # ------------------------------------------------------------- write
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(merged, args.output)

    meta = {
        'source_checkpoint': args.old_checkpoint,
        'new_config': args.new_config,
        'dataset': args.dataset,
        'pretrained_charset': PRETRAINED_CHARSET,
        'new_charset': charset,
        'vocab_old': v_old,
        'vocab_new': v_new,
        'n_exact': len(exact),
        'partial': [{'key': k, 'old': list(o), 'new': list(n)} for k, o, n in partial],
        'new_random': [{'key': k, 'shape': list(s)} for k, s in new_random],
    }
    with open(args.output + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nWrote {args.output}  ({size_mb:.1f} MB)")
    print(f"Wrote {args.output}.meta.json")
    print("\nNext: validate with T0 before fine-tuning (expect exact_accuracy ~0.8510).")


if __name__ == '__main__':
    main()