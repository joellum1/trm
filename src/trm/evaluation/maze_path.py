"""Structural evaluation of maze path predictions.

Exact-match scoring cannot distinguish "emitted a valid alternate shortest
path" from "emitted garbage". Both score zero. This evaluator decomposes the
prediction into properties that exact-match collapses:

  path/connected              single 4-connected component from S to G
  path/legal                  no predicted path cell sits on a wall
  path/valid                  connected AND legal AND has exactly one S, one G
  path/optimal                valid AND same length as the reference solution
  path/valid_not_exact        valid, optimal length, but DIFFERS from the label
                              -> the alternate-path mislabel rate
  path/checkpoint_coverage    fraction of mandatory C tiles the path crosses
  path/all_checkpoints        fraction of mazes hitting every C tile
  path/largest_component      largest connected fragment / total path cells
                              -> moves long before exact_accuracy does; the
                                 early-warning signal during fine-tuning
  path/length_ratio           predicted length / reference length (valid only)

Token ids (trm/data/custom/build_maze_dataset.py, CHARSET = "# SGoCRrNn"):
  0 PAD   1 '#' wall   2 ' ' empty   3 'S' start   4 'G' goal   5 'o' path
  6 'C' checkpoint     7 'R' reward  8 'r' reward crossed
  9 'N' penalty       10 'n' penalty crossed

Enable with, in the training config:

  evaluators:
    - name: maze_path@MazePath
"""

import os
import json
from collections import deque
from typing import Dict, Optional, Sequence, Any

import numpy as np
import torch
import torch.distributed as dist

from trm.data.puzzle_dataset import PuzzleDatasetMetadata

PAD, WALL, EMPTY, START, GOAL, PATH = 0, 1, 2, 3, 4, 5
CHECKPOINT, REWARD, REWARD_X, PENALTY, PENALTY_X = 6, 7, 8, 9, 10

# Cells that count as "on the path" in a solution grid.
PATH_TOKENS = frozenset({START, GOAL, PATH, CHECKPOINT, REWARD_X, PENALTY_X})
# Cells that are impassable in the input grid.
BLOCKED_TOKENS = frozenset({WALL})


def _components(mask: np.ndarray):
    """4-connected components of a boolean grid. Returns list of coord lists."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            comp, q = [], deque([(r0, c0)])
            seen[r0, c0] = True
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            out.append(comp)
    return out


def _score_one(inp: np.ndarray, lab: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Score a single (input, label, prediction) grid triple."""
    pred_path = np.isin(pred, list(PATH_TOKENS))
    lab_path = np.isin(lab, list(PATH_TOKENS))
    blocked = np.isin(inp, list(BLOCKED_TOKENS))

    n_pred = int(pred_path.sum())
    n_lab = int(lab_path.sum())

    # --- component structure (partial credit, works even when badly wrong) ---
    comps = _components(pred_path)
    largest = max((len(c) for c in comps), default=0)
    largest_frac = largest / n_pred if n_pred else 0.0

    # --- hard validity checks ---
    n_start = int((pred == START).sum())
    n_goal = int((pred == GOAL).sum())
    has_endpoints = (n_start == 1) and (n_goal == 1)

    legal = not bool((pred_path & blocked).any())

    connected = False
    if has_endpoints and comps:
        sr, sc = map(int, np.argwhere(pred == START)[0])
        gr, gc = map(int, np.argwhere(pred == GOAL)[0])
        for comp in comps:
            s = set(comp)
            if (sr, sc) in s and (gr, gc) in s:
                connected = len(comp) == n_pred   # single component only
                break

    valid = bool(connected and legal and has_endpoints)
    optimal = bool(valid and n_pred == n_lab)
    exact = bool(np.array_equal(pred, lab))
    valid_not_exact = bool(optimal and not exact)

    # --- mandatory checkpoints ---
    cp = (inp == CHECKPOINT)
    n_cp = int(cp.sum())
    if n_cp:
        hit = int((cp & pred_path).sum())
        cp_cov = hit / n_cp
        cp_all = float(hit == n_cp)
    else:
        cp_cov, cp_all = float("nan"), float("nan")

    return {
        "connected": float(connected),
        "legal": float(legal),
        "has_endpoints": float(has_endpoints),
        "valid": float(valid),
        "optimal": float(optimal),
        "valid_not_exact": float(valid_not_exact),
        "largest_component": largest_frac,
        "n_components": float(len(comps)),
        "length_ratio": (n_pred / n_lab) if (valid and n_lab) else float("nan"),
        "checkpoint_coverage": cp_cov,
        "all_checkpoints": cp_all,
    }


class MazePath:
    """Structural path metrics for maze predictions."""

    required_outputs = {"inputs", "labels", "preds"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata,
                 max_examples: int = 2000, save_failures: int = 32, **kwargs):
        self.data_path = data_path
        self.seq_len = eval_metadata.seq_len
        side = int(round(self.seq_len ** 0.5))
        assert side * side == self.seq_len, f"seq_len {self.seq_len} is not square"
        self.side = side
        self.max_examples = max_examples
        self.save_failures = save_failures
        self._rows = []
        self._failures = []

    def begin_eval(self):
        self._rows = []
        self._failures = []

    def update_batch(self, batch: Dict[str, torch.Tensor],
                     preds: Dict[str, torch.Tensor]):
        got = {}
        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    got[k] = v.detach().cpu()
        if not self.required_outputs.issubset(got):
            return
        if len(self._rows) >= self.max_examples:
            return

        inputs = got["inputs"].numpy()
        labels = got["labels"].numpy()
        predsn = got["preds"].numpy()

        # Padding rows are all-PAD labels. (Do NOT mask on puzzle_identifiers:
        # maze has num_puzzle_identifiers=1 and blank_identifier_id=0, so every
        # real example carries the blank id.)
        keep = (labels != PAD).any(axis=-1)

        s = self.side
        for i in np.nonzero(keep)[0]:
            if len(self._rows) >= self.max_examples:
                break
            inp = inputs[i].reshape(s, s)
            lab = labels[i].reshape(s, s)
            prd = predsn[i].reshape(s, s)
            row = _score_one(inp, lab, prd)
            self._rows.append(row)
            if not row["valid"] and len(self._failures) < self.save_failures:
                self._failures.append(
                    {"input": inp.tolist(), "label": lab.tolist(),
                     "pred": prd.tolist(), "scores": row})

    def result(self, save_path: Optional[str], rank: int, world_size: int,
               group: Optional[Any] = None) -> Optional[Dict[str, float]]:
        rows = self._rows
        if world_size > 1:
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(rows, gathered, dst=0, group=group)
            if rank != 0:
                return None
            rows = [r for part in gathered for r in part]  # type: ignore
        elif rank != 0:
            return None

        if not rows:
            return None

        keys = rows[0].keys()
        out = {}
        for k in keys:
            vals = np.array([r[k] for r in rows], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if len(vals):
                out[f"path/{k}"] = float(vals.mean())
        out["path/n_scored"] = float(len(rows))

        if save_path and self._failures:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "failures.json"), "w") as f:
                json.dump(self._failures, f)

        return out