"""Route checker: maze + predicted grid -> valid_shortest | valid_longer | broken, with reasons.

A route is a walk, not a simple path: a checkpoint in a dead end is entered and
left through the same corridor, so labels contain spurs. A prediction is valid when
  1. every cell keeps its tile, except path cells take their marker
     (' '->'o', 'R'->'r', 'N'->'n'; S, G, C unchanged);
  2. S, G and every C are marked and all marked cells are connected;
  3. the cheapest S -> all C -> G walk within the marked cells uses every marked cell.
It is shortest when that walk's cost (solve.py tile costs; steps on Maze-Hard)
equals the optimum, even if it differs from the label. Works for Maze-Hard too:
its charset "# SGo" is a prefix of "# SGoCRrNn".

    python -m trm.evaluation.maze_checker preds.pt --out verdicts.jsonl
(preds.pt holds {"inputs", "labels", "preds"} token tensors.)
"""

import math
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from trm.data.custom.solve import (
    solve_maze, DEFAULT_REWARD_BONUS, DEFAULT_PENALTY_COST,
)

CHARSET = "# SGoCRrNn"   # token id k <-> CHARSET[k - 1]; 0 is PAD

VALID_SHORTEST = "valid_shortest"
VALID_LONGER = "valid_longer"
BROKEN = "broken"
CATEGORIES = (VALID_SHORTEST, VALID_LONGER, BROKEN)

# Reported in this order, most fundamental first.
REASON_CODES = (
    "bad_shape",          # prediction is not the same size as the maze
    "bad_token",          # PAD or an id outside the charset
    "path_on_wall",       # a path marker where the maze has a wall
    "tile_altered",       # a cell changed to something other than its path marker
    "wrong_marker",       # wrong marker for the tile (e.g. 'o' on R)
    "start_missing",      # S is not on the predicted route
    "goal_missing",       # G is not on the predicted route
    "disconnected",       # S and G are in different marked pieces
    "missed_checkpoint",  # a mandatory C is not reached from S
    "stray_cells",        # marked cells in pieces separate from the S-G route
    "branch",             # marked cells joined to the route but not needed by it
    "longer",             # valid_longer only
)

_PATH_MARKER = {" ": "o", "R": "r", "N": "n", "S": "S", "G": "G", "C": "C"}
_ON_PATH = frozenset("oSGCrn")
_EPS = 1e-6

Grid = Union[str, Sequence[str], np.ndarray]


@dataclass
class Verdict:
    category: str
    reasons: List[Tuple[str, str]] = field(default_factory=list)  # (code, message)
    pred_cost: Optional[float] = None
    opt_cost: Optional[float] = None
    pred_steps: Optional[int] = None
    opt_steps: Optional[int] = None
    exact: Optional[bool] = None  # pred == label, when a label is given

    @property
    def primary_reason(self) -> Optional[str]:
        return self.reasons[0][0] if self.reasons else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = [{"code": c, "message": m} for c, m in self.reasons]
        return d


def to_rows(grid: Grid, side: Optional[int] = None) -> List[str]:
    """Row strings from rows, a newline-joined string, or a token-id array (flat or
    2-D). PAD/unknown ids become '?' (reported as bad_token)."""
    if isinstance(grid, str):
        rows = grid.replace("\r", "").split("\n")
        return [r for r in rows if r != ""] if "\n" in grid else _square(grid, side)
    if isinstance(grid, np.ndarray) or hasattr(grid, "numpy"):
        arr = np.asarray(grid.numpy() if hasattr(grid, "numpy") else grid)
        if arr.dtype.kind in "iu":
            if arr.ndim == 1:
                s = side or int(round(math.sqrt(arr.size)))
                arr = arr.reshape(s, s)
            lut = np.array(["?"] + list(CHARSET) + ["?"] * 256)
            ids = np.clip(arr.astype(np.int64), -1, len(CHARSET) + 1)
            ids = np.where(ids < 0, len(CHARSET) + 1, ids)
            return ["".join(lut[row]) for row in ids]
        return ["".join(map(str, row)) for row in arr]
    return [str(r) for r in grid]


def _square(flat: str, side: Optional[int]) -> List[str]:
    s = side or int(round(math.sqrt(len(flat))))
    return [flat[i * s:(i + 1) * s] for i in range(s)]


def _fmt(cells: Sequence[Tuple[int, int]], limit: int = 4) -> str:
    shown = ", ".join(f"({r},{c})" for r, c in cells[:limit])
    return shown + (f" +{len(cells) - limit} more" if len(cells) > limit else "")


def _components(cells: set) -> List[set]:
    seen, comps = set(), []
    for start in cells:
        if start in seen:
            continue
        comp, q = {start}, deque([start])
        seen.add(start)
        while q:
            r, c = q.popleft()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def optimal_cost(maze_rows: List[str], reward_bonus: float = DEFAULT_REWARD_BONUS,
                 penalty_cost: float = DEFAULT_PENALTY_COST) -> Tuple[float, int]:
    """(cost, steps) of the optimum; the expensive part, so cache it per maze."""
    _, path, cost = solve_maze(maze_rows, reward_bonus=reward_bonus, penalty_cost=penalty_cost)
    return cost, len(path) - 1


def check(maze: Grid, pred: Grid, label: Optional[Grid] = None, *,
          opt: Optional[Tuple[float, int]] = None,
          reward_bonus: float = DEFAULT_REWARD_BONUS,
          penalty_cost: float = DEFAULT_PENALTY_COST) -> Verdict:
    """Classify one prediction. `opt`: precomputed optimal_cost(maze)."""
    maze_rows = to_rows(maze)
    side = len(maze_rows)
    pred_rows = to_rows(pred, side)
    exact = None
    if label is not None:
        exact = to_rows(label, side) == pred_rows

    reasons: List[Tuple[str, str]] = []

    if len(pred_rows) != side or any(len(p) != len(m) for p, m in zip(pred_rows, maze_rows)):
        return Verdict(BROKEN, [("bad_shape", f"prediction is {len(pred_rows)} rows, maze is {side}")],
                       exact=exact)

    # 1. per-cell consistency
    bad_token, on_wall, altered, wrong_marker = [], [], [], []
    marked = set()
    for r, (mrow, prow) in enumerate(zip(maze_rows, pred_rows)):
        for c, (t, p) in enumerate(zip(mrow, prow)):
            if p not in CHARSET:
                bad_token.append((r, c))
                continue
            if p in _ON_PATH:
                marked.add((r, c))
            if p == t or p == _PATH_MARKER.get(t):
                continue
            if t == "#" and p in _ON_PATH:
                on_wall.append((r, c))
            elif p in "orn" and t in _PATH_MARKER and t in " RN":
                wrong_marker.append((r, c))
            else:
                altered.append((r, c))

    if bad_token:
        reasons.append(("bad_token", f"{len(bad_token)} cells hold PAD/unknown ids at {_fmt(bad_token)}"))
    if on_wall:
        reasons.append(("path_on_wall", f"route crosses {len(on_wall)} wall cells at {_fmt(on_wall)}"))
    if altered:
        detail = [f"({r},{c}) {maze_rows[r][c]!r}->{pred_rows[r][c]!r}" for r, c in altered[:4]]
        more = f" +{len(altered) - 4} more" if len(altered) > 4 else ""
        reasons.append(("tile_altered", f"{len(altered)} cells changed: " + ", ".join(detail) + more))
    if wrong_marker:
        detail = [f"({r},{c}) {maze_rows[r][c]!r}->{pred_rows[r][c]!r}" for r, c in wrong_marker[:4]]
        more = f" +{len(wrong_marker) - 4} more" if len(wrong_marker) > 4 else ""
        reasons.append(("wrong_marker", f"{len(wrong_marker)} path cells mis-marked: " + ", ".join(detail) + more))

    # 2. structure of the marked cells
    starts = [(r, c) for r, row in enumerate(maze_rows) for c, t in enumerate(row) if t == "S"]
    goals = [(r, c) for r, row in enumerate(maze_rows) for c, t in enumerate(row) if t == "G"]
    cps = [(r, c) for r, row in enumerate(maze_rows) for c, t in enumerate(row) if t == "C"]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError(f"maze must have exactly one S and one G (found {len(starts)}, {len(goals)})")
    s, g = starts[0], goals[0]

    if s not in marked:
        reasons.append(("start_missing", f"S at {s} is not marked"))
    if g not in marked:
        reasons.append(("goal_missing", f"G at {g} is not marked"))

    comps = _components(marked)
    route = next((comp for comp in comps if s in comp), set())
    if s in marked and g in marked and g not in route:
        reasons.append(("disconnected", f"S and G are in separate pieces ({len(comps)} marked pieces in total)"))
    missed = [cp for cp in cps if cp not in route]
    if missed and s in marked:
        reasons.append(("missed_checkpoint", f"{len(missed)}/{len(cps)} checkpoints not reached from S: {_fmt(missed)}"))
    stray = sorted(marked - route) if route else []
    if stray and g in route:
        n_pieces = sum(1 for comp in comps if comp is not route)
        reasons.append(("stray_cells", f"{len(stray)} marked cells in {n_pieces} piece(s) off the route, e.g. {_fmt(stray)}"))

    # 3. the route the marks describe
    pred_cost = pred_steps = None
    if s in route and g in route and not missed:
        restricted = [
            "".join(
                ("#" if (r, c) not in route else (t if t in "SGCRN" else " "))
                for c, t in enumerate(row)
            )
            for r, row in enumerate(maze_rows)
        ]
        _, walk, pred_cost = solve_maze(restricted, reward_bonus=reward_bonus, penalty_cost=penalty_cost)
        pred_steps = len(walk) - 1
        unused = sorted(route - set(walk))
        if unused:
            reasons.append(("branch", f"{len(unused)} marked cells joined to the route but not on it "
                                      f"(branch/loop/dead-end), e.g. {_fmt(unused)}"))

    if reasons:
        return Verdict(BROKEN, reasons, pred_cost=pred_cost, pred_steps=pred_steps, exact=exact)

    opt_cost, opt_steps = opt if opt is not None else optimal_cost(maze_rows, reward_bonus, penalty_cost)
    if pred_cost <= opt_cost + _EPS:
        return Verdict(VALID_SHORTEST, [], pred_cost, opt_cost, pred_steps, opt_steps, exact)
    msg = (f"cost {pred_cost:g} vs optimal {opt_cost:g} (+{pred_cost - opt_cost:g}); "
           f"{pred_steps} steps vs {opt_steps}")
    return Verdict(VALID_LONGER, [("longer", msg)], pred_cost, opt_cost, pred_steps, opt_steps, exact)


def summarize(verdicts: Sequence[Verdict]) -> Dict[str, float]:
    """Fractions per category, per primary broken reason, and per any-reason."""
    n = len(verdicts)
    if n == 0:
        return {}
    out: Dict[str, float] = {"n": float(n)}
    cat = Counter(v.category for v in verdicts)
    for k in CATEGORIES:
        out[k] = cat[k] / n
    broken = [v for v in verdicts if v.category == BROKEN]
    prim = Counter(v.primary_reason for v in broken)
    anyr = Counter(code for v in broken for code in {c for c, _ in v.reasons})
    for code in REASON_CODES[:-1]:
        out[f"broken_primary/{code}"] = prim[code] / n
        out[f"broken_any/{code}"] = anyr[code] / n
    with_label = [v for v in verdicts if v.exact is not None]
    if with_label:
        out["exact"] = sum(v.exact for v in with_label) / len(with_label)
        out["shortest_not_exact"] = sum(v.category == VALID_SHORTEST and not v.exact for v in with_label) / len(with_label)
    return out


def _main():
    import argparse
    import json
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("preds", help='torch file with {"inputs", "labels", "preds"} token tensors')
    ap.add_argument("--out", help="write one JSON verdict per maze to this .jsonl")
    ap.add_argument("--only-failures", action="store_true", help="score only mazes where pred != label")
    args = ap.parse_args()

    data = torch.load(args.preds, map_location="cpu")
    inputs, labels, preds = (data[k].numpy() for k in ("inputs", "labels", "preds"))
    keep = (inputs != 0).any(axis=-1)  # skip padding rows
    verdicts, rows = [], []
    for i in np.nonzero(keep)[0]:
        if args.only_failures and np.array_equal(preds[i], labels[i]):
            continue
        v = check(inputs[i], preds[i], labels[i])
        verdicts.append(v)
        rows.append({"index": int(i), **v.to_dict()})
    summary = summarize(verdicts)
    print(json.dumps({k: round(x, 4) for k, x in summary.items() if x}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    _main()
