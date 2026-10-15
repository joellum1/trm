"""Build the derived custom-maze splits.

  data/maze-custom-eval1k/test      the 500 test mazes + 500 new ones not in train/test
  data/maze-custom-train500/train   first 500 training mazes (mazes are i.i.d.)

    python scripts/build_maze_splits.py
"""

import os
import sys
import csv
import random
import argparse
import json
import shutil

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# generator.py imports its solver as a top-level module ("from solve import ...")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'trm', 'data', 'custom')))

import generator  # noqa: E402
from solve import solve_maze, DEFAULT_REWARD_BONUS, DEFAULT_PENALTY_COST  # noqa: E402
from trm.data.custom.build_maze_dataset import DataProcessConfig, convert_subset  # noqa: E402


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return list(reader)


def build_train_subset(src: str, dst: str, n: int):
    """First n puzzles of src/train (one example per puzzle and group)."""
    with open(os.path.join(src, "train", "dataset.json")) as f:
        meta = json.load(f)
    assert meta["mean_puzzle_examples"] == 1, "prefix slicing assumes one example per puzzle"
    out = os.path.join(dst, "train")
    os.makedirs(out, exist_ok=True)
    load = lambda k: np.load(os.path.join(src, "train", f"all__{k}.npy"))
    np.save(os.path.join(out, "all__inputs.npy"), load("inputs")[:n])
    np.save(os.path.join(out, "all__labels.npy"), load("labels")[:n])
    np.save(os.path.join(out, "all__puzzle_identifiers.npy"), load("puzzle_identifiers")[:n])
    np.save(os.path.join(out, "all__puzzle_indices.npy"), load("puzzle_indices")[:n + 1])
    np.save(os.path.join(out, "all__group_indices.npy"), load("group_indices")[:n + 1])
    meta.update(total_groups=n, total_puzzles=n)
    with open(os.path.join(out, "dataset.json"), "w") as f:
        json.dump(meta, f)
    shutil.copy(os.path.join(src, "identifiers.json"), dst)
    print(f"{n} training mazes -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/maze-custom-raw")
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output-dir", default="data/maze-custom-eval1k")
    p.add_argument("--train-src", default="data/maze-custom")
    p.add_argument("--train-subset", type=int, default=500)
    args = p.parse_args()

    train = read_rows(os.path.join(args.raw_dir, "train.csv"))
    test = read_rows(os.path.join(args.raw_dir, "test.csv"))
    seen = {r[1] for r in train} | {r[1] for r in test}

    rows = [[i, q, a, rating] for i, (_, q, a, rating) in enumerate(test)]
    random.seed(args.seed)
    while len(rows) < args.size:
        maze = generator.place_objectives(generator.generate_maze())
        try:
            solution, _, _ = solve_maze(maze, reward_bonus=DEFAULT_REWARD_BONUS, penalty_cost=DEFAULT_PENALTY_COST)
        except Exception:
            continue
        q = "\n".join("".join(r) for r in maze)
        if q in seen:
            continue
        seen.add(q)
        rows.append([len(rows), q, "\n".join(solution), "custom"])

    csv_path = os.path.join(args.raw_dir, f"eval{args.size // 1000}k.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "q", "a", "r"])
        w.writerows(rows)

    convert_subset("test", csv_path, DataProcessConfig(output_dir=args.output_dir))
    print(f"{len(rows)} mazes ({len(test)} from test.csv, {len(rows) - len(test)} new) -> {args.output_dir}/test")

    build_train_subset(args.train_src, f"{args.train_src}-train{args.train_subset}", args.train_subset)


if __name__ == "__main__":
    main()
