#!/usr/bin/env python3
"""
Visualise TRM Maze predictions.

Produces figures of the form

+---------+------------+--------------+------------+
|  Input  | Prediction | Ground Truth | Difference |
+---------+------------+--------------+------------+

Difference colours:
    Green  = correctly predicted path
    Red    = predicted path that should not exist
    Yellow = missed path
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

GRID_SIZE = 30

# Dataset IDs -- must match CHARSET order in build_maze_dataset.py
# ("# SGoCRrNn"). The first 5 match the original sapientinc charset.
PAD = 0
WALL = 1
EMPTY = 2
START = 3
GOAL = 4
PATH = 5          # 'o' -- empty tile that was crossed
CHECKPOINT = 6     # 'C' -- mandatory tile, always crossed
REWARD = 7         # 'R' -- optional bonus tile, not crossed
REWARD_PATH = 8    # 'r' -- optional bonus tile, crossed
PENALTY = 9        # 'N' -- optional penalty tile, not crossed
PENALTY_PATH = 10  # 'n' -- optional penalty tile, crossed

# Any of these IDs mean "this cell is part of the solved route"
PATH_IDS = {PATH, REWARD_PATH, PENALTY_PATH}

ID_COLOUR = {
    WALL: "black",
    EMPTY: "white",
    PAD: "white",
    START: "limegreen",
    GOAL: "red",
    PATH: "dodgerblue",
    CHECKPOINT: "gold",
    REWARD: "palegreen",
    REWARD_PATH: "darkgreen",
    PENALTY: "orchid",
    PENALTY_PATH: "indigo",
}


def draw_maze(ax, grid, title):
    """Draw one maze."""

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):

            value = int(grid[r, c])
            colour = ID_COLOUR.get(value, "magenta")

            ax.add_patch(
                patches.Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=colour,
                    edgecolor="lightgrey",
                    linewidth=0.15,
                )
            )


def draw_difference(ax, pred, label):
    """
    Difference map.

    Green  = correct path
    Red    = false positive
    Yellow = false negative

    Applies to any path-bearing tile (plain path, or a crossed reward/penalty
    tile), not just the original plain-path case.
    """

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Difference", fontsize=13)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):

            gt = int(label[r, c])
            pd = int(pred[r, c])

            # Draw maze background first (static tiles keep their identity;
            # checkpoints/rewards/penalties render at their "base" colour)

            if gt == WALL:
                colour = "black"
            elif gt == START:
                colour = "limegreen"
            elif gt == GOAL:
                colour = "red"
            elif gt == CHECKPOINT:
                colour = "gold"
            elif gt in (REWARD, REWARD_PATH):
                colour = "palegreen"
            elif gt in (PENALTY, PENALTY_PATH):
                colour = "orchid"
            else:
                colour = "white"

            # Overlay prediction correctness for path-bearing tiles

            gt_is_path = gt in PATH_IDS
            pd_is_path = pd in PATH_IDS

            if gt_is_path and pd_is_path:
                colour = "lime"
            elif not gt_is_path and pd_is_path:
                colour = "red"
            elif gt_is_path and not pd_is_path:
                colour = "yellow"

            ax.add_patch(
                patches.Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=colour,
                    edgecolor="lightgrey",
                    linewidth=0.15,
                )
            )


def exact_accuracy(pred, label):
    return np.array_equal(pred, label)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preds",
        required=True,
        help="Prediction file produced by run_eval_only.py",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output_dir",
        default="results/maze",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading predictions...")

    data = torch.load(args.preds, map_location="cpu")

    inputs = data["inputs"].numpy().reshape(-1, GRID_SIZE, GRID_SIZE)
    labels = data["labels"].numpy().reshape(-1, GRID_SIZE, GRID_SIZE)
    preds = data["preds"].numpy().reshape(-1, GRID_SIZE, GRID_SIZE)

    total = len(inputs)

    print(f"Loaded {total} maze predictions.")

    rng = np.random.default_rng(args.random_seed)

    indices = rng.choice(
        total,
        size=min(args.num_samples, total),
        replace=False,
    )

    for idx in indices:

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4),
        )

        draw_maze(axes[0], inputs[idx], "Input")
        draw_maze(axes[1], preds[idx], "Prediction")
        draw_maze(axes[2], labels[idx], "Ground Truth")
        draw_difference(axes[3], preds[idx], labels[idx])

        fig.suptitle(
            f"Maze {idx}   |   Exact Match = {exact_accuracy(preds[idx], labels[idx])}",
            fontsize=15,
        )

        plt.tight_layout()

        save_path = os.path.join(
            args.output_dir,
            f"maze_{idx:04d}.png",
        )

        plt.savefig(save_path, dpi=250)
        plt.close(fig)

        print(f"Saved {save_path}")

    print()
    print(f"Finished! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()