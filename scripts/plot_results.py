#!/usr/bin/env python3
"""Plot TRM maze fine-tune results for the Thesis B presentation.

Reads the eval-metric history written by scripts/train.py (either the
eval_metrics.jsonl written alongside checkpoints, or a CSV with the same
columns) and emits three figures.

    python scripts/plot_results.py results/v4_eval_metrics.csv --outdir figures

Figures:
  1_learning_curve.png   exact-match vs step -- the headline result
  2_two_phase.png        checkpoint routing saturates ~2000 steps before
                         exact-match does; routing and precision separate
  3_structure.png        connectivity / endpoints / fragment count

Requires: matplotlib, pandas.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Colours chosen to stay distinguishable in greyscale print and for the
# most common forms of colour-vision deficiency.
C_EXACT = "#2a78d6"   # blue   -- exact match
C_ROUTE = "#1baf7a"   # teal   -- checkpoint routing
C_TOKEN = "#eda100"   # amber  -- token accuracy
C_STRUCT = "#e87ba4"  # pink   -- endpoints
C_CONN = "#4a3aa7"    # violet -- connectivity
C_MUTED = "#898781"


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.open() if l.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
    df = df.sort_values("step").reset_index(drop=True)
    missing = {"step", "all/exact_accuracy", "all/accuracy"} - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    return df


def style(ax, xlabel="training step", ylabel=None, pct=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.grid(axis="y", color=C_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=10, color="#333")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#333")
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(-0.03, 1.05)


def fig_learning_curve(df, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(df["step"], df["all/exact_accuracy"], "-o", color=C_EXACT,
            lw=2.2, ms=5, label="exact match (whole grid correct)")
    ax.plot(df["step"], df["all/accuracy"], "--s", color=C_TOKEN,
            lw=1.8, ms=4, label="per-token accuracy")

    first, last = df.iloc[0], df.iloc[-1]
    ax.annotate(f"{first['all/exact_accuracy']:.0%}",
                (first["step"], first["all/exact_accuracy"]),
                textcoords="offset points", xytext=(6, 10),
                fontsize=10, color=C_EXACT)
    best = df.loc[df["all/exact_accuracy"].idxmax()]
    ax.annotate(f"{best['all/exact_accuracy']:.0%}",
                (best["step"], best["all/exact_accuracy"]),
                textcoords="offset points", xytext=(-6, -18),
                fontsize=11, color=C_EXACT, weight="medium")

    style(ax, ylabel="accuracy")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_title("Fine-tuning on objective-conditioned mazes",
                 fontsize=12, color="#222", loc="left", pad=12)
    fig.tight_layout()
    p = outdir / "1_learning_curve.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_two_phase(df, outdir):
    """Routing saturates well before precision does."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    cov = ("path/all_checkpoints" if "path/all_checkpoints" in df
           else "path/checkpoint_coverage")
    ax.plot(df["step"], df[cov], "-o", color=C_ROUTE, lw=2.2, ms=5,
            label="all mandatory checkpoints visited")
    ax.plot(df["step"], df["all/exact_accuracy"], "-o", color=C_EXACT,
            lw=2.2, ms=5, label="exact match")

    # mark where routing first saturates
    sat = df[df[cov] >= 0.999]
    if len(sat):
        s = sat.iloc[0]
        ax.axvline(s["step"], color=C_MUTED, ls=":", lw=1.2)
        ax.annotate(
            f"routing solved at step {int(s['step'])}\n"
            f"exact match still {s['all/exact_accuracy']:.0%}",
            (s["step"], 0.45), textcoords="offset points", xytext=(12, 0),
            fontsize=9, color="#444", va="center")

    style(ax, ylabel="fraction of test mazes")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Routing is learned before precision",
                 fontsize=12, color="#222", loc="left", pad=12)
    fig.tight_layout()
    p = outdir / "2_two_phase.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_structure(df, outdir):
    have = [c for c in ("path/connected", "path/largest_component",
                        "path/has_endpoints") if c in df]
    if not have:
        return None

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                  gridspec_kw={"width_ratios": [1.4, 1]})

    labels = {"path/connected": ("single connected path", C_CONN, "-o"),
              "path/largest_component": ("largest fragment / path", C_TOKEN, "--s"),
              "path/has_endpoints": ("exactly one start and goal", C_STRUCT, "-.^")}
    for c in have:
        lab, col, ls = labels[c]
        ax.plot(df["step"], df[c], ls, color=col, lw=1.9, ms=4, label=lab)
    style(ax, ylabel="fraction of test mazes")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Path structure", fontsize=12, color="#222", loc="left", pad=12)

    if "path/n_components" in df:
        ax2.plot(df["step"], df["path/n_components"], "-o",
                 color=C_MUTED, lw=1.9, ms=4)
        style(ax2, ylabel="mean connected components", pct=False)
        ax2.set_ylim(0, None)
        ax2.set_title("Fragmentation", fontsize=12, color="#222",
                      loc="left", pad=12)
    else:
        ax2.axis("off")

    fig.tight_layout()
    p = outdir / "3_structure.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def summary(df):
    first, last = df.iloc[0], df.iloc[-1]
    best = df.loc[df["all/exact_accuracy"].idxmax()]
    cov = ("path/all_checkpoints" if "path/all_checkpoints" in df
           else "path/checkpoint_coverage")
    lines = [
        f"evals            : {len(df)}  (step {int(first['step'])} -> {int(last['step'])})",
        f"exact match      : {first['all/exact_accuracy']:.1%} -> {last['all/exact_accuracy']:.1%}"
        f"   (best {best['all/exact_accuracy']:.1%} @ step {int(best['step'])})",
        f"token accuracy   : {first['all/accuracy']:.2%} -> {last['all/accuracy']:.2%}",
    ]
    sat = df[df[cov] >= 0.999]
    if len(sat):
        s = sat.iloc[0]
        lines.append(
            f"routing saturated: step {int(s['step'])} "
            f"(exact match was {s['all/exact_accuracy']:.0%} at that point)")
    if "path/legal" in df:
        lines.append(f"legal (no wall)  : min {df['path/legal'].min():.3f} "
                     f"across all evals")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path,
                    help="eval_metrics.jsonl or .csv")
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    a = ap.parse_args()

    df = load(a.metrics)
    a.outdir.mkdir(parents=True, exist_ok=True)

    print(summary(df))
    print()
    for p in (fig_learning_curve(df, a.outdir),
              fig_two_phase(df, a.outdir),
              fig_structure(df, a.outdir)):
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()
