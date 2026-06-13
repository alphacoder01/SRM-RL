"""Visualize an RL run's metrics.jsonl.

Reads the JSONL log written by GRPOTrainer and saves a multi-panel PNG (and the
key eval curve separately) into the run folder.

Usage:
    python -m src.rl.plot_metrics outputs_rl/ms1000_28/run3
    python -m src.rl.plot_metrics outputs_rl/ms1000_28/run3/metrics.jsonl
    python -m src.rl.plot_metrics <path> --smooth 5 --out custom_dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# (key, human label) panels drawn if the key is present in the train records
TRAIN_PANELS = [
    ("accuracy", "train rollout accuracy"),
    ("reward_mean", "train reward (mean)"),
    ("distance", "train rule distance"),
    ("kl", "KL to reference"),
    ("sigma_aux", "sigma aux (NLL on real data)"),
    ("flow_anchor", "flow anchor loss"),
    ("ratio", "importance ratio"),
    ("clip_frac", "PPO clip fraction"),
    ("pg_loss", "policy-gradient loss"),
    ("advantage_abs", "|advantage|"),
    ("degenerate_group_frac", "degenerate group frac"),
    ("optimizer_steps", "optimizer steps / iter"),
]


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_phase(records: list[dict], phase: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = [r for r in records if r.get("phase") == phase]
    iters = np.array([r["iteration"] for r in rows], dtype=float)
    keys = {k for r in rows for k in r if isinstance(r.get(k), (int, float)) and k != "iteration"}
    series = {
        k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys
    }
    return iters, series


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < 2:
        return y
    window = min(window, y.size)
    kernel = np.ones(window) / window
    # 'same' length via reflect padding to keep endpoints sensible
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="same")[pad : pad + y.size]


def plot_overview(records: list[dict], out_path: Path, window: int) -> None:
    train_it, train = split_phase(records, "train")
    eval_it, eval_ = split_phase(records, "eval")

    panels = [(k, lbl) for k, lbl in TRAIN_PANELS for lbl in [lbl] if k in train]
    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)

    for ax, (key, label) in zip(axes.flat, panels):
        y = train[key]
        ax.plot(train_it, y, color="tab:blue", alpha=0.3, lw=1, label="train")
        if window > 1:
            ax.plot(train_it, smooth(y, window), color="tab:blue", lw=2,
                    label=f"train (smooth {window})")
        # overlay the matching eval series where it exists
        if key in eval_ and eval_it.size:
            ax.plot(eval_it, eval_[key], color="tab:red", marker="o", ms=4,
                    lw=1.5, label="eval")
        ax.set_title(label)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"RL metrics — {out_path.parent.name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_eval_accuracy(records: list[dict], out_path: Path) -> bool:
    eval_it, eval_ = split_phase(records, "eval")
    if not eval_it.size or "accuracy" not in eval_:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eval_it, eval_["accuracy"], color="tab:red", marker="o", lw=2,
            label="eval accuracy")
    if "distance" in eval_:
        ax2 = ax.twinx()
        ax2.plot(eval_it, eval_["distance"], color="tab:gray", marker="s",
                 ms=3, lw=1, alpha=0.6, label="eval distance")
        ax2.set_ylabel("rule distance", color="tab:gray")
    ax.set_xlabel("iteration")
    ax.set_ylabel("accuracy", color="tab:red")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Eval accuracy — {out_path.parent.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def resolve_metrics_path(path: Path) -> Path:
    if path.is_dir():
        return path / "metrics.jsonl"
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an RL run's metrics.jsonl")
    parser.add_argument("path", type=Path,
                        help="run folder or path to metrics.jsonl")
    parser.add_argument("--smooth", type=int, default=5,
                        help="rolling-mean window for noisy train curves (1 disables)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: alongside metrics.jsonl)")
    args = parser.parse_args()

    metrics_path = resolve_metrics_path(args.path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file at {metrics_path}")

    records = load_records(metrics_path)
    if not records:
        raise ValueError(f"{metrics_path} is empty")

    out_dir = args.out or metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = out_dir / "metrics_overview.png"
    plot_overview(records, overview, args.smooth)
    written = [overview]

    eval_plot = out_dir / "metrics_eval_accuracy.png"
    if plot_eval_accuracy(records, eval_plot):
        written.append(eval_plot)

    n_train = sum(r.get("phase") == "train" for r in records)
    n_eval = sum(r.get("phase") == "eval" for r in records)
    print(f"Parsed {len(records)} records ({n_train} train, {n_eval} eval).")
    for p in written:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
