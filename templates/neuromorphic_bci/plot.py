import argparse
import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import json


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_subset(in_dir: str):
    st = np.load(osp.join(in_dir, "spikes_train.npy"))
    lt = np.load(osp.join(in_dir, "labels_train.npy"))
    return st, lt


def plot_raster(spikes: np.ndarray, labels: np.ndarray, out_path: str, n_examples: int = 6):
    n = min(n_examples, spikes.shape[0])
    ch = spikes.shape[1]
    t = spikes.shape[2]
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        s = spikes[i]  # [Ch, T]
        y = labels[i]
        for c in range(ch):
            times = np.where(s[c] > 0)[0]
            ax.vlines(times, c + 0.1, c + 0.9, color="k", linewidth=0.5)
        ax.set_ylim(0, ch + 1)
        ax.set_xlim(0, t)
        ax.set_ylabel(f"Ch\n(y={int(y)})", rotation=0, labelpad=20, va="center")
        ax.grid(alpha=0.1)
    axes[-1].set_xlabel("Time bin")
    fig.suptitle("Neuromorphic BCI: Spike Raster (subset)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics(in_dir: str, out_path: str):
    metrics_path = osp.join(in_dir, "metrics.json")
    if not osp.exists(metrics_path):
        return
    with open(metrics_path, "r") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    history = payload.get("history", {})

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    tr_loss = history.get("train_loss", [])
    ax[0].plot(tr_loss, label="train_loss", color="tab:blue")
    ax[0].set_title(f"Train Loss (final={metrics.get('train_loss', float('nan')):.3f})")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(alpha=0.3)

    val_acc = history.get("val_acc", [])
    ax[1].plot(val_acc, label="val_acc", color="tab:green")
    ax[1].set_title(f"Val Acc (final={val_acc[-1] if len(val_acc)>0 else float('nan'):.3f})")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Neuromorphic BCI: Training Curves")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="run_0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_examples", type=int, default=6)
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir or in_dir
    ensure_dir(out_dir)

    spikes, labels = load_subset(in_dir)
    plot_raster(spikes, labels, osp.join(out_dir, "raster.png"), n_examples=args.n_examples)
    plot_metrics(in_dir, osp.join(out_dir, "metrics.png"))


if __name__ == "__main__":
    main()
