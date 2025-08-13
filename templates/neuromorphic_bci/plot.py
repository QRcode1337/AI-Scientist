

import argparse
import os
import os.path as osp
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 13
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["figure.titlesize"] = 14


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_subset(in_dir: str):
    st = np.load(osp.join(in_dir, "spikes_train.npy"))
    lt = np.load(osp.join(in_dir, "labels_train.npy"))
    return st, lt


def plot_raster(
    spikes: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    n_examples: int = 6,
    max_channels_display: int = 32,
    line_color: str = "#1f77b4",
):
    n = min(n_examples, spikes.shape[0])
    ch = min(max_channels_display, spikes.shape[1])
    t = spikes.shape[2]
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.5 * n + 1.0), sharex=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        s = spikes[i][:ch]  # [Ch, T]
        y = labels[i]
        for c in range(ch):
            times = np.where(s[c] > 0)[0]
            ax.vlines(times, c + 0.1, c + 0.9, color=line_color, linewidth=0.5)
        ax.set_ylim(0, ch + 1)
        ax.set_xlim(0, t)
        ax.set_ylabel(f"Channels (y={int(y)})")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Time (bins)")
    fig.suptitle("Neuromorphic BCI: Example Spike Rasters")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metrics(in_dir: str, out_path: str):
    metrics_path = osp.join(in_dir, "metrics.json")
    final_path = osp.join(in_dir, "final_info.json")
    if not osp.exists(metrics_path):
        return
    with open(metrics_path, "r") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    history = payload.get("history", {})

    acc_mean = None
    if osp.exists(final_path):
        with open(final_path, "r") as f:
            fin = json.load(f)
        acc_mean = fin.get("accuracy", {}).get("means", None)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    tr_loss = history.get("train_loss", [])
    ax[0].plot(tr_loss, label="Train loss", color="#d62728", linewidth=2)
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(alpha=0.3)
    if len(tr_loss) > 0:
        ax[0].annotate(f"Final: {tr_loss[-1]:.3f}", xy=(len(tr_loss) - 1, tr_loss[-1]), xytext=(-40, 15),
                       textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#444"))

    val_acc = history.get("val_acc", [])
    ax[1].plot(val_acc, label="Validation accuracy", color="#2ca02c", linewidth=2)
    ax[1].set_title("Validation Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].grid(alpha=0.3)
    if len(val_acc) > 0:
        ax[1].annotate(f"Final: {val_acc[-1]:.3f}", xy=(len(val_acc) - 1, val_acc[-1]), xytext=(-40, 15),
                       textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#444"))

    subtitle = "Neuromorphic BCI: Training Curves"
    if acc_mean is not None:
        subtitle += f"  |  Baseline accuracy: {acc_mean:.3f}"
    fig.suptitle(subtitle)
    handles = []
    labels = []
    for a in ax:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if len(handles) > 0:
        fig.legend(handles, labels, loc="lower center", ncols=2, frameon=False)
        fig.subplots_adjust(bottom=0.18)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="run_0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--max_channels_display", type=int, default=32)
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir or in_dir
    ensure_dir(out_dir)

    spikes, labels = load_subset(in_dir)
    plot_raster(
        spikes,
        labels,
        osp.join(out_dir, "raster.png"),
        n_examples=args.n_examples,
        max_channels_display=args.max_channels_display,
    )
    plot_metrics(in_dir, osp.join(out_dir, "metrics.png"))


if __name__ == "__main__":
    main()
