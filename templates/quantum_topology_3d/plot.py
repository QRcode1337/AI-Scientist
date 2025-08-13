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


def plot_curve(in_dir: str, out_path: str):
    mpath = osp.join(in_dir, "metrics.json")
    if not osp.exists(mpath):
        return
    with open(mpath, "r") as f:
        payload = json.load(f)
    results = payload.get("results", [])
    L = payload.get("L", None)

    ps = [r["p"] for r in results]
    lers = [r["LER"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(ps, lers, marker="o", color="#1f77b4", linewidth=2, label=f"L={L}")
    ax.set_xlabel("Physical error probability p")
    ax.set_ylabel("Logical error rate (LER)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title("3D Topology: LER vs. p")
    ax.legend(frameon=False)
    if len(ps) > 0:
        ax.annotate(f"LER@p={ps[-1]:.2f}: {lers[-1]:.2f}",
                    xy=(ps[-1], lers[-1]), xytext=(-40, 15),
                    textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#444"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_slices(in_dir: str, out_path: str, max_slices: int = 6):
    gpath = osp.join(in_dir, "sample_grid.npy")
    if not osp.exists(gpath):
        return
    grid = np.load(gpath)
    Lz = grid.shape[2]
    idxs = np.linspace(0, Lz - 1, min(max_slices, Lz)).astype(int).tolist()

    cols = 3
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, z) in enumerate(zip(axes, idxs)):
        ax.imshow(grid[:, :, z].T, origin="lower", cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Slice z={z}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("3D Defect Field Slices (toy proxy)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="run_0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_slices", type=int, default=6)
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir or in_dir
    ensure_dir(out_dir)

    plot_curve(in_dir, osp.join(out_dir, "curve.png"))
    plot_slices(in_dir, osp.join(out_dir, "slices.png"), max_slices=args.max_slices)


if __name__ == "__main__":
    main()
