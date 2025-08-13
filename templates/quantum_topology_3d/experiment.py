import argparse
import json
import os
import os.path as osp
from datetime import datetime

import numpy as np


def set_seed(seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def gen_defects(rng: np.random.Generator, L: int, p: float) -> np.ndarray:
    return (rng.random((L, L, L)) < p).astype(np.uint8)


def label_clusters_3d(grid: np.ndarray) -> np.ndarray:
    Lx, Ly, Lz = grid.shape
    labels = -np.ones_like(grid, dtype=np.int32)
    label = 0
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                if grid[x, y, z] == 1 and labels[x, y, z] == -1:
                    stack = [(x, y, z)]
                    labels[x, y, z] = label
                    while stack:
                        cx, cy, cz = stack.pop()
                        for dx, dy, dz in neighbors:
                            nx, ny, nz = cx + dx, cy + dy, cz + dz
                            if 0 <= nx < Lx and 0 <= ny < Ly and 0 <= nz < Lz:
                                if grid[nx, ny, nz] == 1 and labels[nx, ny, nz] == -1:
                                    labels[nx, ny, nz] = label
                                    stack.append((nx, ny, nz))
                    label += 1
    return labels


def spans_axis(labels: np.ndarray, axis: int) -> bool:
    if labels.size == 0:
        return False
    Lx, Ly, Lz = labels.shape
    max_label = labels.max()
    if max_label < 0:
        return False
    for lab in range(max_label + 1):
        coords = np.argwhere(labels == lab)
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        if axis == 0 and mins[0] == 0 and maxs[0] == Lx - 1:
            return True
        if axis == 1 and mins[1] == 0 and maxs[1] == Ly - 1:
            return True
        if axis == 2 and mins[2] == 0 and maxs[2] == Lz - 1:
            return True
    return False


def logical_error_trial(rng: np.random.Generator, L: int, p: float) -> int:
    grid = gen_defects(rng, L, p)
    labels = label_clusters_3d(grid)
    return int(spans_axis(labels, axis=0) or spans_axis(labels, axis=1) or spans_axis(labels, axis=2))


def estimate_ler(rng: np.random.Generator, L: int, p: float, trials: int) -> float:
    errs = 0
    for _ in range(trials):
        errs += logical_error_trial(rng, L, p)
    return errs / max(1, trials)


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--p_min", type=float, default=0.02)
    parser.add_argument("--p_max", type=float, default=0.14)
    parser.add_argument("--p_steps", type=int, default=7)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    if args.tiny:
        args.L = min(args.L, 6)
        args.p_steps = min(args.p_steps, 4)
        args.trials = min(args.trials, 30)

    rng = set_seed(args.seed)
    ensure_dir(args.out_dir)

    log_path = osp.join(args.out_dir, "log.txt")
    with open(log_path, "a") as log:
        log.write(f"[{datetime.now().isoformat()}] Starting quantum_topology_3d baseline\n")

    ps = np.linspace(args.p_min, args.p_max, args.p_steps).tolist()
    results = []
    for p in ps:
        ler = float(estimate_ler(rng, args.L, float(p), args.trials))
        results.append({"p": float(p), "LER": ler})

    ler_mean = results[-1]["LER"] if len(results) > 0 else 1.0
    accuracy = 1.0 - ler_mean

    metrics = {
        "L": args.L,
        "results": results,
        "summary": {"p_ref": ps[-1] if len(ps) > 0 else None, "LER_ref": ler_mean, "accuracy_ref": accuracy},
    }
    save_json(osp.join(args.out_dir, "metrics.json"), metrics)

    final_info = {"accuracy": {"means": float(accuracy)}}
    save_json(osp.join(args.out_dir, "final_info.json"), final_info)

    sample_grid = gen_defects(rng, args.L, ps[min(1, len(ps) - 1)] if len(ps) > 1 else args.p_min)
    np.save(osp.join(args.out_dir, "sample_grid.npy"), sample_grid)

    with open(log_path, "a") as log:
        log.write(f"[{datetime.now().isoformat()}] Done. Accuracy (1-LER@p_ref)={accuracy:.3f}\n")


if __name__ == "__main__":
    main()
