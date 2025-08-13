

import argparse
import json
import os
import os.path as osp
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_poisson_spikes(
    n_samples: int,
    n_classes: int,
    n_channels: int,
    time_bins: int,
    base_rate_hz: float,
    class_rate_delta_hz: float,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    dt = 1e-3
    class_channel_weights = []
    for c in range(n_classes):
        pos = (c + 1) / (n_classes + 1)
        centers = np.linspace(0, 1, n_channels)
        weight = np.exp(-0.5 * ((centers - pos) / 0.2) ** 2)
        weight = 0.7 * weight + 0.3 * rng.uniform(0.8, 1.2, size=n_channels)
        class_channel_weights.append(weight.astype(np.float32))
    class_channel_weights = np.stack(class_channel_weights, axis=0)  # [C, Ch]

    t = np.linspace(0, 1, time_bins, dtype=np.float32)
    class_time_weights = []
    freqs = np.linspace(1.0, 2.5, n_classes)
    phases = np.linspace(0.0, np.pi / 2, n_classes)
    for f, ph in zip(freqs, phases):
        w = 0.5 * (1.0 + np.cos(2 * np.pi * f * t + ph))
        w = 0.7 * w + 0.3
        class_time_weights.append(w.astype(np.float32))
    class_time_weights = np.stack(class_time_weights, axis=0)  # [C, T]

    labels = rng.integers(0, n_classes, size=(n_samples,), dtype=np.int64)
    spikes = np.zeros((n_samples, n_channels, time_bins), dtype=np.uint8)

    for i in range(n_samples):
        c = labels[i]
        rate_hz = (
            base_rate_hz
            + class_rate_delta_hz * class_channel_weights[c][:, None] * class_time_weights[c][None, :]
        ).astype(np.float32)
        p = 1.0 - np.exp(-rate_hz * dt)
        p = np.clip(p, 0.0, 1.0)
        spikes[i] = rng.binomial(1, p, size=(n_channels, time_bins)).astype(np.uint8)

    return spikes, labels


class SimpleSpikeDecoder(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def make_features(spikes: np.ndarray, agg: str = "sum", num_bins: int = 1):
    if agg == "binned" and num_bins > 1:
        N, Ch, T = spikes.shape
        splits = np.array_split(np.arange(T), num_bins)
        feats = []
        for idxs in splits:
            feats.append(spikes[:, :, idxs].sum(axis=2))
        features = np.concatenate(feats, axis=1)
    elif agg == "sum":
        features = spikes.sum(axis=2)
    elif agg == "mean":
        features = spikes.mean(axis=2)
    else:
        N, Ch, T = spikes.shape
        features = spikes.reshape(N, Ch * T)
    return features.astype(np.float32)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean().item()


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def train_and_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    n_classes: int,
    epochs: int = 10,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
):
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)
    yte_t = torch.from_numpy(yte).to(device)

    model = SimpleSpikeDecoder(Xtr.shape[1], n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = criterion(logits, ytr_t)
        loss.backward()
        opt.step()

        tr_acc = accuracy(logits, ytr_t)
        model.eval()
        with torch.no_grad():
            val_logits = model(Xte_t)
            va_acc = accuracy(val_logits, yte_t)

        history["train_loss"].append(float(loss.item()))
        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(va_acc))

    model.eval()
    with torch.no_grad():
        tr_logits = model(Xtr_t)
        te_logits = model(Xte_t)
        tr_acc = accuracy(tr_logits, ytr_t)
        te_acc = accuracy(te_logits, yte_t)

    metrics = {
        "train_loss": float(np.mean(history["train_loss"])),
        "train_acc": float(tr_acc),
        "accuracy": float(te_acc),
    }
    return metrics, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--n_channels", type=int, default=64)
    parser.add_argument("--time_bins", type=int, default=200)
    parser.add_argument("--train_samples", type=int, default=600)
    parser.add_argument("--test_samples", type=int, default=300)
    parser.add_argument("--base_rate_hz", type=float, default=5.0)
    parser.add_argument("--class_rate_delta_hz", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agg", type=str, default="sum", choices=["sum", "mean", "flatten", "binned"])
    parser.add_argument("--num_bins", type=int, default=1)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    if args.tiny:
        args.n_channels = min(args.n_channels, 16)
        args.time_bins = min(args.time_bins, 60)
        args.train_samples = min(args.train_samples, 120)
        args.test_samples = min(args.test_samples, 60)
        args.epochs = min(args.epochs, 3)

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    log_path = osp.join(args.out_dir, "log.txt")
    with open(log_path, "a") as log:
        log.write(f"[{datetime.now().isoformat()}] Starting neuromorphic_bci baseline\n")

    spikes_tr, ytr = generate_poisson_spikes(
        n_samples=args.train_samples,
        n_classes=args.n_classes,
        n_channels=args.n_channels,
        time_bins=args.time_bins,
        base_rate_hz=args.base_rate_hz,
        class_rate_delta_hz=args.class_rate_delta_hz,
        seed=args.seed,
    )
    spikes_te, yte = generate_poisson_spikes(
        n_samples=args.test_samples,
        n_classes=args.n_classes,
        n_channels=args.n_channels,
        time_bins=args.time_bins,
        base_rate_hz=args.base_rate_hz,
        class_rate_delta_hz=args.class_rate_delta_hz,
        seed=args.seed + 1,
    )

    np.save(osp.join(args.out_dir, "spikes_train.npy"), spikes_tr[: min(64, len(spikes_tr))])
    np.save(osp.join(args.out_dir, "labels_train.npy"), ytr[: min(64, len(ytr))])
    np.save(osp.join(args.out_dir, "spikes_test.npy"), spikes_te[: min(64, len(spikes_te))])
    np.save(osp.join(args.out_dir, "labels_test.npy"), yte[: min(64, len(yte))])

    Xtr = make_features(spikes_tr, agg=args.agg, num_bins=args.num_bins)
    Xte = make_features(spikes_te, agg=args.agg, num_bins=args.num_bins)

    metrics, history = train_and_eval(
        Xtr, ytr, Xte, yte, n_classes=args.n_classes, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
    )

    save_json(osp.join(args.out_dir, "metrics.json"), {"metrics": metrics, "history": history})

    final_info = {
        "accuracy": {"means": float(metrics["accuracy"])},
        "train_loss": {"means": float(metrics["train_loss"])},
    }
    save_json(osp.join(args.out_dir, "final_info.json"), final_info)

    with open(log_path, "a") as log:
        log.write(f"[{datetime.now().isoformat()}] Done. Test accuracy={metrics['accuracy']:.3f}\n")


if __name__ == "__main__":
    main()
