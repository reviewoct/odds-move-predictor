"""Train OddsMoveNet on the synthetic feed and report out-of-sample metrics.

Usage:  python train.py [--steps 200000] [--epochs 3]
Writes: artifacts/model.pt, artifacts/results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oddsnet import OddsMoveNet, SimConfig, WindowSampler, build_features, make_labels, simulate
from oddsnet.features import DOWN, FLAT, UP


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return total


def hit_rate(pred_up: torch.Tensor, labels: torch.Tensor) -> float:
    """Directional hit rate on samples where the price actually moved."""
    moved = labels != FLAT
    return ((pred_up[moved]) == (labels[moved] == UP)).float().mean().item()


@torch.no_grad()
def collect(model, sampler, lo, hi, batch=2048):
    model.eval()
    logits, labels, attn, feats = [], [], [], []
    for s in range(lo, hi, batch):
        idx = torch.arange(s, min(s + batch, hi))
        x, y = sampler.batch(idx)
        z, a = model.encode(x)
        logits.append(model.head(z))  # raw, un-tempered
        labels.append(y)
        attn.append(a.sum(0))
        feats.append(x[:, -8:, :, :])  # recent ticks, for the baselines
    n = hi - lo
    return torch.cat(logits), torch.cat(labels), torch.stack(attn).sum(0) / n, torch.cat(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"simulating {args.steps:,} ticks ...")
    sim = simulate(SimConfig(n_steps=args.steps, seed=args.seed))
    T = args.steps
    train_end, val_end = int(T * 0.70), int(T * 0.85)  # chronological split

    feats, spread_mean = build_features(sim["prices"], fit_end=train_end)
    labels = make_labels(sim["prices"], args.horizon)
    sampler = WindowSampler(feats, labels, args.window, args.horizon)

    # Leave a `horizon` gap before each boundary so labels never peek across it.
    tr = sampler.valid_range(0, train_end - args.horizon)
    va = sampler.valid_range(train_end, val_end - args.horizon)
    te = sampler.valid_range(val_end, T)

    K = sim["prices"].shape[1]
    model = OddsMoveNet(K)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n_train = tr[1] - tr[0]
    steps_per_epoch = n_train // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=args.epochs * steps_per_epoch
    )

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train) + tr[0]
        running = 0.0
        for i in range(steps_per_epoch):
            x, y = sampler.batch(perm[i * args.batch : (i + 1) * args.batch])
            loss = F.cross_entropy(model.head(model.encode(x)[0]).reshape(-1, 3), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        v_logits, v_labels, _, _ = collect(model, sampler, *va)
        v_acc = (v_logits.argmax(-1) == v_labels).float().mean().item()
        print(f"epoch {epoch + 1}: train loss {running / steps_per_epoch:.4f}  val acc {v_acc:.4f}")
    train_seconds = time.time() - t0

    # --- calibration on validation, final scoring on the untouched test split
    v_logits, v_labels, _, _ = collect(model, sampler, *va)
    temperature = model.fit_temperature(v_logits, v_labels)

    logits, y, attn, recent = collect(model, sampler, *te)
    p_raw = F.softmax(logits, -1).reshape(-1, 3)
    p_cal = F.softmax(logits / temperature, -1).reshape(-1, 3)
    y_flat = y.reshape(-1)

    model_up = logits[..., UP] > logits[..., DOWN]

    # Baselines, using the same information the model sees.
    last_ret = torch.zeros_like(recent[:, 0, :, 0])
    for step in range(recent.shape[1]):  # most recent non-zero return
        r = recent[:, step, :, 0]
        last_ret = torch.where(r != 0, r, last_ret)
    momentum_up = last_ret > 0
    gap_up = recent[:, -1, :, 1] < 0  # quoted below consensus -> expect a move up

    per_book = [hit_rate(model_up[:, k], y[:, k]) for k in range(K)]
    results = {
        "ticks": T,
        "test_samples": int(y_flat.numel()),
        "class_balance_test": {
            "down": (y_flat == DOWN).float().mean().item(),
            "flat": (y_flat == FLAT).float().mean().item(),
            "up": (y_flat == UP).float().mean().item(),
        },
        "three_class_accuracy": (p_cal.argmax(-1) == y_flat).float().mean().item(),
        "majority_class_accuracy": max(
            (y_flat == c).float().mean().item() for c in (DOWN, FLAT, UP)
        ),
        "directional_hit_rate": hit_rate(model_up, y),
        "directional_hit_rate_per_book": per_book,
        "baseline_hit_rate_momentum": hit_rate(momentum_up, y),
        "baseline_hit_rate_consensus_gap": hit_rate(gap_up, y),
        "ece_before_temperature": ece(p_raw, y_flat),
        "ece_after_temperature": ece(p_cal, y_flat),
        "temperature": temperature,
        "mean_attention_row_listens_to_col": [[round(v, 3) for v in row] for row in attn.tolist()],
        "train_seconds": round(train_seconds, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    args.out.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_books": K,
            "window": args.window,
            "horizon": args.horizon,
            "spread_mean": spread_mean.tolist(),
        },
        args.out / "model.pt",
    )
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
