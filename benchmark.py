"""Inference latency / throughput benchmark for a trained OddsMoveNet (CPU).

Usage:  python benchmark.py [--model artifacts/model.pt]
Writes: artifacts/latency.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oddsnet import OddsMoveNet
from oddsnet.features import N_FEATURES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    model = OddsMoveNet(ckpt["n_books"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    torch.set_num_threads(1)  # single core: the conservative number

    K, L = ckpt["n_books"], ckpt["window"]
    out = {"threads": 1, "window": L, "books": K}

    with torch.inference_mode():
        for batch in (1, 64, 512):
            x = torch.randn(batch, L, K, N_FEATURES)
            for _ in range(50):  # warm-up
                model.predict_proba(x)
            iters = args.iters if batch == 1 else max(50, args.iters // batch * 4)
            times = np.empty(iters)
            for i in range(iters):
                t0 = time.perf_counter()
                model.predict_proba(x)
                times[i] = (time.perf_counter() - t0) * 1e3
            out[f"batch_{batch}"] = {
                "p50_ms": round(float(np.percentile(times, 50)), 3),
                "p99_ms": round(float(np.percentile(times, 99)), 3),
                "market_snapshots_per_s": round(batch / (times.mean() / 1e3)),
            }

    (args.model.parent / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
