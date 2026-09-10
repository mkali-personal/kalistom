#!/usr/bin/env python3
"""
Re-examines a trained model's operating point without refitting anything.

trainer/train.py --save-oof writes the out-of-fold predictions; every threshold question can then
be answered from that file in a second, rather than by waiting a quarter of an hour for weights
that would come out identical. Separating the two is the point: the model estimates a probability
once, and how aggressively to act on it is a policy decision to be revisited freely.

Usage:
    python trainer/sweep.py captures/stitched/oof.npz
    python trainer/sweep.py oof.npz --per-recording
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import sweep                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", help=".npz written by train.py --save-oof")
    ap.add_argument("--per-recording", action="store_true",
                    help="sweep each held-out recording separately as well as pooled")
    args = ap.parse_args()

    d = np.load(args.oof, allow_pickle=True)
    y, p, groups, names = d["y"], d["p"], d["groups"], d["names"]
    print(f"{len(y)} frames across {len(names)} recording(s)")
    sweep(y, p)

    if args.per_recording:
        for i, nm in enumerate(names):
            m = groups == i
            print(f"\n=== {nm} ===")
            sweep(y[m], p[m])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
