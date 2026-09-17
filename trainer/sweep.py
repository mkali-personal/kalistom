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
from rectime import in_hours, parse_window, started_at     # noqa: E402
from train import EMBEDDING_DIM, HOP_S, sweep               # noqa: E402


def hour_mask(names, groups, dirs, window) -> np.ndarray:
    """Rows belonging to recordings whose midpoint falls inside `window`.

    The model is trained on everything - music-heavy evenings are what teach it that music is not
    advertising - but it is only ever going to run while you are listening. Judging it on the whole
    pool answers a question nobody asked; this narrows the same predictions, with no refitting, to
    the hours that will actually reach your ears.
    """
    lo, hi = parse_window(window)
    keep_group = np.zeros(len(names), dtype=bool)
    unmatched = []
    for i, nm in enumerate(names):
        # Merged folds carry several recordings; the group counts if any member does.
        for member in nm.split(" + "):
            hit = next((d / f"{member}.f16" for d in dirs if (d / f"{member}.f16").exists()), None)
            if hit is None:
                unmatched.append(member)
                continue
            t = started_at(hit)
            if t is None:
                continue
            from datetime import timedelta
            frames = hit.stat().st_size // (EMBEDDING_DIM * 2)
            mid = t + timedelta(seconds=frames * HOP_S / 2)
            if in_hours(mid, lo, hi):
                keep_group[i] = True
    if unmatched:
        print(f"  (no .f16 found for {', '.join(sorted(set(unmatched)))} - treated as outside)")
    return keep_group[groups]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", help=".npz written by train.py --save-oof")
    ap.add_argument("--per-recording", action="store_true",
                    help="sweep each held-out recording separately as well as pooled")
    ap.add_argument("--hours", default="", metavar="HH-HH",
                    help="judge only recordings whose midpoint falls in this window, e.g. 07-11. "
                         "Needs --dir to find the recordings' start times")
    ap.add_argument("--dir", action="append", default=None,
                    help="where the recordings live; repeat for several directories")
    args = ap.parse_args()

    d = np.load(args.oof, allow_pickle=True)
    y, p, groups, names = d["y"], d["p"], d["groups"], d["names"]
    print(f"{len(y)} frames across {len(names)} recording(s)")
    sweep(y, p)

    if args.hours:
        if not args.dir:
            print("\n--hours needs --dir to locate the recordings")
            return 1
        m = hour_mask(names, groups, [Path(d) for d in args.dir], args.hours)
        kept = sorted({names[g] for g in np.unique(groups[m])})
        print(f"\n=== judged on {args.hours} only: {m.sum() * HOP_S / 3600:.2f} h, "
              f"{len(kept)} recording(s) ===")
        for nm in kept:
            print(f"  {nm}")
        if m.sum():
            sweep(y[m], p[m])
        else:
            print("  nothing in that window")

    if args.per_recording:
        for i, nm in enumerate(names):
            m = groups == i
            print(f"\n=== {nm} ===")
            sweep(y[m], p[m])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
