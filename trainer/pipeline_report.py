#!/usr/bin/env python3
"""
Precision and recall of the WHOLE pipeline - model plus smoothing plus thresholds.

Every precision/recall figure reported before this was the bare head at p >= 0.5: what the
classifier believes about one 0.48 s window, judged in isolation. That is a fact about the model,
not about the app. What the ear experiences is the mute decision, after the scores have been
averaged and run through two thresholds, and those are different numbers - the smoothing is there
precisely to change them.

Reported both ways, because they answer different questions:

  frame level   of the seconds actually muted, how many were advertising (precision), and of the
                advertising, how much got muted (recall). This is what you hear.
  break level   of the ad breaks, how many were caught at all, and how much of each was covered.
                A break caught 6 s late is a success here and a partial failure above.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import DROP, HOP_S, POSITIVE                             # noqa: E402
from smooth import decompose, hysteresis, rolling                    # noqa: E402
from sweep import hour_mask                                          # noqa: E402


def prf(y, muted):
    """Precision, recall and F1 over the frames that carry a usable label."""
    k = y != DROP
    yt, mt = y[k] == POSITIVE, muted[k]
    tp = int((mt & yt).sum())
    fp = int((mt & ~yt).sum())
    fn = int((~mt & yt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9), tp, fp, fn


def breaks(y, muted):
    """Per-break coverage: caught at all, and what fraction of each break was silenced."""
    d = np.diff(np.concatenate(([0], (y == POSITIVE).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    cov = np.array([muted[a:b].mean() for a, b in zip(starts, ends)]) if len(starts) else np.array([])
    return len(starts), int((cov > 0).sum()), cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", help=".npz written by train.py --save-oof")
    ap.add_argument("--dir", action="append", default=None)
    ap.add_argument("--hours", default="", metavar="HH-HH")
    ap.add_argument("--weights", default="head_weights.json")
    args = ap.parse_args()

    cfg = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    n = int(cfg.get("smoothing", {}).get("frames", 1))
    on, off = float(cfg["on"]), float(cfg["off"])

    d = np.load(args.oof, allow_pickle=True)
    y, p, groups, names = d["y"], d["p"], d["groups"], d["names"]
    if args.hours:
        m = hour_mask(names, groups, [Path(x) for x in args.dir], args.hours)
        y, p = y[m], p[m]
    hours = len(y) * HOP_S / 3600
    base = int((y == POSITIVE).sum()) * HOP_S / hours

    print(f"{hours:.2f} h of held-out audio"
          f"{' from ' + args.hours if args.hours else ''}, "
          f"{base:.0f} ad-sec/h if nothing is done\n")

    variants = [
        ("raw model, no smoothing, p>=0.5", p >= 0.5),
        ("raw model, thresholds only (on=0.999 off=0.50)", hysteresis(p, 0.999, 0.50)),
        (f"SHIPPING: mean of {n} frames ({n * HOP_S:.1f}s), on={on} off={off}",
         rolling(p, n, on, off)),
    ]
    for nm, muted in variants:
        prec, rec, f1, tp, fp, fn = prf(y, muted)
        nb, caught, cov = breaks(y, muted)
        c = decompose(y, muted, hours)
        heard = int(((y == POSITIVE) & ~muted).sum()) * HOP_S / hours
        lost = int(((y == 0) & muted).sum()) * HOP_S / hours
        print(nm)
        print(f"  frame level   precision {100 * prec:5.1f}%   recall {100 * rec:5.1f}%   "
              f"F1 {100 * f1:5.1f}%")
        print(f"  breaks        {caught} of {nb} caught, median coverage "
              f"{100 * np.median(cov):.0f}% of each break")
        print(f"  what you hear {heard:5.0f} ad-sec/h  (of {base:.0f})")
        print(f"  what you lose {lost:5.0f} content-sec/h, {c['toggles']:.0f} mute toggles/h, "
              f"median {c['median_latency_s']:.1f}s late")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
