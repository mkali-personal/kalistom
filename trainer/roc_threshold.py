#!/usr/bin/env python3
"""
ROC curves swept over the upper (mute) threshold, before and after hysteresis.

Each point on a curve is one value of `on`. The axes are per frame:

    true-positive rate  = advertising frames muted / advertising frames
    false-positive rate = content frames muted / content frames

Frames labelled DROP (-1) are not counted, but they still pass through the hysteresis state
machine, because the phone sees them too.

Two plots:

  single window   muted = p >= on. Each 0.48 s frame judged alone - the head's own ROC.
  hysteresis      mute when p reaches `on`, release when p falls below `off`, with `off` held
                  fixed (by default the shipped value in head_weights.json). Where the sweep takes
                  `on` below `off`, `off` follows it down, so the rule degenerates to the single
                  window and the curve still spans the whole range. State resets at the start of
                  every recording, as it does when a phone session opens.

The false-positive axis is logarithmic, because the operating points that matter sit at a
fraction of a percent; a linear axis squeezes all of them into the corner.

Usage:
    python trainer/roc_threshold.py captures/heads/oof_k40_mlp16.npz
    python trainer/roc_threshold.py captures/heads/oof_k40_mlp16.npz \\
        --dir captures/stitched --dir captures/sessions --hours 07-11 --off 0.2 --show
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import NEGATIVE, POSITIVE                                  # noqa: E402

HEAD_WEIGHTS = Path(__file__).resolve().parent.parent / "app/src/main/assets/head_weights.json"
MARKED = (0.5, 0.9, 0.99, 0.999, 0.9999)


def recording_of_row(d) -> np.ndarray:
    """Which recording each row came from. Hysteresis resets per recording, as a phone session
    does; `groups` is not enough, because two recordings of one broadcast share a group."""
    if "rec_bounds" not in d.files:
        return d["groups"]
    rec = np.empty(len(d["y"]), dtype=np.int32)
    for i, (a, b) in enumerate(d["rec_bounds"]):
        rec[a:b] = i
    return rec


def segment_floor(groups: np.ndarray) -> np.ndarray:
    """For each frame, the index just before its recording starts."""
    n = len(groups)
    starts = np.concatenate(([True], groups[1:] != groups[:-1]))
    start_idx = np.where(starts, np.arange(n), 0)
    return np.maximum.accumulate(start_idx) - 1


def hysteresis(p: np.ndarray, floor: np.ndarray, on: float, off: float) -> np.ndarray:
    """smooth.hysteresis, vectorised and reset per recording.

    Muted at frame i exactly when the latest frame at or above `on` is more recent than the latest
    frame below `off`. With off <= on no frame can be both, so the two never tie. A recording's
    start acts as a frame below `off`, so each one opens unmuted.
    """
    idx = np.arange(len(p))
    last_on = np.maximum.accumulate(np.where(p >= on, idx, -1))
    last_off = np.maximum(np.maximum.accumulate(np.where(p < off, idx, -1)), floor)
    return last_on > last_off


def rates(y: np.ndarray, muted: np.ndarray) -> tuple[float, float]:
    pos, neg = y == POSITIVE, y == NEGATIVE
    return muted[pos].mean(), muted[neg].mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", type=Path, help=".npz written by train.py --save-oof")
    ap.add_argument("--off", type=float, default=None,
                    help="release threshold for the hysteresis plot (default: head_weights.json)")
    ap.add_argument("--dir", action="append", default=None,
                    help="where the recordings live, needed only with --hours; repeatable")
    ap.add_argument("--hours", default="", metavar="HH-HH",
                    help="judge only recordings whose midpoint falls in this window")
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default beside the .npz)")
    ap.add_argument("--show", action="store_true", help="also open the figure in a window")
    args = ap.parse_args()

    shipped = json.loads(HEAD_WEIGHTS.read_text(encoding="utf-8"))
    off = args.off if args.off is not None else float(shipped["off"])
    shipped_on = float(shipped["on"])

    d = np.load(args.oof, allow_pickle=True)
    y, p, groups = d["y"], d["p"].astype(np.float64), d["groups"]
    rec = recording_of_row(d)
    # A quick run (train.py --test-newest-hours) scores only its test days.
    scored = d["scored"] if "scored" in d.files else np.ones(len(y), bool)
    if args.hours:
        from sweep import hour_mask
        if not args.dir:
            ap.error("--hours needs --dir to find the recordings' start times")
        m = hour_mask(d["names"], groups, [Path(x) for x in args.dir], args.hours)
        scored = scored & m
    y, p, rec = y[scored], p[scored], rec[scored]
    floor = segment_floor(rec)
    hours = len(y) * 0.48 / 3600

    # Even steps in log-odds, which spends points where the scores actually crowd: near 1.
    grid = 1.0 / (1.0 + np.exp(-np.linspace(-10.0, 16.0, 400)))
    grid = np.unique(np.concatenate((grid, MARKED, [shipped_on])))

    single = np.array([rates(y, p >= t) for t in grid])
    hyst = np.array([rates(y, hysteresis(p, floor, t, min(off, t))) for t in grid])

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    panels = ((axes[0], single, "Single window: mute when p ≥ on"),
              (axes[1], hyst, f"After hysteresis: mute at p ≥ on, release below off = {off:g}"))
    for ax, r, title in panels:
        tpr, fpr = r[:, 0], r[:, 1]
        order = np.argsort(fpr, kind="stable")
        auc = np.trapezoid(np.concatenate(([0], tpr[order], [1])),
                           np.concatenate(([0], fpr[order], [1])))
        ax.plot(fpr, tpr, color="#2a6fdb", lw=2)
        for t in MARKED:
            i = int(np.flatnonzero(grid == t)[0])
            ax.plot(fpr[i], tpr[i], "o", color="#2a6fdb", ms=5)
            ax.annotate(f"{t:g}", (fpr[i], tpr[i]), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color="#333")
        i = int(np.flatnonzero(grid == shipped_on)[0])
        ax.plot(fpr[i], tpr[i], "*", color="#e8590c", ms=15, zorder=5,
                label=f"shipped on = {shipped_on:g}: TPR {tpr[i]:.3f}, FPR {fpr[i]:.4f}")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("false-positive rate (content frames muted), log scale")
        ax.set_title(f"{title}\nAUC {auc:.4f}", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("true-positive rate (advertising frames muted)")
    scope = f", {args.hours}" if args.hours else ""
    fig.suptitle(f"{args.oof.name}: {hours:.1f} h{scope}, "
                 f"{int((y == POSITIVE).sum())} ad frames, {int((y == NEGATIVE).sum())} content "
                 f"frames. Labels on the dots are values of the upper threshold.", fontsize=10)
    fig.tight_layout()

    out = args.out or args.oof.with_name(f"roc_{args.oof.stem}.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for name, r in (("single window", single), ("hysteresis", hyst)):
        i = int(np.flatnonzero(grid == shipped_on)[0])
        print(f"  {name:14s} at on={shipped_on:g}: TPR {r[i, 0]:.4f}  FPR {r[i, 1]:.5f}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
