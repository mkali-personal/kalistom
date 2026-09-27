#!/usr/bin/env python3
"""
The phone's decision view, drawn for a whole recorded session at once.

ScopeView shows the last half-minute live. This shows every frame of a session in the same axes,
which is where it earns its keep: the live dot tells you what is happening, the replay tells you
what the model is *systematically* doing - which clusters it gets right, and which region of the
plot its mistakes come from.

Same axes as the phone, and for the same reasons (see Scope.kt): horizontal is the model's smoothed
score, so the shaded bands are exactly the mute rule rather than an approximation of it, and
vertical is YAMNet's "Music" score, because the model's known weakness is confusing music with
advertising. The x axis is log-odds, or 0.70 and 0.99 would sit on top of each other.

With --truth, points are coloured by the ground-truth label instead of by the mute decision, which
turns the picture into an error map: red on the right of the mute line is content being silenced,
blue on the left is advertising getting through.

    python trainer/scope_replay.py captures/live/sess_20260927_121551
    python trainer/scope_replay.py captures/stitched/run_20260916_073546 --truth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import DROP, HOP_S, POSITIVE, frame_labels, read_points, read_spans   # noqa: E402
from train import LABEL_ORDER                                        # noqa: E402

LIMIT = 7.0


def warp(p):
    q = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.clip(np.log(q / (1 - q)), -LIMIT, LIMIT)


def read_frames(jsonl: Path):
    ad, ads, mus, mute = [], [], [], []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if '"ad"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:                                            # noqa: BLE001
            continue
        ad.append(float(o["ad"]))
        ads.append(float(o.get("ads", o["ad"])))
        mus.append(float(o.get("mus", float("nan"))))
        mute.append(bool(o.get("mute", 0)))
    return (np.array(ad), np.array(ads), np.array(mus), np.array(mute, bool))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="session stem, or any of its files")
    ap.add_argument("--weights", default="head_weights.json")
    ap.add_argument("--truth", action="store_true",
                    help="colour by the hand/draft labels instead of by the mute decision")
    ap.add_argument("--out", default="", help="PNG to write (default: alongside the session)")
    args = ap.parse_args()

    stem = Path(args.session)
    if stem.suffix in (".f16", ".jsonl", ".wav"):
        stem = stem.with_suffix("")
    jsonl = stem.with_suffix(".jsonl")
    if not jsonl.exists():
        print(f"missing {jsonl}")
        return 1

    cfg = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    on, off = float(cfg["on"]), float(cfg["off"])

    ad, ads, mus, mute = read_frames(jsonl)
    if len(ad) == 0:
        print("No 'ad' fields in this session - it predates the classifier build.")
        return 1
    if np.isnan(mus).all():
        print("No 'mus' fields - recorded before the music score was added; y will be flat.")
        mus = np.zeros_like(ad)

    y = None
    if args.truth:
        for suf in LABEL_ORDER:
            lp = stem.with_name(stem.name + suf)
            if lp.exists():
                joins = stem.with_name(stem.name + ".joins.txt")
                y = frame_labels(len(ad), read_spans(lp),
                                 read_points(joins) if joins.exists() else [])
                print(f"labels from {lp.name}")
                break
        if y is None:
            print("--truth asked for but no label file found; colouring by decision instead")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axp = plt.subplots(figsize=(11, 6.5))
    axp.axvspan(-LIMIT, warp(off), color="#EFF3F9", zorder=0)
    axp.axvspan(warp(off), warp(on), color="#FBF1DC", zorder=0)
    axp.axvspan(warp(on), LIMIT, color="#FBE3E3", zorder=0)
    axp.axvline(warp(off), color="#C9A227", lw=1.2, zorder=1)
    axp.axvline(warp(on), color="#C0392B", lw=1.2, zorder=1)

    x = warp(ads)
    if y is not None:
        k = y != DROP
        axp.scatter(x[~k], mus[~k], s=6, c="#CCCCCC", alpha=0.4, lw=0,
                    label="dropped (boundary)", zorder=2)
        isad = k & (y == POSITIVE)
        iscon = k & (y == 0)
        axp.scatter(x[iscon], mus[iscon], s=7, c="#2F6BE0", alpha=0.45, lw=0,
                    label="truth: programme", zorder=3)
        axp.scatter(x[isad], mus[isad], s=7, c="#E5484D", alpha=0.55, lw=0,
                    label="truth: advertising", zorder=4)
        lost = int((iscon & (x > warp(on))).sum()) * HOP_S
        leak = int((isad & (x < warp(off))).sum()) * HOP_S
        axp.set_title(
            f"{stem.name} - coloured by ground truth\n"
            f"blue right of the mute line = programme silenced ({lost:.0f}s); "
            f"red left of the release line = advertising heard ({leak:.0f}s)")
    else:
        axp.scatter(x[~mute], mus[~mute], s=7, c="#2F6BE0", alpha=0.45, lw=0,
                    label="playing", zorder=3)
        axp.scatter(x[mute], mus[mute], s=7, c="#E5484D", alpha=0.6, lw=0,
                    label="muted", zorder=4)
        axp.set_title(f"{stem.name} - {len(ad) * HOP_S / 60:.1f} min, "
                      f"muted {100 * mute.mean():.1f}% of frames")

    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    axp.set_xticks(warp(ticks))
    axp.set_xticklabels([str(t) for t in ticks])
    axp.set_xlim(-LIMIT, LIMIT)
    axp.set_ylim(-0.02, 1.02)
    axp.set_xlabel("model's smoothed ad score  (log-odds spacing)")
    axp.set_ylabel("YAMNet \"Music\" score")
    axp.legend(loc="upper left", framealpha=0.9)
    axp.grid(alpha=0.15)

    out = Path(args.out) if args.out else stem.with_name(stem.name + ".scope.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
