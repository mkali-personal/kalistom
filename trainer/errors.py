#!/usr/bin/env python3
"""
Turns a model's mistakes into Audacity label tracks, so they can be listened to.

Precision and recall say how much the model got wrong. They do not say *what*, and on this project
that is the more urgent question: every label in the training set was drawn by a model reading a
transcript, and their error has never been measured. A "false positive" may be the model hearing
advertising that my label track failed to mark - in which case the model is right and the data is
wrong, and no amount of model work will fix the number.

Reads the out-of-fold predictions saved by `train.py --save-oof`, so these are the model's errors
on recordings it never trained on, and writes two label tracks per recording:

    <stem>.fp.txt   marked as advertising, labelled as content
    <stem>.fn.txt   labelled as advertising, not marked

Load the audio, the ground-truth track (.truth.txt / .claude.txt / .gemini.txt) and these two, and
each mistake is a region you can select and play.

WHAT THE LABEL TEXT TELLS YOU. Each region carries its length, the model's mean confidence across
it, and its distance to the nearest true advertising. That last one separates the two failures that
look identical in a score and need opposite responses:

    "adj"     the error touches a real break, so this is a boundary drawn a few seconds out -
              a labelling or hysteresis question, not a detection failure
    "+42s"    the error stands alone, 42 seconds from any advertising - the model genuinely
              heard something that is not there, or heard through something that is

Frames excluded from training are excluded here too. A frame straddling a label boundary has no
correct answer and was dropped when the labels were built; counting it as an error now would
manufacture mistakes that were never made.

Usage:
    python trainer/errors.py captures/stitched/oof.npz --dir captures/stitched --dir captures/sessions
    python trainer/errors.py OOF.npz --dir DIR --on 0.9 --off 0.5 --min-seconds 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import DROP, HOP_S, POSITIVE, WINDOW_S                      # noqa: E402
from train import policy                                               # noqa: E402


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True stretches as (first frame, one past last)."""
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def span_seconds(a: int, b: int) -> tuple[float, float]:
    """Frames [a, b) as a time region. Frame k covers [k*HOP, k*HOP + WINDOW), so the region ends
    one window after the last frame starts, not one hop."""
    return a * HOP_S, (b - 1) * HOP_S + WINDOW_S


def distance_to_truth(a: int, b: int, truth: np.ndarray) -> float:
    """Seconds from this stretch of frames to the nearest frame labelled as advertising, or 0 if
    it touches one. This is what distinguishes a boundary that drifted from a detection that is
    simply wrong."""
    idx = np.flatnonzero(truth == POSITIVE)
    if idx.size == 0:
        return float("inf")
    before = idx[idx < a]
    after = idx[idx >= b]
    gap = min(
        (a - before[-1]) if before.size else np.inf,
        (after[0] - (b - 1)) if after.size else np.inf,
    )
    return max(0.0, (float(gap) - 1) * HOP_S)


def write_track(path: Path, regions: list[tuple[float, float, str]]) -> None:
    path.write_text(
        "".join(f"{a:.3f}\t{b:.3f}\t{txt}\n" for a, b, txt in regions), encoding="utf-8"
    )


def locate(stem: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        if (d / f"{stem}.wav").exists():
            return d / stem
    return None


def one_recording(name, y, p, dirs, args, totals, worst) -> None:
    """Writes the two error tracks for a single recording and folds its numbers into `totals`."""
    muted = policy(p, args.on, args.off)
    usable = y != DROP                 # a dropped frame had no correct answer, so it has none now
    masks = {"fp": muted & (y == 0) & usable,
             "fn": (~muted) & (y == POSITIVE) & usable}

    base = locate(name, dirs)
    if base is None:
        print(f"{name:30s} (no .wav found in {', '.join(str(x) for x in dirs)})")
        return

    counts, secs, adj_secs = {}, {}, {}
    for kind, mask in masks.items():
        regions = []
        for a, b in find_runs(mask):
            t0, t1 = span_seconds(a, b)
            if t1 - t0 < args.min_seconds:
                continue
            conf = float(p[a:b].mean())
            gap = distance_to_truth(a, b, y)
            where = "adj" if gap <= 0.0 else f"+{gap:.0f}s"
            regions.append((t0, t1, f"{kind.upper()} {t1 - t0:.1f}s p={conf:.2f} {where}"))
            worst.append((t1 - t0, f"{name} {kind.upper()} {t0 / 60:.1f}-{t1 / 60:.1f} min "
                                   f"p={conf:.2f} {where}"))
        write_track(base.with_name(base.name + f".{kind}.txt"), regions)
        counts[kind] = len(regions)
        secs[kind] = sum(b - a for a, b, _ in regions)
        adj_secs[kind] = sum(b - a for a, b, t in regions if t.endswith("adj"))

    total = secs["fp"] + secs["fn"]
    adj = adj_secs["fp"] + adj_secs["fn"]
    totals["fp"] += secs["fp"]
    totals["fn"] += secs["fn"]
    totals["adj"] += adj
    print(f"{name:30s} {counts['fp']:5d} {counts['fn']:5d} {secs['fp']:7.0f} {secs['fn']:7.0f}  "
          f"{(100 * adj / total if total else 0):17.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", help=".npz written by train.py --save-oof")
    ap.add_argument("--dir", action="append", default=None, required=True,
                    help="where the recordings live; repeat for several directories")
    ap.add_argument("--on", type=float, default=0.99, help="start attenuating above this")
    ap.add_argument("--off", type=float, default=0.98, help="stop attenuating below this")
    ap.add_argument("--min-seconds", type=float, default=1.0,
                    help="ignore mistakes shorter than this; a single frame is 0.48 s and a "
                         "track full of them is unreadable")
    ap.add_argument("--top", type=int, default=12, help="longest mistakes to print")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dir]
    d = np.load(args.oof, allow_pickle=True)
    y_all, p_all, groups, names = d["y"], d["p"], d["groups"], d["names"]

    print(f"operating point: attenuate above {args.on}, release below {args.off}")
    print(f"{'recording':30s} {'FP':>5s} {'FN':>5s} {'FP s':>7s} {'FN s':>7s}  "
          f"{'of which boundary':>18s}")
    print("-" * 82)

    totals = {"fp": 0.0, "fn": 0.0, "adj": 0.0}
    worst: list[tuple[float, str]] = []

    for i, name in enumerate(names):
        m = groups == i
        # A fold may hold several recordings of the same broadcast, folded together so none of it
        # leaks into training. Their frames sit end to end in discover() order, so split them back
        # apart by frame count - an error track has to line up with an actual file.
        members = str(name).split(" + ")
        offsets, cursor = [], 0
        for mem in members:
            base = locate(mem, dirs)
            n_f = (base.with_name(base.name + ".f16").stat().st_size // 2048) if base else 0
            offsets.append((mem, cursor, cursor + n_f))
            cursor += n_f
        if len(members) > 1 and cursor == int(m.sum()):
            rows = np.flatnonzero(m)
            for mem, a, b in offsets:
                one_recording(mem, y_all[rows[a:b]], p_all[rows[a:b]], dirs, args, totals, worst)
        else:
            one_recording(str(name), y_all[m], p_all[m], dirs, args, totals, worst)

    print("-" * 82)
    tot = totals["fp"] + totals["fn"]
    print(f"{'TOTAL':30s} {'':5s} {'':5s} {totals['fp']:7.0f} {totals['fn']:7.0f}  "
          f"{(100 * totals['adj'] / tot if tot else 0):17.0f}%")
    print()
    print(f"{totals['adj'] / 60:.1f} of {tot / 60:.1f} error-minutes touch a real break, so that")
    print("share is a boundary question rather than a detection one - check those against the")
    print("audio before concluding anything about the model.")

    if worst:
        print()
        print("Longest mistakes, listen to these first:")
        for secs, desc in sorted(worst, reverse=True)[:args.top]:
            print(f"  {secs:6.1f}s  {desc}")

    print()
    print("Wrote <stem>.fp.txt and <stem>.fn.txt beside each recording.")
    print("In Audacity: open the .wav, then File > Import > Labels three times - the ground truth")
    print("track, the .fp.txt and the .fn.txt - and each mistake becomes a region you can play.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
