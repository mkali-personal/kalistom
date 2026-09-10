#!/usr/bin/env python3
"""
Turns an Audacity label track into one label per YAMNet frame.

The join key between audio and embeddings is the frame index, and the frame grid is fixed by the
model: frame k covers samples [k*HOP, k*HOP + WINDOW), which is 0.48 k to 0.48 k + 0.975 seconds.
Everything here follows from that one fact.

THREE OUTCOMES PER FRAME, not two. A frame wholly inside a labelled span is positive, a frame
wholly outside every span is negative, and a frame that straddles a boundary is **dropped**. It is
not guessed at and not assigned to the majority: a frame containing half an ad and half the news
has no correct answer, and teaching the model that it does is teaching it noise. The same applies
to frames touching a join, where trainer/stitch.py recorded that about a second of audio is
missing - the embedding there describes a splice, not a broadcast.

Usage:
    python trainer/labels.py run_20260909_235255.gemini.txt --frames 3782
    python trainer/labels.py LABELS.txt --frames N --joins run_X.joins.txt --out run_X.y.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SR = 16000
WINDOW = 15600          # must match Yamnet.kt
HOP = 7680
WINDOW_S = WINDOW / SR  # 0.975
HOP_S = HOP / SR        # 0.48

POSITIVE, NEGATIVE, DROP = 1, 0, -1


def read_spans(path: Path) -> list[tuple[float, float, str]]:
    """Audacity label track. Point labels (start == end) are markers, not spans, and are skipped -
    stitch.py writes join markers in the same format."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        p = line.split("\t")
        if len(p) < 2:
            continue
        try:
            a, b = float(p[0]), float(p[1])
        except ValueError:
            continue
        if b > a:
            out.append((a, b, p[2] if len(p) > 2 else "ad"))
    return sorted(out)


def read_points(path: Path) -> list[float]:
    """The point labels the other reader throws away - here, the joins."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 2:
            try:
                a, b = float(p[0]), float(p[1])
            except ValueError:
                continue
            if b == a:
                out.append(a)
    return sorted(out)


def frame_labels(n_frames: int, spans, joins=(), join_guard: float = 2.0) -> np.ndarray:
    """One label per frame: 1 ad, 0 content, -1 do not train on this frame."""
    y = np.full(n_frames, NEGATIVE, dtype=np.int8)
    k = np.arange(n_frames)
    f0 = k * HOP_S
    f1 = f0 + WINDOW_S

    for a, b, _ in spans:
        inside = (f0 >= a) & (f1 <= b)
        touching = (f1 > a) & (f0 < b)
        y[touching] = DROP          # anything overlapping the span at all is suspect ...
        y[inside] = POSITIVE        # ... except what lies wholly within it
    for t in joins:
        y[(f1 > t - join_guard) & (f0 < t + join_guard)] = DROP
    return y


def summarise(y: np.ndarray) -> str:
    pos, neg, drop = (y == POSITIVE).sum(), (y == NEGATIVE).sum(), (y == DROP).sum()
    tot = max(len(y), 1)
    return (f"{len(y)} frames = {len(y) * HOP_S / 60:.1f} min\n"
            f"  ad       {pos:6d}  ({100 * pos / tot:4.1f}%)  {pos * HOP_S / 60:5.1f} min\n"
            f"  content  {neg:6d}  ({100 * neg / tot:4.1f}%)  {neg * HOP_S / 60:5.1f} min\n"
            f"  dropped  {drop:6d}  ({100 * drop / tot:4.1f}%)  boundary and join frames")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="Audacity label track with the ad spans")
    ap.add_argument("--frames", type=int, required=True, help="number of embedding frames")
    ap.add_argument("--joins", default="", help="join marker track from stitch.py")
    ap.add_argument("--join-guard", type=float, default=2.0,
                    help="seconds either side of a join to drop")
    ap.add_argument("--out", default="", help="write the label vector as .npy")
    args = ap.parse_args()

    spans = read_spans(Path(args.labels))
    joins = read_points(Path(args.joins)) if args.joins else []
    y = frame_labels(args.frames, spans, joins, args.join_guard)

    print(f"{len(spans)} ad span(s), {len(joins)} join(s)")
    print(summarise(y))
    if args.out:
        np.save(args.out, y)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
