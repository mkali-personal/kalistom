#!/usr/bin/env python3
"""
Measures how far a draft label track sits from a hand-marked one.

This is the gate on machine labelling. Drafts from a keyword scan or a language model are useful
only if we know their error, and the way to know it is to mark some audio by hand WITHOUT looking
at the draft first, then compare. Skipping this step would mean training on label noise of unknown
size, and later being unable to tell a model failure from a labelling failure.

Two different questions are reported, because they fail differently:

  Coverage    How many seconds the two tracks agree are advertising, and how many each has that
              the other does not. Missed ad seconds are ads that would play at full volume; false
              ad seconds are programme content that would be muted. The plan's targets are
              expressed in exactly these units, so they carry straight through to the app.

  Boundaries  For breaks both tracks found, how far apart their edges are. A draft that finds
              every break but is consistently four seconds late at the start is a different and
              much more fixable problem than one that misses breaks outright.

Usage:
    python trainer/compare_labels.py truth.txt draft.txt
    python trainer/compare_labels.py truth.txt draft.txt --resolution 0.48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_labels(path: Path) -> list[tuple[float, float, str]]:
    """Reads an Audacity label track. Point labels (start == end) are markers, not spans, and are
    ignored - trainer/stitch.py writes join markers into the same kind of file."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            print(f"  {path.name}:{n}: not a label line, skipped", file=sys.stderr)
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            print(f"  {path.name}:{n}: unreadable times, skipped", file=sys.stderr)
            continue
        if b > a:
            out.append((a, b, parts[2] if len(parts) > 2 else ""))
    return sorted(out)


def mask(spans, n: int, res: float) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b, _ in spans:
        m[int(a / res):int(np.ceil(b / res))] = True
    return m


def runs(m: np.ndarray, res: float) -> list[tuple[float, float]]:
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    return [(s * res, e * res) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth", help="hand-marked labels")
    ap.add_argument("draft", help="machine-generated labels")
    ap.add_argument("--resolution", type=float, default=0.48,
                    help="seconds per cell; defaults to the model's frame hop")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="length of the recording (default: last label end)")
    args = ap.parse_args()

    t = read_labels(Path(args.truth))
    d = read_labels(Path(args.draft))
    if not t:
        print("the truth track has no spans - nothing to compare against")
        return 1

    res = args.resolution
    dur = args.duration or max([b for _, b, _ in t + d] or [0.0])
    n = int(np.ceil(dur / res))
    mt, md = mask(t, n, res), mask(d, n, res)

    both = int((mt & md).sum())
    only_t = int((mt & ~md).sum())
    only_d = int((~mt & md).sum())
    neither = int((~mt & ~md).sum())
    hours = dur / 3600

    print(f"recording          {dur / 60:.1f} min, compared at {res:.2f}s resolution")
    print(f"truth  {len(t):3d} span(s)  {mt.sum() * res / 60:6.1f} min of ads "
          f"({100 * mt.mean():.1f}%)")
    print(f"draft  {len(d):3d} span(s)  {md.sum() * res / 60:6.1f} min of ads "
          f"({100 * md.mean():.1f}%)")
    print()
    print("COVERAGE")
    print(f"  agreed advertising      {both * res / 60:6.1f} min")
    print(f"  missed by the draft     {only_t * res / 60:6.1f} min"
          f"   = {only_t * res / max(hours, 1e-9):5.0f} ad-seconds heard per hour")
    print(f"  invented by the draft   {only_d * res / 60:6.1f} min"
          f"   = {only_d * res / max(hours, 1e-9):5.0f} content-seconds muted per hour")
    recall = both / max(both + only_t, 1)
    prec = both / max(both + only_d, 1)
    iou = both / max(both + only_t + only_d, 1)
    print(f"  recall {100 * recall:5.1f}%   precision {100 * prec:5.1f}%   IoU {100 * iou:5.1f}%")
    print(f"  agreement over the whole recording {100 * (both + neither) / max(n, 1):.1f}%")

    print("\nBOUNDARIES")
    tr, dr = runs(mt, res), runs(md, res)
    starts, ends, found = [], [], 0
    for a, b in tr:
        # The draft break that overlaps this one most; a break with no overlap at all was missed
        # outright and has no boundary error to report.
        best, ov = None, 0.0
        for c, e in dr:
            o = min(b, e) - max(a, c)
            if o > ov:
                best, ov = (c, e), o
        if best:
            found += 1
            starts.append(best[0] - a)
            ends.append(best[1] - b)
    print(f"  breaks in truth {len(tr)}, in draft {len(dr)}, matched {found}"
          f" ({len(tr) - found} missed, {len(dr) - found} spurious)")
    if starts:
        for name, v in (("start", np.array(starts)), ("end", np.array(ends))):
            print(f"  {name:5s} offset  median {np.median(v):+6.2f}s   "
                  f"p90 |error| {np.percentile(np.abs(v), 90):5.2f}s   "
                  f"worst {v[np.argmax(np.abs(v))]:+6.2f}s")
        print("  (positive = the draft is late)")

    print("\nVERDICT")
    if recall > 0.9 and prec > 0.9:
        print("  The draft is close enough to correct rather than redo. Label from drafts.")
    elif recall > 0.75:
        print("  The draft finds most advertising but needs real correction. Still worth using -")
        print("  correcting a draft is far cheaper than marking from a blank track - but do not")
        print("  train on uncorrected drafts.")
    else:
        print("  The draft misses too much to be a starting point. Look at what it missed before")
        print("  labelling more by hand: a systematic miss is usually one absent marker phrase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
