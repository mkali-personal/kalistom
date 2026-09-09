#!/usr/bin/env python3
"""
Reads the newest session's per-frame rms trace and reports whether the level stepped down,
answering: does the phone's media volume affect AudioPlaybackCapture?

Run after tools/volume_test.sh + trainer/ingest.py --pull.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HOP, SAMPLE_RATE = 7680, 16000
DEFAULT = Path("captures/sessions")


def load(meta: Path) -> np.ndarray:
    rms = []
    with meta.open(encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if "f" in o and "rms" in o:
                rms.append(o["rms"])
    return np.asarray(rms, dtype=np.float64)


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    metas = sorted(d.glob("sess_*.jsonl"))
    if not metas:
        print(f"no sessions in {d}")
        return 1
    meta = metas[-1]
    rms = load(meta)
    if len(rms) < 20:
        print(f"{meta.name}: only {len(rms)} frames, need a longer recording")
        return 1

    dur = len(rms) * HOP / SAMPLE_RATE
    print(f"{meta.name}: {len(rms)} frames, {dur:.0f}s\n")

    # Summarise in 10 equal slices so a staircase is obvious without plotting.
    slices = np.array_split(rms, 10)
    print("  slice     t(s)      median rms")
    for i, sl in enumerate(slices):
        t0 = sum(len(s) for s in slices[:i]) * HOP / SAMPLE_RATE
        bar = "#" * int(max(0, (np.median(sl) + 80) / 2))
        print(f"   {i:2d}    {t0:6.0f}     {np.median(sl):7.1f}  {bar}")

    span = float(np.median(slices[0]) - np.median(slices[-1]))
    quiet = float(np.median(slices[-1]))
    print()
    if span > 12 or quiet < -70:
        print(f"VERDICT: capture FOLLOWS the volume slider (dropped {span:.1f} dB).")
        print("         Recording silently is not an option - keep the volume up.")
    elif abs(span) < 4:
        print(f"VERDICT: capture is INDEPENDENT of the volume slider ({span:+.1f} dB drift).")
        print("         You can record with the phone silent.")
    else:
        print(f"VERDICT: unclear ({span:+.1f} dB). Programme material may have changed;")
        print("         re-run against steadier audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
