#!/usr/bin/env python3
"""
Does a session-0 AudioEffect attenuate what AudioPlaybackCapture sees?

BLOCKING for Phase 4. The media volume was already measured not to affect capture, which is what
lets the detector keep watching the audio while it mutes. If the session-0 effect behaves
differently, that property is lost: muting an ad would feed the model silence, so it could never
detect the ad ending - it would mute once and sit there until a timeout.

Reads the markers the recorder wrote into the session, so stage boundaries are exact rather than
estimated from wall clock.

Usage:  python trainer/attenuation_report.py [sessions_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HOP, SAMPLE_RATE = 7680, 16000
DEFAULT = Path("captures/sessions")

# Ignore this much either side of a marker: markers are applied when the recorder loop next
# drains its queue (within one hop), and effects take a moment to take hold.
EDGE_TRIM_S = 2.0


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    metas = sorted(d.glob("sess_*.jsonl"))
    if not metas:
        print(f"no sessions in {d}")
        return 1

    meta = None
    for m in reversed(metas):
        txt = m.read_text(encoding="utf-8")
        if "loudness_-40db" in txt or "baseline" in txt:
            meta = m
            break
    if meta is None:
        print("no session contains attenuation-test markers - run the test from the app first")
        return 1

    rms, markers = [], []
    with meta.open(encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("type") == "marker":
                markers.append((float(o["sample"]) / SAMPLE_RATE, o["label"]))
            elif "f" in o and "rms" in o:
                rms.append(o["rms"])
    rms = np.asarray(rms, dtype=np.float64)
    t = np.arange(len(rms)) * HOP / SAMPLE_RATE

    print(f"{meta.name}: {len(rms)} frames, {t[-1] if len(t) else 0:.0f}s, "
          f"{len(markers)} markers\n")
    if len(markers) < 2:
        print("not enough markers")
        return 1

    # Each marker opens a stage that runs until the next marker.
    bounds = [(lbl, t0, markers[i + 1][0] if i + 1 < len(markers) else float(t[-1]))
              for i, (t0, lbl) in enumerate(markers)]

    print(f"{'stage':>16}  {'window':>14}  {'frames':>6}  {'median rms':>11}  {'delta':>8}")
    base = None
    results = {}
    for lbl, a, b in bounds:
        m = (t >= a + EDGE_TRIM_S) & (t < b - EDGE_TRIM_S)
        if m.sum() < 3:
            print(f"{lbl:>16}  {a:6.0f}-{b:<6.0f}  {m.sum():6d}   (too short)")
            continue
        med = float(np.median(rms[m]))
        if base is None:
            base = med
        results[lbl] = med
        print(f"{lbl:>16}  {a:6.0f}-{b:<6.0f}s {m.sum():6d}  {med:11.2f}  {med - base:+8.2f}")

    print()
    verdict_lines = []
    for kind, label in (("LoudnessEnhancer", "loudness_-40db"),
                        ("DynamicsProcessing", "dynamics_-60db")):
        if label not in results or base is None:
            verdict_lines.append(f"{kind:<20} not measured")
            continue
        delta = results[label] - base
        if delta < -10:
            verdict_lines.append(
                f"{kind:<20} ATTENUATES CAPTURE ({delta:+.1f} dB) - unusable as the actuator")
        elif abs(delta) < 3:
            verdict_lines.append(
                f"{kind:<20} does NOT affect capture ({delta:+.1f} dB) - safe to use")
        else:
            verdict_lines.append(
                f"{kind:<20} unclear ({delta:+.1f} dB) - programme level may have shifted")
    print("\n".join(verdict_lines))

    print()
    safe = [l for l in verdict_lines if "does NOT affect" in l]
    if safe:
        print("=> The session-0 actuator keeps the detector sighted while muting. Phase 4 can")
        print("   build on it (still needs a watchdog: it attenuates alarms and calls too).")
    else:
        print("=> Fall back to AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK for Phase 4: a global effect")
        print("   that also attenuates the capture path would blind the detector mid-ad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
