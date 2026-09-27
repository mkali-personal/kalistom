"""Transliteration of Detector.kt's ring buffer, checked against stack_context.

Detector.kt cannot import train.py, so the two definitions of "the last ten frames, oldest first,
clamped at the start" exist twice - once in Python and once in Kotlin. This mirrors the Kotlin
indexing arithmetic line for line and asserts it selects the same source frame for every tap of
every frame, so a slot or lag mistake is caught here rather than after an hour of recording.

    python trainer/test_detector_logic.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import stack_context                                      # noqa: E402


def kotlin_sources(n_frames: int, context: int) -> np.ndarray:
    """Which source frame each (frame, tap) reads, per Detector.kt's arithmetic."""
    out = np.empty((n_frames, context), dtype=int)
    ring_holds = {}                       # slot -> frame index currently stored there
    frames_seen = 0
    write_slot = 0
    for k in range(n_frames):
        # --- push(): store this frame, then advance, exactly as the Kotlin does ---
        ring_holds[write_slot] = k
        write_slot = (write_slot + 1) % context
        frames_seen += 1

        for tap in range(context):
            lag = context - 1 - tap
            slot = ((frames_seen - 1 - lag) % context) if frames_seen > lag else 0
            out[k, tap] = ring_holds[slot]
    return out


def main() -> int:
    for context in (1, 2, 3, 10, 11):
        for n_frames in (1, 2, 5, 9, 10, 11, 37, 100):
            d0 = 3
            # A distinguishable value per frame, so a wrong source frame cannot coincide.
            E = (np.arange(n_frames)[:, None] * 10 + np.arange(d0)[None, :]).astype(np.float32)
            want = stack_context(E, context).reshape(n_frames, context, d0)
            src = kotlin_sources(n_frames, context)
            got = E[src]
            if not np.array_equal(want, got):
                bad = np.argwhere((want != got).any(-1))[0]
                print(f"MISMATCH context={context} frames={n_frames} at frame {bad[0]} "
                      f"tap {bad[1]}: want {want[tuple(bad)]}, got {got[tuple(bad)]}")
                return 1
    print("OK - Detector.kt's ring arithmetic selects the same frames as stack_context")
    print("     (checked for context 1,2,3,10,11 x up to 100 frames, including before the ring fills)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
