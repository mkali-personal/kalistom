"""Transliteration of Detector.kt's ring buffer, checked against stack_context.

Detector.kt cannot import train.py, so the two definitions of "the last ten frames, oldest first,
clamped at the start" exist twice - once in Python and once in Kotlin. This mirrors the Kotlin
indexing arithmetic line for line and asserts it selects the same source frame for every tap of
every frame, so a slot or lag mistake is caught here rather than after an hour of recording.

    python trainer/test_detector_logic.py
"""
import json
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
    return check_mlp()


def kotlin_mlp_scores(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Detector.kt's Mlp scorer, transliterated: the folded first layer (w1 / scale, and the
    constant b1 - (w1 / scale) . mean), a ring of hidden vectors written at writeSlot, and the
    same slot arithmetic as the linear head. Returns the sigmoid of each frame's logit."""
    taps, hidden = int(cfg["context_frames"]), int(cfg["hidden"])
    mean = np.asarray(cfg["mean"], np.float32)
    scale = np.asarray(cfg["scale"], np.float32)
    w1 = np.asarray(cfg["w1"], np.float32).reshape(hidden, -1)
    w1s = w1 / scale[None, :]
    c1 = np.asarray(cfg["b1"], np.float64) - (w1s.astype(np.float64) @ mean.astype(np.float64))
    w2 = np.asarray(cfg["w2"], np.float32).reshape(taps, hidden)
    ring = np.zeros((taps, hidden), np.float32)
    frames_seen, write_slot = 0, 0
    out = np.empty(len(x))
    for k in range(len(x)):
        s = c1 + w1s.astype(np.float64) @ x[k].astype(np.float64)
        ring[write_slot] = np.maximum(s, 0.0).astype(np.float32)
        write_slot = (write_slot + 1) % taps
        frames_seen += 1
        acc = float(cfg["b2"])
        for tap in range(taps):
            lag = taps - 1 - tap
            slot = ((frames_seen - 1 - lag) % taps) if frames_seen > lag else 0
            acc += float(w2[tap] @ ring[slot])
        out[k] = 1.0 / (1.0 + np.exp(-max(min(acc, 30.0), -30.0)))
    return out


def check_mlp() -> int:
    """The Kotlin mlp's arithmetic against parity_check.desktop_score, the desktop definition,
    on the shipped head and a real session - or on random weights if those are not present."""
    from parity_check import desktop_score
    from train import load_embeddings, load_rms
    root = Path(__file__).resolve().parent.parent
    head = root / "app" / "src" / "main" / "assets" / "head_weights.json"
    stem = root / "captures" / "sessions" / "sess_20260929_073233"
    rng = np.random.default_rng(0)
    cfg = json.loads(head.read_text(encoding="utf-8")) if head.exists() else {}
    if cfg.get("type") != "mlp" or not stem.with_suffix(".f16").exists():
        hidden, taps, d0 = 8, 10, 1025
        cfg = {"type": "mlp", "context_frames": taps, "hidden": hidden,
               "mean": rng.normal(size=d0).tolist(), "scale": (rng.random(d0) + 0.5).tolist(),
               "w1": (rng.normal(size=hidden * d0) * 0.05).tolist(),
               "b1": rng.normal(size=hidden).tolist(),
               "w2": rng.normal(size=taps * hidden).tolist(), "b2": 0.1}
        emb = rng.normal(size=(300, 1024)).astype(np.float32)
        rms = rng.uniform(-50, -10, 300).astype(np.float32)
        where = "random weights and frames"
    else:
        emb = load_embeddings(stem.with_suffix(".f16"))[:3000]
        rms = load_rms(stem.with_suffix(".jsonl"))[:3000]
        where = f"the shipped head on {stem.name}"
    x = np.hstack([emb.astype(np.float32), ((rms[:, None] + 60.0) / 60.0).astype(np.float32)])
    got = kotlin_mlp_scores(x, cfg)
    want = desktop_score(emb, rms, cfg)
    d = float(np.abs(got - want).max())
    if d > 1e-4:
        print(f"MISMATCH - the Kotlin mlp arithmetic differs from the desktop by {d:.2e} ({where})")
        return 1
    print(f"OK - Detector.kt's mlp arithmetic matches the desktop score to {d:.1e}")
    print(f"     ({len(x)} frames, {where})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
