#!/usr/bin/env python3
"""
Does the phone compute the same score the desktop does?

The app now carries the trained head, and Detector.kt has to reproduce train.py exactly: the same
column order, the same tap order, the same edge clamping, the same standardisation. None of that
fails loudly. Get the tap order backwards and the app still produces a confident-looking number
every half second, from the wrong weights - a plausible score that has nothing to do with what was
fitted. So it has to be measured on a real session rather than reasoned about.

The check is only possible because the phone writes its own score into the session's .jsonl. This
reads the embeddings the phone wrote, scores them here with head_weights.json, and compares the two
series frame by frame. They should agree to about float precision; anything worse is a bug in one
of the two implementations, and the histogram of disagreement says which kind.

    python trainer/parity_check.py captures/sessions/sess_20260927_101500

Also checks the decision, not just the score, because the smoothing and the two thresholds are
part of the app too and can drift independently of the dot product.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smooth import hysteresis, rolling                               # noqa: E402
from train import EMBEDDING_DIM, load_embeddings, load_rms, stack_context   # noqa: E402


def phone_series(jsonl: Path):
    """The app's own per-frame score, smoothed score and mute decision."""
    raw, sm, mute, idx = [], [], [], []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if '"ad"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:                                            # noqa: BLE001
            continue
        idx.append(int(o["f"]))
        raw.append(float(o["ad"]))
        sm.append(float(o["ads"]))
        mute.append(bool(o.get("mute", 0)))
    return (np.array(idx), np.array(raw, np.float64),
            np.array(sm, np.float64), np.array(mute, bool))


def desktop_score(emb: np.ndarray, rms: np.ndarray | None, cfg: dict) -> np.ndarray:
    """The same score computed here, by the same definition train.py used."""
    taps = int(cfg["context_frames"])
    x = emb.astype(np.float32)
    if rms is not None and len(rms) == len(emb):
        x = np.hstack([x, ((rms[:, None] + 60.0) / 60.0).astype(np.float32)])
    else:
        x = np.hstack([x, np.zeros((len(x), 1), np.float32)])
    mu = np.asarray(cfg["mean"], np.float32)
    sd = np.asarray(cfg["scale"], np.float32)
    if cfg.get("type") == "mlp":
        # Each frame's hidden vector first, then the context over those - context_mlp.predict's
        # order, and the phone's. stack_context repeats frame 0's hidden vector at the start,
        # which is the clamping the phone's ring does too.
        hidden = int(cfg["hidden"])
        w1 = np.asarray(cfg["w1"], np.float32).reshape(hidden, -1)
        w2 = np.asarray(cfg["w2"], np.float32).reshape(taps, hidden)
        h = np.maximum(((x - mu) / sd) @ w1.T + np.asarray(cfg["b1"], np.float32), 0.0)
        logit = stack_context(h, taps) @ w2.ravel() + float(cfg["b2"])
    else:
        s = stack_context(x, taps)
        logit = ((s - mu) / sd) @ np.asarray(cfg["weights"], np.float32) + float(cfg["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="session stem, or its .f16 / .jsonl")
    ap.add_argument("--weights", default="head_weights.json")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="largest score disagreement treated as float noise")
    args = ap.parse_args()

    stem = Path(args.session)
    if stem.suffix in (".f16", ".jsonl", ".wav"):
        stem = stem.with_suffix("")
    embp, jsonlp = stem.with_suffix(".f16"), stem.with_suffix(".jsonl")
    for f in (embp, jsonlp):
        if not f.exists():
            print(f"missing {f}")
            return 1

    cfg = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    emb = load_embeddings(embp)
    rms = load_rms(jsonlp)
    idx, praw, psm, pmute = phone_series(jsonlp)

    if len(idx) == 0:
        print("This session carries no 'ad' fields, so it was recorded by a build without the")
        print("classifier, or with no head_weights.json in assets. Record a new session first.")
        return 1
    if not np.array_equal(idx, np.arange(len(idx))):
        print(f"frame indices are not contiguous from 0 - first gap at {np.argmax(np.diff(idx) != 1)}")
        return 1

    n = min(len(emb), len(idx))
    draw = desktop_score(emb[:n], None if rms is None else rms[:n], cfg)
    praw, psm, pmute = praw[:n], psm[:n], pmute[:n]

    d = np.abs(draw - praw)
    print(f"{n} frames ({n * 0.48 / 60:.1f} min)\n")
    print("RAW SCORE, phone against desktop")
    print(f"  max difference    {d.max():.2e}")
    print(f"  median difference {np.median(d):.2e}")
    print(f"  99th percentile   {np.percentile(d, 99):.2e}")
    for t in (1e-4, 1e-3, 1e-2, 1e-1):
        print(f"  frames off by >{t:g}: {int((d > t).sum()):6d}  ({100 * (d > t).mean():5.2f}%)")

    taps = int(cfg.get("smoothing", {}).get("frames", 1))
    on, off = float(cfg["on"]), float(cfg["off"])
    dsm = rolling(draw.astype(np.float32), taps, on, off) if taps > 1 else \
        hysteresis(draw.astype(np.float32), on, off)
    agree = int((dsm == pmute).sum())
    print(f"\nMUTE DECISION (mean of {taps} frames, on={on} off={off})")
    print(f"  phone muted   {100 * pmute.mean():5.1f}% of frames")
    print(f"  desktop muted {100 * dsm.mean():5.1f}% of frames")
    print(f"  agreement     {100 * agree / n:5.2f}%  ({n - agree} frames differ)")

    ok = d.max() <= args.tol
    print()
    if ok:
        print(f"PASS - the phone reproduces the desktop score to {d.max():.1e}. The head is being")
        print("fed the same vectors it was trained on.")
    else:
        worst = int(np.argmax(d))
        print(f"FAIL - worst disagreement {d.max():.3e} at frame {worst} "
              f"(phone {praw[worst]:.4f}, desktop {draw[worst]:.4f}).")
        print("A near-constant offset points at standardisation; disagreement concentrated in the")
        print("first frames points at edge clamping; disagreement everywhere points at tap or")
        print("column order, which is the one that still looks plausible on the phone.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
