#!/usr/bin/env python3
"""
Trains the ad/content head on YAMNet embeddings, and evaluates it honestly.

The head is small on purpose: logistic regression over a stack of consecutive 1024-d embeddings,
optionally with the per-frame loudness the recorder already stores. YAMNet does the hard work; this
only has to draw a boundary in the space it produces.

WHY THE EVALUATION IS THE INTERESTING PART. Adjacent frames overlap in time - the hop is 0.48 s and
the window 0.975 s, so consecutive frames literally share half their audio - which means a random
train/test split puts near-identical rows on both sides and reports a number that has nothing to do
with how the app will behave. This splits by contiguous blocks instead, and reports:

  * frame-level precision and recall, which is what the model optimises, and
  * ad-seconds-heard per hour and content-seconds-muted per hour, which is what you will notice.

The second pair is the one that matters. They come from running the frame probabilities through the
same two-threshold policy the app will use: start attenuating above --on, stop below --off. That
asymmetry is a statement about costs, not probabilities - a second of ad leaking is more annoying
than a second of content muted - which is why it lives in the policy and not in the model.

A NOTE ON WHAT ONE RECORDING CAN SHOW. With a single ad break there is no split that puts unseen
advertising in the test half, so a good score here demonstrates that the pipeline runs, not that
the model works. The script says so in its verdict rather than letting the number speak.

Usage:
    python trainer/train.py --emb run_X.f16 --labels run_X.gemini.txt --joins run_X.joins.txt
    python trainer/train.py ... --context 10 --out head_weights.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from labels import DROP, HOP_S, POSITIVE, frame_labels, read_points, read_spans, summarise

EMBEDDING_DIM = 1024


def load_embeddings(path: Path) -> np.ndarray:
    """The .f16 the phone writes: float16, EMBEDDING_DIM per frame, nothing else."""
    raw = np.fromfile(path, dtype="<f2")
    if raw.size % EMBEDDING_DIM:
        raise SystemExit(f"{path.name}: {raw.size} values is not a whole number of "
                         f"{EMBEDDING_DIM}-d frames")
    return raw.reshape(-1, EMBEDDING_DIM).astype(np.float32)


def load_rms(path: Path) -> np.ndarray | None:
    """Per-frame loudness from the .jsonl beside the embeddings. YAMNet normalises level away, so
    this carries something the embedding genuinely does not: ads are compressed and flat."""
    if not path.exists():
        return None
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if '"rms"' in line:
            try:
                vals.append(float(json.loads(line)["rms"]))
            except Exception:                                   # noqa: BLE001
                pass
    return np.array(vals, dtype=np.float32) if vals else None


def stack_context(X: np.ndarray, n: int) -> np.ndarray:
    """Concatenates each frame with the n-1 before it, so the head sees a few seconds rather than
    one. Edges repeat the first frame instead of being dropped, which costs nothing and keeps the
    row count equal to the frame count."""
    if n <= 1:
        return X
    idx = np.arange(len(X))[:, None] - np.arange(n)[None, ::-1]
    return X[np.clip(idx, 0, None)].reshape(len(X), -1)


def blocks(n: int, block_frames: int, folds: int) -> np.ndarray:
    """Fold assignment by contiguous block, so overlapping neighbours stay on the same side."""
    return (np.arange(n) // block_frames) % folds


def fit(X: np.ndarray, y: np.ndarray, l2: float, iters: int) -> np.ndarray:
    """Logistic regression by gradient descent, class-balanced. Written out rather than pulled from
    sklearn so the weights are plainly ours to ship to the phone as JSON."""
    Xb = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
    w = np.zeros(Xb.shape[1], dtype=np.float32)
    pos, neg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / neg).astype(np.float32) * len(y)
    lr = 1.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (sw * (p - y)) / len(y) + l2 * np.concatenate([w[:-1], [0.0]])
        w -= lr * g
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))


def policy(p: np.ndarray, on: float, off: float) -> np.ndarray:
    """The app's decision, not the model's: attenuate above `on`, release below `off`. Between the
    two, keep doing whatever you were doing - that gap is the hysteresis."""
    out = np.zeros(len(p), dtype=bool)
    state = False
    for i, v in enumerate(p):
        if state:
            state = v >= off
        else:
            state = v >= on
        out[i] = state
    return out


def report(y: np.ndarray, p: np.ndarray, on: float, off: float, title: str) -> dict:
    keep = y != DROP
    yt, pt = y[keep], p[keep]
    pred = pt >= 0.5
    tp = int((pred & (yt == 1)).sum())
    fp = int((pred & (yt == 0)).sum())
    fn = int((~pred & (yt == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    muted = policy(p, on, off)
    hours = len(y) * HOP_S / 3600
    heard = int(((y == POSITIVE) & ~muted).sum()) * HOP_S
    wrong = int(((y == 0) & muted).sum()) * HOP_S

    print(f"\n{title}")
    print(f"  frame level    precision {100 * prec:5.1f}%   recall {100 * rec:5.1f}%   "
          f"F1 {100 * 2 * prec * rec / max(prec + rec, 1e-9):5.1f}%")
    print(f"  what you hear  {heard / max(hours, 1e-9):6.0f} ad-seconds heard per hour")
    print(f"  what you lose  {wrong / max(hours, 1e-9):6.0f} content-seconds muted per hour")
    return {"precision": prec, "recall": rec,
            "ad_seconds_heard_per_hour": heard / max(hours, 1e-9),
            "content_seconds_muted_per_hour": wrong / max(hours, 1e-9)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="the .f16 embeddings")
    ap.add_argument("--labels", required=True, help="Audacity label track with ad spans")
    ap.add_argument("--joins", default="", help="join markers from stitch.py")
    ap.add_argument("--meta", default="", help=".jsonl beside the embeddings (for rms)")
    ap.add_argument("--context", type=int, default=10, help="frames of context per row")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-seconds", type=float, default=30.0)
    ap.add_argument("--on", type=float, default=0.6, help="start attenuating above this")
    ap.add_argument("--off", type=float, default=0.2, help="stop attenuating below this")
    ap.add_argument("--out", default="", help="write head weights here as JSON")
    args = ap.parse_args()

    emb = load_embeddings(Path(args.emb))
    meta = Path(args.meta) if args.meta else Path(args.emb).with_suffix(".jsonl")
    rms = load_rms(meta)

    spans = read_spans(Path(args.labels))
    joins = read_points(Path(args.joins)) if args.joins else []
    y = frame_labels(len(emb), spans, joins)
    print(f"{len(spans)} ad span(s), {len(joins)} join(s)")
    print(summarise(y))

    X = emb
    if rms is not None and len(rms) == len(emb):
        X = np.hstack([X, (rms[:, None] + 60.0) / 60.0])
        print("using per-frame loudness alongside the embedding")
    elif rms is not None:
        print(f"ignoring rms: {len(rms)} values for {len(emb)} frames")
    X = stack_context(X, args.context)
    print(f"features: {X.shape[1]} per frame ({args.context} frames of context = "
          f"{(args.context - 1) * HOP_S + 0.975:.1f}s)")

    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd

    if (y == POSITIVE).sum() == 0:
        print("\nNo positive frames - nothing to learn. Check the label track's time range.")
        return 1

    keep = y != DROP
    fold = blocks(len(y), max(int(args.block_seconds / HOP_S), 1), args.folds)

    # Out-of-fold predictions: every frame is scored by a model that never saw its block.
    oof = np.zeros(len(y), dtype=np.float32)
    for f in range(args.folds):
        tr = keep & (fold != f)
        if (y[tr] == POSITIVE).sum() == 0 or (y[tr] == 0).sum() == 0:
            print(f"fold {f}: one class missing in training - skipped")
            oof[fold == f] = 0.0
            continue
        w = fit(X[tr], y[tr].astype(np.float32), args.l2, args.iters)
        oof[fold == f] = predict(X[fold == f], w)

    w_full = fit(X[keep], y[keep].astype(np.float32), args.l2, args.iters)
    report(y, predict(X, w_full), args.on, args.off, "FIT (trained on everything - memorisation)")
    m = report(y, oof, args.on, args.off,
               f"HELD OUT ({args.folds}-fold by {args.block_seconds:.0f}s block)")

    n_breaks = len(spans)
    print("\nVERDICT")
    if n_breaks < 3:
        print(f"  Only {n_breaks} ad break(s) in this data. Every fold's test half contains part")
        print("  of the same break the model trained on, so the held-out numbers above are")
        print("  optimistic by an unknown amount. This run shows the pipeline works end to end;")
        print("  it does not show the model works. That needs recordings from several days.")
    elif m["recall"] > 0.8 and m["content_seconds_muted_per_hour"] < 60:
        print("  Held-out performance is in the range the plan calls usable. Worth shipping to")
        print("  the phone and listening to.")
    else:
        print("  Held-out performance is not yet usable. More labelled breaks before more model.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "type": "logistic_regression",
            "context_frames": args.context,
            "uses_rms": bool(rms is not None and len(rms) == len(emb)),
            "input_dim": int(X.shape[1]),
            "mean": mu.tolist(), "scale": sd.tolist(),
            "weights": w_full[:-1].tolist(), "bias": float(w_full[-1]),
            "on": args.on, "off": args.off,
            "trained_on": {"embeddings": Path(args.emb).name, "labels": Path(args.labels).name,
                           "ad_breaks": n_breaks},
            "held_out": m,
        }, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
