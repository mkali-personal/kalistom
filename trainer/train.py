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
from rectime import describe, in_hours, parse_window, started_at

EMBEDDING_DIM = 1024
MIN_SAME_BROADCAST_S = 30.0   # shorter than this is two recordings abutting, not one broadcast


def load_embeddings(path: Path) -> np.ndarray:
    """The .f16 the phone writes: float16, EMBEDDING_DIM per frame, nothing else."""
    raw = np.fromfile(path, dtype="<f2")
    if raw.size % EMBEDDING_DIM:
        raise SystemExit(f"{path.name}: {raw.size} values is not a whole number of "
                         f"{EMBEDDING_DIM}-d frames")
    # Left as float16, the precision the phone actually wrote. Upcasting here doubles a design
    # matrix that is already the largest thing in the process and adds no information whatsoever.
    return raw.reshape(-1, EMBEDDING_DIM)


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


# Hand-marked labels beat machine drafts wherever both exist; that precedence is the whole point
# of keeping them in separate files.
LABEL_ORDER = (".truth.txt", ".gemini.txt", ".claude.txt")


def discover(d: Path) -> list[tuple[Path, Path, Path, Path | None]]:
    """Every recording in `d` that has embeddings and some kind of ad labels."""
    out = []
    for emb in sorted(d.glob("*.f16")):
        stem = emb.with_suffix("")
        for suffix in LABEL_ORDER:
            lab = stem.with_name(stem.name + suffix)
            if lab.exists():
                joins = stem.with_name(stem.name + ".joins.txt")
                out.append((stem, emb, lab, joins if joins.exists() else None))
                break
    return out


def overlap_groups(recs, spans) -> list[int]:
    """Fold assignment that keeps recordings of the SAME broadcast together.

    Leave-one-recording-out is only honest if the held-out recording's audio appears nowhere in
    the training half. That stops being true the moment the same broadcast is captured twice - the
    phone session of 2026-09-14 08:46-09:28 covers the same half hour as two desktop recordings,
    because the desktop recorder was running at the time. Holding out the phone session while its
    own audio sits in training, arriving by a different route, would measure nothing and would
    look excellent doing it.

    So recordings whose wall-clock ranges intersect share a fold.
    """
    n = len(recs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            (a0, a1), (b0, b1) = spans[i], spans[j]
            if a0 is None or b0 is None:
                continue
            # Consecutive stitched runs abut, and frame-count rounding can make one appear to
            # start a fraction of a second before the previous ended. That is adjacency, not the
            # same broadcast twice, so require a real overlap before folding them together.
            if (min(a1, b1) - max(a0, b0)).total_seconds() > MIN_SAME_BROADCAST_S:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def frame_mid(start, emb_path: Path):
    """Midpoint of a recording, from the embedding file's size - no need to open the audio."""
    from datetime import timedelta
    frames = emb_path.stat().st_size // (EMBEDDING_DIM * 2)
    return start + timedelta(seconds=frames * HOP_S / 2)


CHUNK = 4000          # rows per pass; 4000 x 10250 float32 is about 160 MB


def chunk_ranges(n: int):
    for a in range(0, n, CHUNK):
        yield a, min(a + CHUNK, n)


def mean_std(X: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations over `idx`, in chunks.

    X.mean(0) on a boolean-indexed slice would first materialise that slice - several gigabytes
    for a copy that is read once and thrown away. This walks it instead.
    """
    d = X.shape[1]
    s = np.zeros(d, np.float64)
    ss = np.zeros(d, np.float64)
    for a, b in chunk_ranges(len(idx)):
        z = X[idx[a:b]].astype(np.float32)
        s += z.sum(0, dtype=np.float64)
        ss += np.einsum("ij,ij->j", z, z, dtype=np.float64)
    n = max(len(idx), 1)
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 0.0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def blocks(n: int, block_frames: int, folds: int) -> np.ndarray:
    """Fold assignment by contiguous block, so overlapping neighbours stay on the same side."""
    return (np.arange(n) // block_frames) % folds


def fit(X, idx, y, mu, sd, l2: float, iters: int) -> np.ndarray:
    """Class-balanced logistic regression over the rows `idx`, solved with L-BFGS.

    Never materialises the training rows. The loss, the probabilities and the gradient are all
    accumulated in one chunked pass, which is possible because the residual is elementwise in the
    scores: a chunk's contribution to X.T @ r needs only that chunk's own rows. Standardisation
    happens per chunk from `mu`/`sd` fitted on the training rows alone.

    The design matrix is 131,000 x 10,250 here. Copying it per fold, as an earlier version did in
    X[tr] and again in the projection, needed 21 GB and the operating system killed the process.
    """
    from scipy.optimize import minimize                    # imported late; only this needs it

    n, d = len(idx), X.shape[1]
    inv_sd = (1.0 / sd).astype(np.float32)
    pos, neg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / neg).astype(np.float64) * n

    def loss_grad(w):
        w32 = w.astype(np.float32)
        ww, bias = w32[:d], float(w32[d])
        ll = 0.0
        g = np.zeros(d + 1, np.float64)
        for a, b in chunk_ranges(n):
            z = (X[idx[a:b]].astype(np.float32) - mu) * inv_sd
            sc = np.clip(z @ ww + bias, -30, 30).astype(np.float64)
            yc, swc = y[a:b], sw[a:b]
            ll += float(np.sum(swc * (np.logaddexp(0, sc) - yc * sc)))
            r = (swc * (1.0 / (1.0 + np.exp(-sc)) - yc)).astype(np.float32)
            g[:d] += z.T @ r
            g[d] += float(r.sum())
        ll /= n
        g /= n
        g[:d] += l2 * w[:d]
        return ll + 0.5 * l2 * float(w[:d] @ w[:d]), g

    res = minimize(loss_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "maxcor": 20})
    return res.x.astype(np.float32)


def predict(X, idx, w, mu, sd) -> np.ndarray:
    """Probabilities for rows `idx`, chunked for the same reason."""
    d = X.shape[1]
    inv_sd = (1.0 / sd).astype(np.float32)
    ww, bias = w[:d].astype(np.float32), float(w[d])
    out = np.empty(len(idx), np.float32)
    for a, b in chunk_ranges(len(idx)):
        z = (X[idx[a:b]].astype(np.float32) - mu) * inv_sd
        out[a:b] = 1.0 / (1.0 + np.exp(-np.clip(z @ ww + bias, -30, 30)))
    return out


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
    # The comparison that decides whether any of this is worth shipping. Switching the app off
    # mutes no content and lets every ad through; a model is only useful if the ad-seconds it
    # saves are worth more than the content-seconds it costs. Printing the do-nothing baseline
    # beside every result keeps that trade visible instead of implied.
    base = int((y == POSITIVE).sum()) * HOP_S / max(hours, 1e-9)
    saved = base - heard / max(hours, 1e-9)
    cost = wrong / max(hours, 1e-9)
    print(f"  do nothing     {base:6.0f} ad-seconds heard per hour, 0 content muted")
    print(f"  the trade      saves {saved:.0f} ad-seconds, costs {cost:.0f} content-seconds "
          f"({saved / max(cost, 1e-9):.2f} saved per second lost)")
    return {"precision": prec, "recall": rec,
            "ad_seconds_saved_per_hour": saved, "trade_ratio": saved / max(cost, 1e-9),
            "ad_seconds_heard_per_hour": heard / max(hours, 1e-9),
            "content_seconds_muted_per_hour": wrong / max(hours, 1e-9)}


def sweep(y: np.ndarray, p: np.ndarray, gap: float = 0.0) -> None:
    """What the same predictions buy at every operating point.

    The model produces a probability; the app chooses what to do about it. Those are separate
    decisions, and reporting a single threshold hides the choice - a model that looks useless at
    0.6 can be worth shipping at 0.95, because muting less often costs a few ad-seconds and saves
    a great many content-seconds, and the programme is the thing worth protecting.

    Both thresholds are swept independently. Tying `off` to `on` by a fixed gap - the obvious
    shortcut - produces an almost flat curve and hides the truth, because raising `on` alone does
    nothing once muting has started: the state persists until the probability falls below `off`.
    """
    hours = len(y) * HOP_S / 3600
    base = int((y == POSITIVE).sum()) * HOP_S / max(hours, 1e-9)
    grid = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
    rows = []
    for on in grid:
        for off in grid:
            if off > on:
                continue
            muted = policy(p, on, off)
            heard = int(((y == POSITIVE) & ~muted).sum()) * HOP_S / max(hours, 1e-9)
            lost = int(((y == 0) & muted).sum()) * HOP_S / max(hours, 1e-9)
            rows.append((base - heard, lost, on, off, heard))

    print()
    print("OPERATING POINT (same predictions, both thresholds swept)")
    print(f"{'on':>5s} {'off':>5s} {'ad-sec heard/h':>15s} {'content-sec lost/h':>19s} "
          f"{'saved per lost':>15s}")
    for saved, lost, on, off, heard in sorted(rows, key=lambda r: -r[0] / max(r[1], 1e-9))[:6]:
        print(f"{on:5.2f} {off:5.2f} {heard:15.0f} {lost:19.0f} "
              f"{saved / max(lost, 1e-9):15.1f}")
    print(f"  (doing nothing: {base:.0f} ad-seconds heard per hour, 0 content lost)")

    for budget in (10.0, 30.0, 60.0):
        ok = [r for r in rows if r[1] <= budget]
        if ok:
            best = max(ok, key=lambda r: r[0])
            print(f"  within {budget:.0f} content-seconds/hour: on={best[2]:.2f} off={best[3]:.2f} "
                  f"saves {best[0]:.0f} ad-seconds/hour")
            return {"budget": budget, "saved": best[0], "lost": best[1],
                    "on": best[2], "off": best[3]}
    worst_case = min(rows, key=lambda r: r[1])
    print(f"  No operating point keeps content loss under 60 s/h - the most conservative setting")
    print(f"  tried (on={worst_case[2]:.2f} off={worst_case[3]:.2f}) still wrongly mutes "
          f"{worst_case[1]:.0f} s/h.")
    print("  This is not a tuning problem. The model is not separating advertising from content")
    print("  well enough for any threshold to help, and more labelled breaks are what it needs.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    # default=None, not "": with action="append" argparse appends to whatever the default is,
    # and appending to a string raises rather than doing anything useful.
    ap.add_argument("--dir", default=None, action="append",
                    help="train on every labelled recording in this directory, holding out one "
                         "recording at a time. Repeat the flag to pool several directories - "
                         "desktop-stitched recordings and phone sessions live in different ones")
    ap.add_argument("--emb", default="", help="single recording: the .f16 embeddings")
    ap.add_argument("--labels", default="", help="single recording: Audacity label track")
    ap.add_argument("--joins", default="", help="join markers from stitch.py")
    ap.add_argument("--meta", default="", help=".jsonl beside the embeddings (for rms)")
    ap.add_argument("--context", type=int, default=10, help="frames of context per row")
    ap.add_argument("--list", action="store_true",
                    help="show which recordings would be used, with their times, and stop")
    ap.add_argument("--hours", default="", metavar="HH-HH",
                    help="keep only recordings whose midpoint falls in this local-time window, "
                         "e.g. 06-23. Windows may wrap midnight: 23-06 means the small hours")
    ap.add_argument("--exclude-hours", default="", metavar="HH-HH",
                    help="the opposite: drop recordings whose midpoint falls in the window, "
                         "e.g. --exclude-hours 00:30-05:30 to leave out the dead of night")
    ap.add_argument("--pca", type=int, default=0,
                    help="project onto this many principal components first (0 = no projection)")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-seconds", type=float, default=30.0)
    ap.add_argument("--on", type=float, default=0.6, help="start attenuating above this")
    ap.add_argument("--off", type=float, default=0.2, help="stop attenuating below this")
    ap.add_argument("--out", default="", help="write head weights here as JSON")
    ap.add_argument("--save-oof", default="", help="save out-of-fold predictions as .npz")
    args = ap.parse_args()

    if args.dir:
        recs = [r for d in args.dir for r in discover(Path(d))]
        if not recs:
            print(f"nothing labelled in {', '.join(args.dir)} - each recording needs a .f16 "
                  f"and one of {', '.join(LABEL_ORDER)}")
            return 1
    else:
        recs = [(Path(args.emb).with_suffix(""), Path(args.emb), Path(args.labels),
                 Path(args.joins) if args.joins else None)]

    # Time-of-day filtering, by the midpoint rather than the start: a half-hour recording that
    # begins at 05:50 is mostly daytime, and calling it night because of its first ten minutes
    # would be the wrong answer.
    keep_win = parse_window(args.hours) if args.hours else None
    drop_win = parse_window(args.exclude_hours) if args.exclude_hours else None
    if keep_win or drop_win:
        kept = []
        for r in recs:
            t = started_at(r[1])
            mid = t if t is None else frame_mid(t, r[1])
            if t is None:
                print(f"  keeping {r[0].name}: no timestamp to filter on")
                kept.append(r)
                continue
            if keep_win and not in_hours(mid, *keep_win):
                print(f"  dropped {r[0].name} ({mid:%H:%M}) - outside {args.hours}")
                continue
            if drop_win and in_hours(mid, *drop_win):
                print(f"  dropped {r[0].name} ({mid:%H:%M}) - inside {args.exclude_hours}")
                continue
            kept.append(r)
        recs = kept
        print(f"{len(recs)} recording(s) left after the time filter")
        print()
        if not recs:
            return 1

    parts_y, parts_g, names = [], [], []
    total_spans = 0
    # Total row count is known before anything is read: the embedding files are a fixed number of
    # bytes per frame. That lets the design matrix be allocated once and filled in place.
    n_total = sum(e.stat().st_size // (EMBEDDING_DIM * 2) for _, e, _, _ in recs)
    X_all = None
    at_row = 0
    print(f"{'recording':28s} {'when':>17s} {'min':>6s} {'breaks':>7s} {'ad frames':>10s}")
    for i, (stem, embp, labp, joinp) in enumerate(recs):
        n_frames = embp.stat().st_size // (EMBEDDING_DIM * 2)
        spans = read_spans(labp)
        joins = read_points(joinp) if joinp and joinp.exists() else []
        total_spans += len(spans)
        if args.list:
            # Reading 40 MB of embeddings to print one line would make --list as slow as the
            # thing it exists to avoid; the frame count is in the file size.
            yi = frame_labels(n_frames, spans, joins)
            parts_y.append(yi)
            names.append(stem.name)
            print(f"{stem.name:28s} {describe(embp, n_frames * HOP_S):>17s} "
                  f"{n_frames * HOP_S / 60:6.1f} {len(spans):7d} "
                  f"{int((yi == POSITIVE).sum()):10d}")
            continue

        emb = load_embeddings(embp)
        rms = load_rms(embp.with_suffix(".jsonl"))
        yi = frame_labels(len(emb), spans, joins)

        Xi = emb
        if rms is not None and len(rms) == len(emb):
            # .astype(float16) matters: hstack promotes to the wider dtype, so a float32 loudness
            # column silently doubles the whole design matrix. That promotion is what made the
            # build hold 10.7 GB and got the process killed.
            Xi = np.hstack([Xi, ((rms[:, None] + 60.0) / 60.0).astype(np.float16)])
        Xi = stack_context(Xi, args.context)
        # Written straight into the final matrix. Accumulating the parts first and copying them
        # afterwards means both exist at once, which is the same doubling by another route.
        if X_all is None:
            X_all = np.empty((n_total, Xi.shape[1]), dtype=Xi.dtype)
        X_all[at_row:at_row + len(Xi)] = Xi
        at_row += len(Xi)
        del Xi
        parts_y.append(yi)
        parts_g.append(np.full(len(yi), i, dtype=np.int32))
        names.append(stem.name)
        print(f"{stem.name:28s} {describe(embp, len(emb) * HOP_S):>17s} "
              f"{len(emb) * HOP_S / 60:6.1f} {len(spans):7d} "
              f"{int((yi == POSITIVE).sum()):10d}")

    if args.list:
        print()
        print(f"{total_spans} ad span(s) across {len(recs)} recording(s), "
              f"{sum(len(v) for v in parts_y) * HOP_S / 3600:.2f} h")
        return 0

    y = np.concatenate(parts_y)
    groups = np.concatenate(parts_g)
    X = X_all
    if at_row != len(y):
        print(f"row count mismatch: filled {at_row}, labels {len(y)}")
        return 1
    print()
    print(f"{total_spans} ad span(s) across {len(recs)} recording(s)")
    print(summarise(y))
    print(f"features: {X.shape[1]} per frame ({args.context} frames of context = "
          f"{(args.context - 1) * HOP_S + 0.975:.1f}s)")

    if (y == POSITIVE).sum() == 0:
        print("\nNo positive frames - nothing to learn. Check the label track's time range.")
        return 1

    keep = y != DROP
    if len(recs) > 1:
        from datetime import timedelta
        spans = []
        for _, embp, _, _ in recs:
            t = started_at(embp)
            n_f = embp.stat().st_size // (EMBEDDING_DIM * 2)
            spans.append((None, None) if t is None
                         else (t, t + timedelta(seconds=n_f * HOP_S)))
        grp = overlap_groups(recs, spans)
        n_groups = len(recs)
        if len(set(grp)) < len(recs):
            merged = {}
            for i, g in enumerate(grp):
                merged.setdefault(g, []).append(names[i])
            for g, members in merged.items():
                if len(members) > 1:
                    print(f"  same broadcast, folded together: {', '.join(members)}")
            groups = np.array([grp[g] for g in groups])
            names = [" + ".join(m) for _, m in sorted(merged.items())]
            # The fold count follows the groups, not the recordings. Without this the loop runs
            # one fold per recording while `names` has one per group, and walks off the end.
            n_groups = len(merged)
    if len(recs) > 1:
        # Hold out whole recordings. Blocks within one recording still share the same ad break,
        # the same presenters and the same hour of broadcast; only a different recording tests
        # whether anything was learnt beyond that.
        fold = groups
        n_folds = n_groups
        print(f"\nsplit: leave-one-recording-out ({n_folds} folds)")
    else:
        fold = blocks(len(y), max(int(args.block_seconds / HOP_S), 1), args.folds)
        n_folds = args.folds
        print(f"\nsplit: {n_folds}-fold by {args.block_seconds:.0f}s block within one recording")
    args.folds = n_folds

    # Out-of-fold predictions: every frame is scored by a model that never saw its recording.
    # Everything fitted to the data - the standardisation and the PCA basis as well as the
    # weights - is fitted inside the fold, on training rows only. Standardising the whole matrix
    # first is a small leak but a real one, and it is exactly the sort that makes a held-out
    # number quietly optimistic.
    oof = np.zeros(len(y), dtype=np.float32)
    for f in range(args.folds):
        tr = keep & (fold != f)
        held = names[f] if len(recs) > 1 else f"block fold {f}"
        if (y[tr] == POSITIVE).sum() == 0 or (y[tr] == 0).sum() == 0:
            print(f"  {held}: one class missing in training - skipped")
            oof[fold == f] = 0.0
            continue
        tr_idx = np.flatnonzero(tr)
        te_idx = np.flatnonzero(fold == f)
        mu, sd = mean_std(X, tr_idx)
        w = fit(X, tr_idx, y[tr].astype(np.float64), mu, sd, args.l2, args.iters)
        oof[te_idx] = predict(X, te_idx, w, mu, sd)
        if (keep & (fold == f)).sum():
            report(y[fold == f], oof[fold == f], args.on, args.off, f"  held out: {held}")

    keep_idx = np.flatnonzero(keep)
    mu_full, sd_full = mean_std(X, keep_idx)
    w_full = fit(X, keep_idx, y[keep].astype(np.float64), mu_full, sd_full, args.l2, args.iters)
    report(y, predict(X, np.arange(len(y)), w_full, mu_full, sd_full), args.on, args.off,
           "FIT (trained on everything - memorisation)")
    split_name = ("leave-one-recording-out" if len(recs) > 1
                  else f"{args.folds}-fold by {args.block_seconds:.0f}s block")
    m = report(y, oof, args.on, args.off, f"HELD OUT, POOLED ({split_name})")
    best_point = sweep(y, oof, args.on - args.off)
    if args.save_oof:
        np.savez(args.save_oof, y=y, p=oof, groups=groups, names=np.array(names))
        print()
        print(f"out-of-fold predictions saved to {args.save_oof} - sweeping thresholds again")
        print("needs no refitting.")

    n_breaks = total_spans
    print("\nVERDICT")
    if n_breaks < 3:
        print(f"  Only {n_breaks} ad break(s) in this data. Every fold's test half contains part")
        print("  of the same break the model trained on, so the held-out numbers above are")
        print("  optimistic by an unknown amount. This run shows the pipeline works end to end;")
        print("  it does not show the model works. That needs recordings from several days.")
    elif len(recs) > 1 and len({n[:13] for n in names}) < 2:
        print(f"  {n_breaks} breaks across {len(recs)} recordings, but all from the same day and")
        print("  the same few hours - the same advertisements recur, so a held-out recording is")
        print("  not really unseen. Treat these numbers as an upper bound until a second day of")
        print("  recording exists."
              )
    elif best_point and best_point["budget"] <= 30.0:
        print(f"  Usable. At on={best_point['on']:.2f} off={best_point['off']:.2f} this saves "
              f"{best_point['saved']:.0f} ad-seconds per hour")
        print(f"  while wrongly muting {best_point['lost']:.0f} - a ratio of "
              f"{best_point['saved'] / max(best_point['lost'], 1e-9):.0f} to 1.")
        print("  Worth shipping to the phone and listening to.")
    elif best_point:
        print(f"  Borderline. The best operating point loses {best_point['lost']:.0f} "
              f"content-seconds per hour, above the plan's budget of 10.")
        print("  Worth trying on the phone, but expect to notice the false mutes.")
    else:
        print("  Held-out performance is not yet usable. More labelled breaks before more model.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "type": "logistic_regression",
            "context_frames": args.context,
            "uses_rms": bool(rms is not None and len(rms) == len(emb)),
            "input_dim": int(X.shape[1]),
            "mean": mu_full.tolist(), "scale": sd_full.tolist(),
            "weights": w_full[:-1].tolist(), "bias": float(w_full[-1]),
            "on": args.on, "off": args.off,
            "trained_on": {"recordings": names, "ad_breaks": n_breaks},
            "held_out": m,
        }, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
