#!/usr/bin/env python3
"""
How much of the held-out score is recognising advertisements the model has heard before?

Leave-one-recording-out holds out a recording, but not the advertisements in it: this station runs
the same spots for weeks, so most of a held-out break has usually aired in some training recording
too. A model can score well on it by recognising the jingle rather than by knowing what a break
sounds like. Both are useful in the app - the spots do repeat - but only the second generalises to
a new campaign, and the pooled numbers cannot tell them apart.

HOW A REPEAT IS RECOGNISED: BY ITS WORDS. A spot that airs twice is transcribed as nearly the same
sentence twice. So every word said inside a labelled break is checked: does a three-word phrase
containing it also occur inside a labelled break in some OTHER fold? If so it is a repeat. An
advertising frame is then

    repeated  if any word it overlaps is a repeat,
    novel     if it overlaps words and none of them is,
    silent    if it overlaps no words at all (music, jingles) - counted separately, undecided.

Comparing the audio directly was tried first and does not work. A spot that airs twice lands at a
different offset against YAMNet's 0.48 s grid each time, so even identical audio gives noticeably
different embeddings, and every advertisement resembles every other one (compressed speech over
music): the similarities form one smooth hump with no gap between "the same spot" and "another
spot". Text has no such problem; three words in the same order rarely recur by chance.

    python trainer/novelty.py captures/kaggle/dataset/pool.npz captures/heads/oof_k40_*.npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import HOP_S, POSITIVE, WINDOW_S                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIRS = [ROOT / "captures" / "stitched", ROOT / "captures" / "sessions"]
_NOT_WORD = re.compile(r"[^\w]+", re.UNICODE)
_NIQQUD = re.compile(r"[֑-ׇ]")


def norm(w: str) -> str:
    return _NOT_WORD.sub("", _NIQQUD.sub("", w)).lower()


def words_of(rec: str) -> list[tuple[str, float, float]]:
    path = next(d / f"{rec}.words.json" for d in DIRS if (d / f"{rec}.words.json").exists())
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for seg in doc["segments"]:
        for w in seg.get("words", []):
            t = norm(w["w"])
            if t:
                out.append((t, float(w["s"]), float(w["e"])))
    return out


def classify(pool, n_gram: int):
    """Per frame: 1 repeated, 0 novel, -1 silent or not advertising."""
    y, fold, names, bounds = pool["y"], pool["fold"], pool["rec_names"], pool["rec_bounds"]
    per_rec = []                                   # (rec index, words inside ads)
    for i, (rec, (a, b)) in enumerate(zip(names, bounds)):
        ad_words = []
        for t, s, e in words_of(str(rec)):
            k = a + int(round(((s + e) / 2 - WINDOW_S / 2) / HOP_S))
            if a <= k < b and y[k] == POSITIVE:
                ad_words.append((t, s, e))
        per_rec.append(ad_words)
    # Which folds each phrase is said in, counting only runs of words that are all inside ads.
    seen: dict[tuple, set] = {}
    for i, ws in enumerate(per_rec):
        f = int(fold[bounds[i][0]])
        for j in range(len(ws) - n_gram + 1):
            seen.setdefault(tuple(w[0] for w in ws[j:j + n_gram]), set()).add(f)

    state = np.full(len(y), -1, np.int8)
    for i, ws in enumerate(per_rec):
        a, b = bounds[i]
        f = int(fold[a])
        rep = np.zeros(len(ws), bool)
        for j in range(len(ws) - n_gram + 1):
            if seen[tuple(w[0] for w in ws[j:j + n_gram])] - {f}:
                rep[j:j + n_gram] = True
        for (t, s, e), r in zip(ws, rep):
            # every frame whose window overlaps the word
            k0 = a + max(0, int(np.floor((s - WINDOW_S) / HOP_S)) + 1)
            k1 = a + int(np.floor(e / HOP_S))
            for k in range(max(k0, a), min(k1, b - 1) + 1):
                if y[k] == POSITIVE:
                    state[k] = 1 if (r or state[k] == 1) else 0
    return state


def breaks_of(y: np.ndarray):
    d = np.diff(np.concatenate(([0], (y == POSITIVE).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("pool", help=".npz written by train.py --export")
    ap.add_argument("oof", nargs="+", help="out-of-fold predictions to judge")
    ap.add_argument("--ngram", type=int, default=3, help="words per compared phrase")
    ap.add_argument("--novel-share", type=float, default=0.5,
                    help="a break counts as novel when at least this share of its spoken "
                         "advertising frames is")
    args = ap.parse_args()

    pool = np.load(args.pool, allow_pickle=True)
    y = pool["y"]
    state = classify(pool, args.ngram)
    ad = y == POSITIVE
    rep, nov, sil = ad & (state == 1), ad & (state == 0), ad & (state == -1)
    print(f"{int(ad.sum())} advertising frames: {100 * rep.sum() / ad.sum():.0f}% repeated "
          f"(a {args.ngram}-word phrase also said in another fold's ads), "
          f"{100 * nov.sum() / ad.sum():.0f}% novel, {100 * sil.sum() / ad.sum():.0f}% silent")

    brk = breaks_of(y)
    novel_breaks = []
    for a, b in brk:
        spoken = rep[a:b].sum() + nov[a:b].sum()
        if spoken and nov[a:b].sum() / spoken >= args.novel_share:
            novel_breaks.append((a, b))
    print(f"{len(brk)} labelled stretches of advertising; {len(novel_breaks)} are mostly novel "
          f"(>= {100 * args.novel_share:.0f}% of their spoken frames), "
          f"{sum(b - a for a, b in novel_breaks) * HOP_S / 60:.1f} min in all")

    print(f"\nframe recall at p >= 0.5")
    print(f"{'predictions':24s} {'repeated':>9s} {'novel':>7s} {'silent':>7s}   "
          f"{'novel stretches caught':>23s} {'their cover':>12s}")
    for f in args.oof:
        o = np.load(f, allow_pickle=True)
        if len(o["p"]) != len(y):
            print(f"{Path(f).name:24s} (made from a different pool - skipped)")
            continue
        hit = o["p"] >= 0.5
        pct = lambda m: 100 * hit[m].mean() if m.any() else float("nan")   # noqa: E731
        caught = sum(hit[a:b].any() for a, b in novel_breaks)
        cover = np.mean([hit[a:b].mean() for a, b in novel_breaks]) if novel_breaks else np.nan
        print(f"{Path(f).stem:24s} {pct(rep):8.1f}% {pct(nov):6.1f}% {pct(sil):6.1f}%   "
              f"{caught:12d} of {len(novel_breaks):<8d} {100 * cover:11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
