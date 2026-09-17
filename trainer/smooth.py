#!/usr/bin/env python3
"""
Turning a stream of per-frame probabilities into a mute/unmute decision.

THE OBSERVATION THIS RESTS ON. An ad break lasts far longer than one frame. The head scores each
0.48 s window independently and then throws that structure away, so its mistakes look nothing like
its successes: measured over the morning pool, false-positive runs have a median of 1.0 s and a
90th percentile of 4.8 s, while real breaks have a median of 49 s and a 10th percentile of 14 s.
Almost every error is shorter than almost every break. A decider that knows how long things last
can separate them; one that looks at a single frame cannot.

WHERE THE LOSS ACTUALLY IS. With per-frame hysteresis, 77 % of the advertising still heard is gaps
punched in the middle of breaks the model already caught - the 0111110111 pattern - not late
detection (21 %) and not missed breaks (2 %). So the thing worth fixing is the gaps.

WHAT IS AND IS NOT POSSIBLE. Deciding a boundary *retrospectively* is free offline and worthless
in the app: audio already sent to the headphones cannot be unplayed. Every rule here is therefore
causal, using only frames already seen, and each keeps O(1) or O(n) state so it ports to the phone
as a few lines in the audio callback. The cost of causality is entry latency, which is a few
seconds of ad at the start of each break.

Six rules, cheapest state first:

    hysteresis  one boolean.  Mute above `on`, release below `off`. No smoothing.
    ema         one float.    Exponentially-weighted average of the score, then hysteresis.
    forward     two floats.   A 2-state HMM filtered forward.
    dwell       one counter.  Hysteresis that cannot change its mind for N seconds.
    rolling     a ring of n.  Plain average of the score over n frames, then hysteresis. BEST.
    majority    a ring of n.  Round each frame to yes/no FIRST, then count the yeses. WORST.

NEVER ROUND BEFORE YOU SMOOTH. `rolling` and `majority` are the same window over the same frames;
the only difference is that majority collapses each score to 0 or 1 before combining them. At an
identical 11 content-seconds lost per hour that one choice costs 115 ad-seconds per hour (441
against 326) and triples the flicker (17 toggles an hour against 52). Rounding discards exactly
the information the averaging needs: a frame the model is agonising over at 0.51 should not carry
the same weight as one it is certain of at 0.999, and after rounding it does. The number is the
model's whole output; the verdict is a summary of it, and summarising before averaging is throwing
the evidence away and keeping the conclusion.

The rest of the ordering, all at 11 content-seconds lost: rolling 441, ema 440, hysteresis 434,
forward 415, majority 326. `dwell` reaches the front only far past any sensible budget.

The largest single win needs no window at all. Lowering `off` - staying muted unless the model is
actively confident content has resumed - takes mid-break gaps from 141 s/h at off=0.80 to 80 at
off=0.50 to 42 at off=0.20, while *reducing* flicker. That is one number in a config file.

Usage:
    python trainer/smooth.py captures/oof_33rec.npz --dir captures/stitched --hours 07-11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import HOP_S, POSITIVE                                    # noqa: E402


def hysteresis(p: np.ndarray, on: float, off: float) -> np.ndarray:
    """Mute above `on`, release below `off`. One boolean of state.

    The asymmetry is the whole point: entering costs you content if you are wrong, staying costs
    you almost nothing, because you are probably still inside a break.
    """
    out = np.zeros(len(p), bool)
    state = False
    for i, v in enumerate(p):
        state = v >= off if state else v >= on
        out[i] = state
    return out


def majority(p: np.ndarray, thr: float, n: int, need: int | None = None) -> np.ndarray:
    """Round the last `n` frames all up or all down: mute when at least `need` of them voted yes.

    This is the simplest thing that uses run length, and it is the weakest of the four, for two
    reasons. It throws away the probabilities, keeping only a yes/no per frame, so a frame at 0.999
    and one at 0.51 count the same. And it is symmetric - the same window that delays the entry
    delays the exit, so it does not fill mid-break gaps any better than it suppresses spikes.
    """
    need = need if need is not None else n // 2 + 1
    b = (p >= thr).astype(np.int32)
    c = np.cumsum(np.concatenate(([0], b)))
    lo = np.maximum(np.arange(len(b)) - n + 1, 0)
    return (c[1:] - c[lo]) >= need


def rolling(p: np.ndarray, n: int, on: float, off: float) -> np.ndarray:
    """Average the *score* over the last n frames, then apply the thresholds to the average.

    This is `majority` done without throwing the scores away first. Majority rounds each frame to
    yes/no and counts the yeses, so a frame the model is agonising over at 0.51 carries exactly the
    weight of one it is certain about at 0.999. Averaging keeps that difference, which is the whole
    reason the model bothers to emit a number instead of a verdict.

    DOES AVERAGING BLUR THE EXTREMES? Barely, and what blurring there is helps. The head's scores
    are strongly bimodal - it saturates - so inside a real break most frames sit at ~1.0 and their
    mean is still ~1.0: the median inside advertising falls only from 0.995 to 0.985. The mean
    lands in the middle only where consecutive frames *disagree*, which is at break boundaries and
    on isolated spikes, and being hesitant in exactly those places is the point.

    Measured over the morning pool, the mean is MORE selective than a single frame at every
    threshold - share of ad frames above it, divided by share of content frames above it:

        threshold      single frame      mean of 11
             0.50            32 : 1          38 : 1
             0.70            45 : 1          70 : 1
             0.90            76 : 1         160 : 1
             0.99           152 : 1         371 : 1

    So thresholds do not need lowering to compensate for averaging; that intuition is wrong here.
    What does change is `off`. The mute is released when the *average* drops below it, and the
    average decays slowly after a break ends, so an `off` tuned for single frames holds the mute
    far too long into the programme. The first sweep of this rule searched only off <= 0.50 and
    every combination therefore lost 35-44 content-seconds an hour; the useful value is 0.70.
    """
    c = np.cumsum(np.concatenate(([0.0], p.astype(np.float64))))
    lo = np.maximum(np.arange(len(p)) - n + 1, 0)
    avg = (c[1:] - c[lo]) / (np.arange(len(p)) - lo + 1)
    return hysteresis(avg, on, off)


def ema(p: np.ndarray, half_life_s: float, on: float, off: float) -> np.ndarray:
    """The same idea with one float of state: an exponentially-weighted average of the score.

    Cheaper than `rolling` - no ring buffer - and it fades old evidence smoothly instead of having
    it drop off a cliff n frames later.
    """
    a = 0.5 ** (HOP_S / max(half_life_s, 1e-6))
    out = np.empty(len(p), np.float64)
    s = 0.0
    for i, v in enumerate(p):
        s = a * s + (1 - a) * float(v)
        out[i] = s
    return hysteresis(out, on, off)


def dwell(p: np.ndarray, on: float, off: float,
          min_mute_s: float = 20.0, min_clear_s: float = 2.0) -> np.ndarray:
    """Hysteresis that is not allowed to change its mind again for a while. One counter of state.

    Aimed squarely at the dominant loss. Once a break is confirmed, the mute holds for at least
    `min_mute_s` no matter what the probabilities do, so a run of weak frames in the middle of a
    break - a quiet passage, a voice-over the model has not seen before - cannot punch a hole in it.
    The median break is 49 s, so a 20 s floor is well inside one and cannot bridge two.

    `min_clear_s` does the same in reverse and should stay small: holding the unmute open is how
    you miss the start of the next break.
    """
    hold_mute = max(int(round(min_mute_s / HOP_S)), 1)
    hold_clear = max(int(round(min_clear_s / HOP_S)), 1)
    out = np.zeros(len(p), bool)
    state, left = False, 0
    for i, v in enumerate(p):
        if left > 0:
            left -= 1
        elif state:
            if v < off:
                state, left = False, hold_clear
        elif v >= on:
            state, left = True, hold_mute
        out[i] = state
    return out


def forward(p: np.ndarray, mean_ad_s: float = 60.0, mean_content_s: float = 300.0,
            temp: float = 0.03, cut: float = 0.9) -> np.ndarray:
    """A 2-state HMM, filtered forward. Two floats of state, and the best of the four.

    The transitions say what the run lengths are - a break lasts about `mean_ad_s`, content about
    `mean_content_s` - and the filter accumulates the head's evidence against that expectation.
    A lone confident frame barely moves the posterior, because the prior says breaks are long and
    one frame is not a break; a sustained run moves it decisively. That is the proposal, done
    without a window and without discarding the probabilities.

    `temp` is not a detail, it is the difference between this working and not working. The head is
    wildly overconfident, as high-dimensional logistic regression always is: it saturates, so a
    single frame contributes up to 14 units of log-evidence, and ten such frames contribute 140 -
    which no realistic transition prior can outvote. Untempered (temp=1.0) this rule loses 66
    content-seconds per hour, far worse than the hysteresis it replaces. At temp=0.03 it loses 10
    and saves more. Tempering is how you stop the model's confidence overriding what you know
    about how long advertising lasts.
    """
    q = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    e = (np.log(q) - np.log1p(-q)) * temp              # log-odds, tempered
    stay_ad = 1.0 - 1.0 / max(mean_ad_s / HOP_S, 1.0)
    stay_con = 1.0 - 1.0 / max(mean_content_s / HOP_S, 1.0)
    lA = np.log(np.array([[stay_con, 1 - stay_con], [1 - stay_ad, stay_ad]]))
    out = np.empty(len(p), np.float32)
    a = np.array([0.0, -10.0])                         # start in content
    for i in range(len(p)):
        a = np.logaddexp(a[0] + lA[0], a[1] + lA[1])
        a[1] += e[i]
        a -= a.max()
        out[i] = 1.0 / (1.0 + np.exp(a[0] - a[1]))
    return out >= cut


# ---------------------------------------------------------------- evaluation

def decompose(y, muted, hours):
    """Ad-seconds heard per hour, split by cause. The three have different remedies: latency is
    irreducible for a causal rule, gaps want a longer hold, missed breaks want a better model."""
    d = np.diff(np.concatenate(([0], (y == POSITIVE).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lat, missed_s, gap_s, lats = 0.0, 0.0, 0.0, []
    for a, b in zip(starts, ends):
        inside = np.flatnonzero(muted[a:b])
        if not len(inside):
            missed_s += (b - a) * HOP_S
            continue
        lats.append(inside[0] * HOP_S)
        lat += inside[0] * HOP_S
        gap_s += int((~muted[a + inside[0]:b]).sum()) * HOP_S
    # Toggles per hour. Nothing else here captures flicker, and flicker is what the ear objects
    # to: a mute that opens and shuts four times inside one break is more irritating than one that
    # engages three seconds late and then holds. Two rules can post identical ad-seconds and
    # content-seconds and feel completely different because of this number.
    toggles = int(np.count_nonzero(np.diff(muted.astype(np.int8))))
    return {"latency": lat / hours, "gaps": gap_s / hours, "missed": missed_s / hours,
            "median_latency_s": float(np.median(lats)) if lats else float("nan"),
            "toggles": toggles / hours,
            "breaks": len(starts), "caught": len(lats)}


def score(y, muted, hours, base):
    heard = int(((y == POSITIVE) & ~muted).sum()) * HOP_S / hours
    lost = int(((y == 0) & muted).sum()) * HOP_S / hours
    return base - heard, lost


def main() -> int:
    from sweep import hour_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("oof", help=".npz written by train.py --save-oof")
    ap.add_argument("--dir", action="append", default=None,
                    help="where the recordings live; repeat for several directories")
    ap.add_argument("--hours", default="", metavar="HH-HH",
                    help="judge only recordings whose midpoint falls in this window")
    ap.add_argument("--budget", type=float, default=10.0,
                    help="content-seconds per hour you are willing to lose")
    args = ap.parse_args()

    d = np.load(args.oof, allow_pickle=True)
    y, p, groups, names = d["y"], d["p"], d["groups"], d["names"]
    if args.hours:
        m = hour_mask(names, groups, [Path(x) for x in args.dir], args.hours)
        y, p = y[m], p[m]
    hours = len(y) * HOP_S / 3600
    base = int((y == POSITIVE).sum()) * HOP_S / hours
    print(f"{hours:.2f} h, doing nothing = {base:.0f} ad-sec/h, "
          f"budget = {args.budget:.0f} content-sec/h lost\n")

    cands = []
    for on in (0.999, 0.99, 0.95):
        for off in (0.99, 0.95, 0.8, 0.5, 0.2):
            if off <= on:
                cands.append((f"hysteresis on={on} off={off}", hysteresis(p, on, off)))
    for thr in (0.5, 0.9, 0.99):
        for n in (5, 11, 21, 41):
            cands.append((f"majority thr={thr} n={n}({n * HOP_S:.0f}s)", majority(p, thr, n)))
    for on in (0.999, 0.99):
        for off in (0.8, 0.5, 0.2):
            for hold in (10.0, 20.0, 30.0, 45.0):
                cands.append((f"dwell on={on} off={off} hold={hold:.0f}s",
                              dwell(p, on, off, hold)))
    for n in (3, 5, 11, 21, 41):
        for on in (0.999, 0.99, 0.97, 0.95, 0.9, 0.8):
            for off in (0.7, 0.5, 0.3, 0.15, 0.05):
                if off <= on:
                    cands.append((f"rolling n={n}({n * HOP_S:.0f}s) on={on} off={off}",
                                  rolling(p, n, on, off)))
    for hl in (0.5, 1.0, 2.0, 4.0, 8.0):
        for on in (0.999, 0.99, 0.97, 0.95, 0.9, 0.8):
            for off in (0.7, 0.5, 0.3, 0.15, 0.05):
                if off <= on:
                    cands.append((f"ema half-life={hl}s on={on} off={off}",
                                  ema(p, hl, on, off)))
    for mad in (45.0, 60.0, 90.0):
        for temp in (0.1, 0.03, 0.01):
            for cut in (0.5, 0.9, 0.99):
                cands.append((f"forward ad~{mad:.0f}s T={temp} cut={cut}",
                              forward(p, mad, 300.0, temp, cut)))

    rows = [(nm, mu, *score(y, mu, hours, base)) for nm, mu in cands]

    # The Pareto front, not the best-within-a-budget. Comparing each family at its own preferred
    # loss level flatters whichever one happens to land just under the line; the front asks the
    # only fair question, which is who saves most at each level of content actually sacrificed.
    print("PARETO FRONT - who saves most at each level of content lost")
    print(f"{'rule':38s} {'saved/h':>8s} {'lost/h':>7s}   "
          f"{'latency':>8s} {'gaps':>6s} {'missed':>7s} {'med lat':>8s} {'toggles/h':>10s}")
    print("-" * 104)
    front, best = [], -1.0
    for nm, mu, saved, lost in sorted(rows, key=lambda r: (r[3], -r[2])):
        if saved > best:
            best = saved
            front.append((nm, mu, saved, lost))
    for nm, mu, saved, lost in front:
        if lost > max(args.budget * 4, 40):
            break
        c = decompose(y, mu, hours)
        print(f"{nm:38s} {saved:8.0f} {lost:7.0f}   "
              f"{c['latency']:8.0f} {c['gaps']:6.0f} {c['missed']:7.0f} "
              f"{c['median_latency_s']:7.1f}s {c['toggles']:10.0f}")

    print(f"\nBEST OF EACH FAMILY within {args.budget:.0f} content-sec/h lost")
    ok = [r for r in rows if r[3] <= args.budget]
    seen = set()
    for nm, mu, saved, lost in sorted(ok, key=lambda r: -r[2]):
        fam = nm.split()[0]
        if fam in seen:
            continue
        seen.add(fam)
        c = decompose(y, mu, hours)
        print(f"{nm:38s} {saved:8.0f} {lost:7.0f}   "
              f"{c['latency']:8.0f} {c['gaps']:6.0f} {c['missed']:7.0f} "
              f"{c['median_latency_s']:7.1f}s {c['toggles']:10.0f}")
    if not ok:
        print("  nothing meets that budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
