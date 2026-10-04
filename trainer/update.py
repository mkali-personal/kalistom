#!/usr/bin/env python3
"""
One command from "new audio on the phone" to "how good is the head now".

    python trainer/update.py                     # pull, check, train (quick), report
    python trainer/update.py --ship              # ... and install the new head on the phone
    python trainer/update.py --full              # every day held out in turn, on Kaggle
    python trainer/update.py --no-pull           # the phone is not attached; use what is here
    python trainer/update.py --no-train          # report on the last run without retraining

The steps, in order:

  1. PULL. Sessions come off the phone and are deleted there once they arrive intact
     (ingest.pull). A session whose WAV header was never finalised - one still recording, or one
     the app did not close - has the header repaired here, keeping a .bak.
  2. INVENTORY. Every recording with embeddings but no labels is listed, with how to label it.
     Training goes ahead on the labelled ones; the rest join the next run once labelled.
  3. TRAIN, in one of two modes.
       quick (default)  Two fits on this machine. One holds out the newest broadcast days, at
                        least --test-hours of them, and is scored on them. The other is fitted on
                        everything and written to captures/heads/head_<name>.json, ready to ship.
                        A few hours of test audio catches a broken model; it is too little to
                        tune thresholds or to choose between two heads.
       --full           One fold per broadcast day, every day held out in turn, on Kaggle by
                        default (a few minutes) or locally with --backend local (where each day's
                        predictions are cached, so only days that gained a recording are refitted).
                        Use it before changing thresholds or the kind of head. It writes no head
                        to ship unless given --final as well.
  4. REPORT. The held-out predictions are judged at the thresholds the phone is using now, from
     head_weights.json, both over all hours and over --hours. Each run is appended to
     captures/heads/history.jsonl and the whole history is printed, so you can see whether the
     new data moved anything. The ROC plot is redrawn beside the predictions.
  5. SHIP (only with --ship). Installs the head from step 3 on the phone with trainer/ship.py,
     keeping the phone's thresholds. Without --ship nothing on the phone changes.

Changing the head itself - another size, another architecture - is still done by hand in train.py
or kaggle_run.py, compared on the same day folds, and shipped only if it wins.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import HOP_S, NEGATIVE, POSITIVE                          # noqa: E402
from roc_threshold import recording_of_row                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "trainer"
DIRS = [ROOT / "captures" / "stitched", ROOT / "captures" / "sessions"]
HEADS = ROOT / "captures" / "heads"
HISTORY = HEADS / "history.jsonl"
HEAD_WEIGHTS = ROOT / "app" / "src" / "main" / "assets" / "head_weights.json"
# Kept literal, as label_editor.py does, so this starts without importing the training stack.
LABEL_ORDER = (".truth.txt", ".gemini.txt", ".claude.txt", ".draft.txt")
# The shipped head's configuration (see head_weights.json and the commit that shipped it).
DEFAULT_HEAD = "head=mlp hidden=16 wd=1e-2"


def step(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * (70 - len(title))}", flush=True)


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


# ---------------------------------------------------------------- 1. pull

def device_attached() -> bool:
    from ingest import adb
    try:
        r = subprocess.run([adb(), "devices"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.strip().endswith("device") for line in r.stdout.splitlines()[1:])


def pull() -> None:
    import wave
    from ingest import DEFAULT_LOCAL, pull as ingest_pull
    if not device_attached():
        print("no phone attached - skipping the pull and using what is already here")
        return
    dest = ROOT / DEFAULT_LOCAL
    ingest_pull(dest)
    # A header that still says zero samples is a session that was never closed. The audio is all
    # there; only the sizes have to be recomputed.
    stale = []
    for wav in sorted(dest.glob("*.wav")):
        try:
            with wave.open(str(wav), "rb") as w:
                declared = w.getnframes()
        except Exception:                                       # noqa: BLE001
            continue
        if declared == 0 and wav.stat().st_size > 44:
            stale.append(wav)
    if stale:
        run([sys.executable, TRAINER / "repair_wav.py", *stale])


# ---------------------------------------------------------------- 2. inventory

def inventory() -> list[Path]:
    """Recordings that have embeddings but no labels of any kind."""
    unlabelled, labelled = [], 0
    for d in DIRS:
        for emb in sorted(d.glob("*.f16")):
            stem = emb.with_suffix("")
            if any(stem.with_name(stem.name + s).exists() for s in LABEL_ORDER):
                labelled += 1
            else:
                unlabelled.append(emb)
    print(f"{labelled} labelled recording(s)")
    if unlabelled:
        print(f"{len(unlabelled)} not labelled yet - they are left out of this run:")
        for emb in unlabelled:
            minutes = emb.stat().st_size // (1024 * 2) * HOP_S / 60
            print(f"  {emb.parent.name}/{emb.stem}  {minutes:6.1f} min")
        print("label them with:  python trainer/label_editor.py")
    return unlabelled


# ---------------------------------------------------------------- 3. train

def spec_args(spec: str) -> list[str]:
    """'head=mlp hidden=16 wd=1e-2' -> ['--head', 'mlp', '--hidden', '16', '--wd', '1e-2']"""
    out = []
    for pair in spec.split():
        k, v = pair.split("=", 1)
        out += [f"--{k}", v]
    return out


def predictions_path(name: str, full: bool) -> Path:
    return HEADS / (f"oof_day_{name}.npz" if full else f"quick_{name}.npz")


def train_quick(name: str, spec: str, on: float, off: float, test_hours: float) -> Path:
    """One held-out fit on the newest days and one fit on everything, in a single train.py run
    so the embeddings are read once. The second fit is the head to ship."""
    oof, head = predictions_path(name, False), HEADS / f"head_{name}.json"
    HEADS.mkdir(parents=True, exist_ok=True)
    run([sys.executable, TRAINER / "train.py", *[a for d in DIRS for a in ("--dir", d)],
         *spec_args(spec), "--test-newest-hours", test_hours, "--on", on, "--off", off,
         "--save-oof", oof, "--out", head])
    if not oof.exists() or not head.exists():
        raise SystemExit("training finished without writing its predictions and head")
    return oof


def train_full(name: str, spec: str, backend: str, on: float, off: float) -> Path:
    oof = predictions_path(name, True)
    HEADS.mkdir(parents=True, exist_ok=True)
    if backend == "kaggle":
        run([sys.executable, TRAINER / "kaggle_run.py", "--export",
             "--run", f"day_{name} {spec}"])
    else:
        run([sys.executable, TRAINER / "train.py", *[a for d in DIRS for a in ("--dir", d)],
             *spec_args(spec), "--fold-by", "day", "--on", on, "--off", off,
             "--save-oof", oof, "--fold-cache", HEADS / f"folds_day_{name}"])
    if not oof.exists():
        raise SystemExit(f"training finished but {oof} is missing")
    return oof


# ---------------------------------------------------------------- 4. report

def judge(y, p, rec, on: float, off: float) -> dict:
    """The phone's rule on held-out scores, reset at every recording, in the units you notice."""
    from roc_threshold import hysteresis, segment_floor
    from smooth import decompose
    muted = hysteresis(p, segment_floor(rec), on, off)
    hours = len(y) * HOP_S / 3600
    ads, content = y == POSITIVE, y == NEGATIVE
    base = int(ads.sum()) * HOP_S / hours
    heard = int((ads & ~muted).sum()) * HOP_S / hours
    c = decompose(y, muted, hours)
    return {"hours": round(hours, 2), "breaks": c["breaks"], "caught": c["caught"],
            "ad_s_per_h": round(base), "saved_per_h": round(base - heard),
            "lost_per_h": round(int((content & muted).sum()) * HOP_S / hours, 1),
            "median_entry_s": round(c["median_latency_s"], 1),
            "toggles_per_h": round(c["toggles"], 1),
            "tpr": round(float(muted[ads].mean()), 4),
            "fpr": round(float(muted[content].mean()), 5)}


def report(oof: Path, name: str, spec: str, on: float, off: float, window: str,
           mode: str) -> None:
    from sweep import hour_mask
    d = np.load(oof, allow_pickle=True)
    y, p, groups = d["y"], d["p"].astype(np.float64), d["groups"]
    rec = recording_of_row(d)
    # The quick split scores only its test days; everything else was training and has no score.
    scored = d["scored"] if "scored" in d.files else np.ones(len(y), bool)
    fold_names = list(d["fold_names"]) if "fold_names" in d.files else None
    n_days = len(fold_names) if fold_names and mode == "full" else None
    rows = {"all hours": judge(y[scored], p[scored], rec[scored], on, off)}
    if window:
        m = hour_mask(d["names"], groups, DIRS, window) & scored
        if m.any():
            rows[window] = judge(y[m], p[m], rec[m], on, off)

    if mode == "quick":
        print(f"{oof.name}: tested on {fold_names[0]} ({int(scored.sum() * HOP_S / 60)} min), "
              f"trained on the earlier days, judged at the phone's on={on:g} off={off:g}\n")
    else:
        print(f"{oof.name}: {len(d['rec_names'])} recordings"
              + (f" in {n_days} day folds" if n_days else "")
              + f", judged at the phone's on={on:g} off={off:g}\n")
    cols = ("hours", "breaks", "caught", "ad_s_per_h", "saved_per_h", "lost_per_h",
            "median_entry_s", "toggles_per_h")
    print(f"{'':10s}" + "".join(f"{c:>15s}" for c in cols))
    for label, r in rows.items():
        print(f"{label:10s}" + "".join(f"{r[c]:>15}" for c in cols))

    entry = {"when": time.strftime("%Y-%m-%d %H:%M"), "mode": mode, "name": name, "spec": spec,
             "recordings": len(d["rec_names"]), "days": n_days,
             "tested_on": fold_names[0] if mode == "quick" else "every day",
             "on": on, "off": off, "results": rows}
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    history = [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    key = window if window else "all hours"
    print(f"\nHISTORY ({key}; saved and lost are seconds per hour of test audio)")
    print(f"{'when':17s} {'mode':6s} {'name':12s} {'recs':>5s} {'test hours':>11s} "
          f"{'saved':>6s} {'lost':>6s} {'entry s':>8s} {'on':>7s} {'off':>5s}  tested on")
    for h in history:
        r = h["results"].get(key) or h["results"]["all hours"]
        print(f"{h['when']:17s} {h.get('mode', 'full'):6s} {h['name']:12s} "
              f"{h['recordings']:5d} {r['hours']:11.1f} {r['saved_per_h']:6d} "
              f"{r['lost_per_h']:6.1f} {r['median_entry_s']:8.1f} {h['on']:7g} {h['off']:5g}  "
              f"{h.get('tested_on', 'every recording' if h.get('days') is None else 'every day')}")
    print("Compare rows of the same mode at the same thresholds. A quick row is tested on a few "
          "hours, so expect it to move by tens of seconds from run to run.")

    roc = [sys.executable, TRAINER / "roc_threshold.py", oof, "--off", off]
    if window:
        roc += [a for d_ in DIRS for a in ("--dir", d_)] + ["--hours", window,
                "--out", oof.with_name(f"roc_{oof.stem}_{window}.png")]
    try:
        run(roc)
    except subprocess.CalledProcessError:
        # Usually the old PNG is open in an image viewer, which on Windows locks it. The numbers
        # above are the result; the plot is a convenience and must not fail the whole update.
        print("could not redraw the ROC plot (is the old one open in a viewer?) - carrying on")


# ---------------------------------------------------------------- 5. final

def final(name: str, spec: str, on: float, off: float) -> Path:
    out = HEADS / f"head_{name}.json"
    run([sys.executable, TRAINER / "train.py", *[a for d in DIRS for a in ("--dir", d)],
         *spec_args(spec), "--final-only", "--on", on, "--off", off, "--out", out])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--full", action="store_true",
                    help="hold out every broadcast day in turn instead of only the newest; for "
                         "choosing thresholds or comparing heads")
    ap.add_argument("--backend", choices=("kaggle", "local"), default="kaggle",
                    help="with --full: where to fit the day folds (default kaggle)")
    ap.add_argument("--test-hours", type=float, default=2.0,
                    help="quick mode: hold out the newest days until they reach this many hours")
    ap.add_argument("--head", default=DEFAULT_HEAD, metavar='"head=... k=v"',
                    help=f"head configuration (default the shipped one: {DEFAULT_HEAD})")
    ap.add_argument("--name", default="mlp16",
                    help="names the predictions file, head file and history rows")
    ap.add_argument("--hours", default="07-11", metavar="HH-HH",
                    help="the listening window to report separately ('' for none)")
    ap.add_argument("--no-pull", action="store_true", help="do not touch the phone")
    ap.add_argument("--no-train", action="store_true",
                    help="report on the predictions from the last run, without retraining")
    ap.add_argument("--final", action="store_true",
                    help="with --full: also fit the head on all the data (quick mode always does)")
    ap.add_argument("--ship", action="store_true",
                    help="install the new head on the phone at the end, keeping its thresholds")
    args = ap.parse_args()
    if args.ship and args.full and not args.final:
        ap.error("--full --ship needs --final too: the day folds alone produce no head to ship")
    if args.ship and args.no_train:
        ap.error("--ship needs a fresh head; drop --no-train, or use trainer/ship.py --head")

    shipped = json.loads(HEAD_WEIGHTS.read_text(encoding="utf-8"))
    on, off = float(shipped["on"]), float(shipped["off"])

    if not args.no_pull:
        step("1. pull from the phone")
        pull()
    step("2. what is labelled")
    inventory()
    mode = "full" if args.full else "quick"
    oof = predictions_path(args.name, args.full)
    head = HEADS / f"head_{args.name}.json"
    if not args.no_train:
        if args.full:
            step(f"3. train (full: every broadcast day held out in turn, {args.backend})")
            oof = train_full(args.name, args.head, args.backend, on, off)
        else:
            step(f"3. train (quick: newest {args.test_hours:g} h held out, then everything)")
            oof = train_quick(args.name, args.head, on, off, args.test_hours)
    elif not oof.exists():
        raise SystemExit(f"--no-train, but there are no predictions at {oof} yet")
    step("4. report")
    report(oof, args.name, args.head, on, off, args.hours, mode)
    if args.full and args.final:
        step("4b. fit the head on everything")
        head = final(args.name, args.head, on, off)
    if args.ship:
        step("5. install on the phone")
        run([sys.executable, TRAINER / "ship.py", "--head", head])
    elif not args.no_train and head.exists() and (not args.full or args.final):
        print(f"\nThe new head is {head.relative_to(ROOT)}. To put it on the phone:\n"
              f"  python trainer/ship.py --head {head.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
