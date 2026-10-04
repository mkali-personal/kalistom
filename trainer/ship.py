#!/usr/bin/env python3
"""
Put a head and its thresholds on the phone: edit the app's head_weights.json, rebuild, install.

    python trainer/ship.py --on 0.99                 # mute threshold 0.999 -> 0.99
    python trainer/ship.py --off 0.3                 # release threshold
    python trainer/ship.py --head captures/heads/head_quick_mlp16.json
                                                     # new weights; the phone's thresholds kept
    python trainer/ship.py --restore captures/heads/shipped/head_weights_20261005_0900.json
                                                     # put back an earlier file exactly as it was
    python trainer/ship.py                           # rebuild and install, changing nothing

Every change first saves the file it replaces under captures/heads/shipped/, so any of them can be
undone with --restore.

The thresholds live in the app, not in the model. The head produces a score between 0 and 1 for
each 0.48 s of audio; the phone mutes when the score reaches `on` and unmutes when it falls below
`off`. Lowering `on` mutes ads sooner and more of them, and also mutes more programme by mistake.
trainer/roc_threshold.py draws that trade for every value of `on`.

Installing replaces the running app, which ends a recording in progress, so this refuses while the
recorder is running unless given --force. The phone must be attached with USB debugging on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "app" / "src" / "main" / "assets" / "head_weights.json"
BACKUPS = ROOT / "captures" / "heads" / "shipped"
APK = ROOT / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
JAVA_HOME = Path(os.environ.get("JAVA_HOME", r"C:\Program Files\Android\Android Studio\jbr"))
# Kept by the head itself, not by the thresholds: replacing the weights must not touch these.
THRESHOLD_KEYS = ("on", "off", "smoothing")


def adb() -> str:
    sys.path.insert(0, str(ROOT / "trainer"))
    from ingest import adb as find_adb
    return find_adb()


def check_head(doc: dict) -> None:
    """Refuses a file the phone could not run. Detector.kt reads exactly these."""
    kind = doc.get("type")
    if kind not in ("mlp", "logistic_regression"):
        raise SystemExit(f"not a head file: type is {kind!r}")
    need = ["context_frames", "input_dim", "on", "off"]
    need += ["w1", "b1", "w2", "b2"] if kind == "mlp" else ["weights", "bias", "mean", "scale"]
    missing = [k for k in need if k not in doc]
    if missing:
        raise SystemExit(f"head file is missing {', '.join(missing)}")
    if doc["input_dim"] != doc["context_frames"] * 1025:
        raise SystemExit(f"input_dim {doc['input_dim']} does not match "
                         f"{doc['context_frames']} frames of 1025")
    if not 0.0 < doc["off"] <= doc["on"] < 1.0:
        raise SystemExit(f"thresholds must satisfy 0 < off <= on < 1, got on={doc['on']} "
                         f"off={doc['off']}")


def recording() -> bool:
    r = subprocess.run([adb(), "shell", "dumpsys", "activity", "services", "com.adsfilter"],
                       capture_output=True, text=True)
    return "RecorderService" in r.stdout


def device_attached() -> bool:
    r = subprocess.run([adb(), "devices"], capture_output=True, text=True)
    return any(line.strip().endswith("device") for line in r.stdout.splitlines()[1:])


def build() -> None:
    launchers = glob.glob(str(Path.home() / ".gradle/wrapper/dists/gradle-8.12-bin/*"
                              "/gradle-8.12/bin/gradle.bat"))
    if not launchers:
        raise SystemExit("gradle 8.12 not found under ~/.gradle - build once from Android Studio")
    env = {**os.environ, "JAVA_HOME": str(JAVA_HOME)}
    print("building the app (about a minute)...", flush=True)
    r = subprocess.run([launchers[0], ":app:assembleDebug", "--no-daemon", "-q"], cwd=ROOT,
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit("build failed")
    print(f"built {APK.relative_to(ROOT)}")


def install(force: bool) -> None:
    if not device_attached():
        raise SystemExit("no phone attached - plug it in with USB debugging on, then run:\n"
                         "  python trainer/ship.py")
    if recording() and not force:
        raise SystemExit("the phone is recording right now, and installing would end that "
                         "session.\nStop the recording on the phone, then run: "
                         "python trainer/ship.py   (or add --force to install anyway)")
    r = subprocess.run([adb(), "install", "-r", str(APK)], capture_output=True, text=True)
    if r.returncode != 0 or "Success" not in r.stdout:
        raise SystemExit(f"install failed:\n{r.stdout}{r.stderr}")
    print("installed on the phone - open the app and start recording as usual")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--on", type=float, help="mute when the score reaches this")
    ap.add_argument("--off", type=float, help="unmute when the score falls below this")
    ap.add_argument("--head", type=Path,
                    help="new weights (from update.py or train.py --out); thresholds stay as "
                         "they are on the phone unless --on/--off are given too")
    ap.add_argument("--restore", type=Path,
                    help="put back an earlier head_weights.json exactly, thresholds included")
    ap.add_argument("--no-install", action="store_true", help="edit and build, but do not install")
    ap.add_argument("--force", action="store_true", help="install even while recording")
    args = ap.parse_args()
    if args.head and args.restore:
        ap.error("--head and --restore are alternatives")

    current = json.loads(ASSET.read_text(encoding="utf-8"))
    if args.restore:
        doc = json.loads(args.restore.read_text(encoding="utf-8"))
    elif args.head:
        doc = json.loads(args.head.read_text(encoding="utf-8"))
        for k in THRESHOLD_KEYS:
            if k in current:
                doc[k] = current[k]
            else:
                doc.pop(k, None)
    else:
        doc = dict(current)
    if args.on is not None:
        doc["on"] = args.on
    if args.off is not None:
        doc["off"] = args.off
    if (doc["on"], doc["off"]) != (current["on"], current["off"]) and not args.restore:
        # The old measurement describes the old thresholds; leaving it would misreport this one.
        doc["operating_point"] = {
            "rule": f"hysteresis, no averaging: mute when the score reaches {doc['on']:g}, "
                    f"release below {doc['off']:g}",
            "set_by": f"trainer/ship.py on {time.strftime('%Y-%m-%d')}, not measured here - "
                      f"see trainer/update.py --full and trainer/roc_threshold.py"}
    check_head(doc)

    if doc != current:
        BACKUPS.mkdir(parents=True, exist_ok=True)
        backup = BACKUPS / f"head_weights_{time.strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(ASSET, backup)
        ASSET.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        what = []
        if doc.get("w1", doc.get("weights")) != current.get("w1", current.get("weights")):
            what.append(f"new {doc['type']} weights")
        for k in ("on", "off"):
            if doc[k] != current[k]:
                what.append(f"{k} {current[k]:g} -> {doc[k]:g}")
        print(f"head_weights.json: {', '.join(what) or 'updated'}")
        print(f"the previous file is saved as {backup.relative_to(ROOT)}; to undo:\n"
              f"  python trainer/ship.py --restore {backup.relative_to(ROOT).as_posix()}")
    else:
        print(f"head_weights.json unchanged (on={doc['on']:g}, off={doc['off']:g})")

    build()
    if not args.no_install:
        install(args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
