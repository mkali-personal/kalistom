#!/usr/bin/env python3
"""
Session ingest + integrity check.

The dataset's whole value rests on one invariant:

    frame k of the .f16 covers samples [k*HOP, k*HOP + WINDOW) of the .wav

If that drifts, every label drawn in Audacity maps to the wrong embeddings and the corruption is
invisible - the model just quietly fails to learn. So this refuses to pass a session it cannot
verify, and distinguishes a fixed offset (recoverable) from a growing one (dropped frames).

Usage:
    python trainer/ingest.py                     # verify ./captures/sessions
    python trainer/ingest.py --dir some/dir
    python trainer/ingest.py --pull              # adb pull from the phone first, then delete there
    python trainer/ingest.py --pull --keep-on-phone   # pull without deleting
    python trainer/ingest.py --verify-alignment  # prove embeddings line up with the audio
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Must match Yamnet.kt.
SAMPLE_RATE = 16000
WINDOW = 15600
HOP = 7680
EMBEDDING_DIM = 1024

REMOTE_DIR = "/sdcard/Android/data/com.adsfilter/files/sessions"
DEFAULT_LOCAL = Path("captures/sessions")


@dataclass
class Session:
    name: str
    wav: Path
    emb: Path
    meta: Path
    samples: int = 0
    frames_emb: int = 0
    frames_meta: int = 0
    duration_s: float = 0.0
    header: dict = field(default_factory=dict)
    markers: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def adb() -> str:
    env = os.environ.get("ADB")
    if env:
        return env
    local = os.environ.get("LOCALAPPDATA", "")
    cand = Path(local) / "Android/Sdk/platform-tools/adb.exe"
    return str(cand) if cand.exists() else "adb"


def remote_hashes() -> dict[str, str]:
    """md5 of every file in the phone's session directory, keyed by file name."""
    r = subprocess.run([adb(), "shell", f"cd {REMOTE_DIR} 2>/dev/null && md5sum * 2>/dev/null"],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0]
    return out


def local_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_closed(meta: Path) -> bool:
    """True once the writer has appended its "end" record, which it does last, on close."""
    try:
        lines = meta.read_text(encoding="utf-8").strip().splitlines()
        return bool(lines) and json.loads(lines[-1]).get("type") == "end"
    except (OSError, ValueError):
        return False


def pull(dest: Path, keep_on_phone: bool = False) -> None:
    """Copies the phone's sessions here, then deletes from the phone every one that arrived intact.

    The phone is not an archive: live inference never reads old sessions, so a session that is
    safely on this machine is only taking space there. A session is deleted from the phone only
    when two things hold. First, every one of its files must hash the same here as on the phone,
    so nothing is removed on the strength of a truncated or interrupted copy. Second, it must be
    finished. Only one session is ever open at a time and it is always the newest, so every older
    one is finished, and the newest is finished once its .jsonl ends with the "end" record. The
    open session is still copied, so the partial copy is usable, and it is deleted on a later pull.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Hash first and pull second: a file still being written then fails the comparison rather than
    # passing it, because the copy holds more bytes than were hashed.
    before = remote_hashes()
    if not before:
        print(f"nothing on the phone in {REMOTE_DIR}")
        return
    print(f"pulling {REMOTE_DIR} -> {dest}")
    # adb.exe is a Windows binary: give it a native destination path.
    win_dest = str(dest.parent.resolve())
    r = subprocess.run([adb(), "pull", REMOTE_DIR, win_dest],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("adb pull failed")
    if keep_on_phone:
        return

    stems = sorted({Path(f).stem for f in before})
    newest = stems[-1]
    removed = 0
    for stem in stems:
        files = sorted(f for f in before if Path(f).stem == stem)
        if stem == newest and not is_closed(dest / f"{stem}.jsonl"):
            print(f"  kept on phone: {stem} is still being recorded")
            continue
        bad = [f for f in files
               if not (dest / f).exists() or local_md5(dest / f) != before[f]]
        if bad:
            print(f"  kept on phone: {stem}, because {', '.join(bad)} did not copy intact")
            continue
        rm = subprocess.run([adb(), "shell", "rm", "-f"] + [f"{REMOTE_DIR}/{f}" for f in files],
                            capture_output=True, text=True)
        if rm.returncode != 0:
            print(f"  could not delete {stem} from the phone: {rm.stderr.strip()}")
            continue
        removed += 1
    print(f"deleted {removed} pulled session(s) from the phone")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit mono, got {w.getnchannels()}ch "
                             f"{w.getsampwidth()*8}-bit")
        sr = w.getframerate()
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def expected_frames(samples: int) -> int:
    return 0 if samples < WINDOW else (samples - WINDOW) // HOP + 1


def verify(name: str, d: Path) -> Session:
    s = Session(name=name, wav=d / f"{name}.wav", emb=d / f"{name}.f16",
                meta=d / f"{name}.jsonl")

    for f in (s.wav, s.emb, s.meta):
        if not f.exists():
            s.problems.append(f"missing {f.name}")
    if s.problems:
        return s

    # ---- audio
    try:
        x, sr = read_wav(s.wav)
    except Exception as e:                                   # noqa: BLE001
        s.problems.append(f"unreadable wav: {e}")
        return s
    s.samples = len(x)
    s.duration_s = s.samples / sr
    if sr != SAMPLE_RATE:
        s.problems.append(f"sample rate {sr} != {SAMPLE_RATE}")

    # ---- embeddings
    emb_bytes = s.emb.stat().st_size
    if emb_bytes % (EMBEDDING_DIM * 2):
        s.problems.append(f"f16 size {emb_bytes} is not a whole number of "
                          f"{EMBEDDING_DIM}-d float16 frames")
    s.frames_emb = emb_bytes // (EMBEDDING_DIM * 2)

    # ---- metadata
    frame_rows, idx_seen = 0, []
    with s.meta.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                s.problems.append(f"{s.meta.name}:{ln} bad json: {e}")
                continue
            t = o.get("type")
            if t == "session":
                s.header = o
            elif t == "marker":
                s.markers.append(o)
            elif t == "end":
                pass
            elif "f" in o:
                frame_rows += 1
                idx_seen.append(o["f"])
    s.frames_meta = frame_rows

    # ---- the invariant
    exp = expected_frames(s.samples)
    if s.frames_emb != exp:
        s.problems.append(
            f"ALIGNMENT: {s.frames_emb} embedding frames but audio implies {exp} "
            f"({s.samples} samples). drift={s.frames_emb - exp} frames "
            f"({(s.frames_emb - exp) * HOP / SAMPLE_RATE:+.2f}s)")
    if s.frames_meta != s.frames_emb:
        s.problems.append(f"metadata has {s.frames_meta} frame rows but f16 has {s.frames_emb}")
    if idx_seen:
        arr = np.array(idx_seen)
        if arr[0] != 0 or not np.array_equal(arr, np.arange(len(arr))):
            gaps = np.where(np.diff(arr) != 1)[0]
            s.problems.append(f"frame indices not contiguous from 0 "
                              f"(first={arr[0]}, {len(gaps)} discontinuities)")

    for key, want in (("window", WINDOW), ("hop", HOP),
                      ("sample_rate", SAMPLE_RATE), ("embedding_dim", EMBEDDING_DIM)):
        if s.header and s.header.get(key) != want:
            s.problems.append(f"header {key}={s.header.get(key)} != {want}")

    return s


def load_embeddings(s: Session) -> np.ndarray:
    raw = np.fromfile(s.emb, dtype="<f2").astype(np.float32)
    return raw.reshape(-1, EMBEDDING_DIM)


def alignment_test(s: Session) -> bool:
    """
    Exact alignment proof.

    The device writes, for every frame, the RMS of that frame's own analysis window. Here we
    recompute the same quantity straight from the WAV using the documented mapping
    (frame k -> samples [k*HOP, k*HOP+WINDOW)) and compare. If the mapping is right the two
    agree to within float16/rounding noise; if the recorder ever dropped or duplicated a frame,
    the error explodes at that point and stays large.

    This is stronger than a correlation: it pins the absolute offset, not just the shape.
    """
    x, _ = read_wav(s.wav)
    n = expected_frames(len(x))

    rec = []
    with s.meta.open(encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if "f" in o and "rms" in o:
                rec.append(o["rms"])
    rec = np.asarray(rec, dtype=np.float64)
    n = min(n, len(rec))
    if n < 4:
        print("   too few frames to verify")
        return False

    starts = np.arange(n) * HOP
    calc = np.array([
        20 * np.log10(max(float(np.sqrt(np.mean(x[a:a + WINDOW] ** 2))), 1e-6))
        for a in starts
    ])
    calc = np.maximum(calc, -120.0)
    err = np.abs(calc - rec[:n])

    # Device rms is written with 3 decimals and the audio round-trips through int16, so a few
    # hundredths of a dB is expected; anything beyond ~0.5 dB means a real mismatch.
    worst = float(err.max())
    median = float(np.median(err))
    bad = int((err > 0.5).sum())
    print(f"   rms match over {n} frames: median {median:.4f} dB, worst {worst:.4f} dB, "
          f"{bad} frame(s) off by >0.5 dB")

    if bad == 0:
        print("   ALIGNMENT VERIFIED - embeddings line up with the audio")
        return True

    first = int(np.argmax(err > 0.5))
    print(f"   !! first mismatch at frame {first} (t={first * HOP / SAMPLE_RATE:.2f}s)")
    tail = err[first:]
    if float(np.median(tail)) > 0.5:
        print("   !! error persists after that point -> a frame was dropped or duplicated; "
              "labels after this timestamp cannot be trusted")
    else:
        print("   !! isolated mismatch -> likely a transient glitch, inspect that frame")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_LOCAL)
    ap.add_argument("--pull", action="store_true",
                    help="adb pull sessions first, deleting from the phone those that copied intact")
    ap.add_argument("--keep-on-phone", action="store_true",
                    help="with --pull, leave the sessions on the phone as well")
    ap.add_argument("--verify-alignment", action="store_true", help="recompute frame rms from the wav and compare")
    args = ap.parse_args()

    if args.pull:
        pull(args.dir, keep_on_phone=args.keep_on_phone)

    d = args.dir
    if not d.exists():
        print(f"no session directory: {d}")
        return 1

    names = sorted({p.stem for p in d.glob("*.wav")
                    if (d / f"{p.stem}.f16").exists()})
    if not names:
        print(f"no verifiable sessions in {d}")
        print("(a session is a .wav with a matching .f16 beside it)")
        return 1

    print(f"{len(names)} session(s) in {d}\n")
    total_s, bad = 0.0, 0
    for nm in names:
        s = verify(nm, d)
        total_s += s.duration_s
        flag = "OK  " if s.ok else "FAIL"
        print(f"[{flag}] {s.name}  {s.duration_s:8.1f}s  {s.frames_emb:6d} frames  "
              f"{s.samples:10d} samples  markers={len(s.markers)}")
        for p in s.problems:
            print(f"        - {p}")
        if not s.ok:
            bad += 1
        for m in s.markers:
            print(f"        marker '{m.get('label')}' @ {m.get('t')}s")
        if args.verify_alignment and s.ok:
            alignment_test(s)

    print(f"\ntotal {total_s / 3600:.2f} h across {len(names)} session(s); {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
