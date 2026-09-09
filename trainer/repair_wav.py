#!/usr/bin/env python3
"""
Repairs WAV files whose header was never finalised.

ffmpeg writes placeholder sizes into a WAV header when it starts and patches them on a clean exit.
If it is killed instead - the terminal window closed, the machine shut down, the process force
quit - the audio is all on disk but the header still claims the placeholder. Tools then read a
nonsense duration (one observed case reported 37 hours for a 24-second file) or refuse the file
outright.

Nothing is lost; the sizes just have to be recomputed from the file itself. This rewrites them in
place, keeping a .bak unless told not to.

Usage:
    python trainer/repair_wav.py FILE.wav [more.wav ...]
    python trainer/repair_wav.py captures/desktop/*.wav      # safe: healthy files are skipped
    python trainer/repair_wav.py --no-backup FILE.wav
"""
from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path


def find_data_chunk(path: Path) -> tuple[int, int]:
    """Returns (offset of data chunk's payload, declared size). Walks chunks rather than
    assuming a layout: ffmpeg writes a LIST/INFO chunk between fmt and data, so data is not
    at the textbook offset 36."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        pos = 12
        while pos < size:
            h = fh.read(8)
            if len(h) < 8:
                break
            cid, csz = h[:4], struct.unpack("<I", h[4:8])[0]
            pos += 8
            if cid == b"data":
                return pos, csz
            # chunks are word-aligned
            skip = csz + (csz & 1)
            fh.seek(skip, 1)
            pos += skip
    raise ValueError("no data chunk found")


def check(path: Path) -> tuple[bool, str, int, int, int]:
    """Returns (needs_repair, message, data_offset, correct_riff, correct_data)."""
    size = path.stat().st_size
    data_off, declared = find_data_chunk(path)
    good_data = size - data_off
    good_riff = size - 8

    with path.open("rb") as fh:
        fh.seek(4)
        riff = struct.unpack("<I", fh.read(4))[0]

    if declared == good_data and riff == good_riff:
        return False, "header already correct", data_off, good_riff, good_data
    return (True,
            f"header claims data={declared:,}, file holds {good_data:,}",
            data_off, good_riff, good_data)


def repair(path: Path, backup: bool) -> bool:
    try:
        needs, msg, data_off, good_riff, good_data = check(path)
    except ValueError as e:
        print(f"skip  {path.name}: {e}")
        return False
    if not needs:
        print(f"skip  {path.name}: {msg}")
        return False
    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    with path.open("r+b") as fh:
        fh.seek(4)
        fh.write(struct.pack("<I", good_riff))
        fh.seek(data_off - 4)
        fh.write(struct.pack("<I", good_data))
    secs = good_data / 2 / 16000        # 16 kHz mono 16-bit, as our recorders write
    print(f"FIXED {path.name}: {msg}")
    print(f"      {good_data:,} bytes of audio = {secs:.1f}s (16 kHz mono 16-bit)")
    return True


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--no-backup"]
    backup = "--no-backup" not in sys.argv
    if not args:
        print(__doc__)
        return 1
    fixed = 0
    for a in args:
        p = Path(a)
        if not p.exists():
            print(f"skip  {a}: no such file")
            continue
        try:
            fixed += repair(p, backup)
        except Exception as e:                                # noqa: BLE001
            print(f"ERROR {p.name}: {e}")
    print(f"\n{fixed} file(s) repaired")
    if fixed and backup:
        print("originals kept as .bak - delete them once you have checked the audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
