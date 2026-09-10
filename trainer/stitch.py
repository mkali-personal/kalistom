#!/usr/bin/env python3
"""
Joins the desktop recorder's many short segments back into contiguous audio for labelling.

ffmpeg keeps deciding the live playlist has ended, so tools/record_stream.sh restarts it - one
night produced 149 segments instead of the 16 it asked for. 149 files is a miserable thing to
label, and speech recognition wants long context, so they have to be put back together.

WHAT A RESTART COSTS. Measured over that night's 148 restarts: consecutive segments share no
audio whatsoever - not one byte matches - and each restart loses roughly a second, the 5th to 95th
percentile of the estimated gap running from -1.8 s to +2.5 s with nothing above 5 s. So the
segments butt together almost exactly. There is no duplicated audio to remove, but there is a
small hole at every join, and a label drawn across one would claim audio that was never recorded.

This therefore does three things. It checks for verbatim duplication and trims it if a future run
ever produces any. It butt-joins the rest, recording every join in the manifest with its estimated
gap. And it writes an Audacity label track marking each join, so the joins are visible while
labelling rather than being invisible seams. Where a gap is genuinely large the run is closed and
a new file begins, because contiguity that cannot be shown should not be implied.

Output goes to captures/stitched:
    run_<timestamp>.wav          contiguous audio
    run_<timestamp>.json         every source segment, its offset in the output, its gap
    run_<timestamp>.joins.txt    Audacity label track marking the joins

Usage:
    python trainer/stitch.py --dry-run                 # report joins, write nothing
    python trainer/stitch.py                           # captures/desktop -> captures/stitched
    python trainer/stitch.py --in DIR --out DIR --max-minutes 30
"""
from __future__ import annotations

import argparse
import json
import struct
import wave
from pathlib import Path

import numpy as np

SR = 16000
PROBE_SEC = 0.25       # verbatim probe taken from the head of the next segment
LOOKBACK_SEC = 120.0   # how far back in the previous segment to look for it
GAP_SPLIT_SEC = 5.0    # a hole this big is a real discontinuity, not a seam
KEEP_TAIL_SEC = 130.0  # audio held in memory for the overlap check


def read_wav(path: Path) -> np.ndarray:
    """Reads 16 kHz mono PCM. Refuses a file whose header was never finalised rather than
    guessing at it - trainer/repair_wav.py exists for that and keeps a backup."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(4)
        riff = struct.unpack("<I", fh.read(4))[0]
    if riff != size - 8:
        raise ValueError(f"header not finalised (claims {riff:,}, file holds {size - 8:,}); "
                         "run trainer/repair_wav.py on it first")
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"expected {SR} Hz mono 16-bit, got {w.getframerate()} Hz "
                             f"{w.getnchannels()} ch {w.getsampwidth() * 8}-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def verbatim_overlap(tail: np.ndarray, head: np.ndarray) -> int:
    """Samples at the start of `head` that repeat the end of `tail`, or 0.

    Both segments come off the same HLS stream through the same decoder, so a genuine overlap is
    byte-identical. Searching for that exactly is both faster and far more trustworthy than
    correlating: a correlation peak can be produced by ordinary similarity in speech, whereas an
    exact byte run of a quarter of a second cannot happen by chance.
    """
    n = int(PROBE_SEC * SR)
    if len(head) < n or len(tail) < n:
        return 0
    tb = tail[-int(LOOKBACK_SEC * SR):].tobytes()
    j = tb.find(head[:n].tobytes())
    if j < 0:
        return 0
    return (len(tb) - j) // 2


class Run:
    """One contiguous stretch of audio, written incrementally so memory stays bounded."""

    def __init__(self, out_dir: Path, stem: str, continues: bool, dry: bool):
        self.path = out_dir / f"{stem}.wav"
        self.dry = dry
        self.samples = 0
        self.parts: list[dict] = []
        self.continues = continues
        self.tail = np.zeros(0, dtype="<i2")
        if not dry:
            self.w = wave.open(str(self.path), "wb")
            self.w.setnchannels(1)
            self.w.setsampwidth(2)
            self.w.setframerate(SR)

    def append(self, name: str, x: np.ndarray, trimmed: int, gap: float | None):
        self.parts.append({"file": name, "out_offset_s": round(self.samples / SR, 3),
                           "seconds": round(len(x) / SR, 3),
                           "trimmed_s": round(trimmed / SR, 3),
                           "gap_before_s": None if gap is None else round(gap, 2)})
        if not self.dry:
            self.w.writeframes(x.tobytes())
        self.samples += len(x)
        keep = int(KEEP_TAIL_SEC * SR)
        self.tail = x[-keep:] if len(x) >= keep else np.concatenate([self.tail, x])[-keep:]

    def close(self):
        if self.dry:
            return
        self.w.close()
        self.path.with_suffix(".json").write_text(json.dumps({
            "source": "desktop stream, stitched",
            "sample_rate": SR,
            "seconds": round(self.samples / SR, 3),
            "continues_previous": self.continues,
            "parts": self.parts,
        }, indent=2), encoding="utf-8")
        # An Audacity label track for the joins. Point labels, so they show as a line you can see
        # while marking ad boundaries - a label that straddles one is not trustworthy.
        lines = [f"{p['out_offset_s']:.3f}\t{p['out_offset_s']:.3f}\t"
                 f"join {p['gap_before_s']:+.1f}s" for p in self.parts[1:]]
        self.path.with_suffix(".joins.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(root / "captures" / "desktop"))
    ap.add_argument("--out", dest="dst", default=str(root / "captures" / "stitched"))
    ap.add_argument("--max-minutes", type=float, default=30.0,
                    help="start a new output file past this length (0 = never split)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="one line per segment")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(src.glob("desk_*.wav"))
    if not files:
        print(f"no desk_*.wav in {src}")
        return 1
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    cap = int(args.max_minutes * 60 * SR) if args.max_minutes > 0 else None
    print(f"{len(files)} segment(s) from {src}\n")
    if args.verbose:
        print(f"{'segment':30s} {'kept':>8s} {'gap':>7s} {'trim':>7s}  join")
        print("-" * 72)

    run: Run | None = None
    runs: list[Run] = []
    in_total = trimmed_total = skipped = 0
    gaps: list[float] = []
    prev_exit = prev_name = None
    continues = False

    for f in files:
        try:
            x = read_wav(f)
        except ValueError as e:
            print(f"SKIPPED {f.name}: {e}")
            skipped += 1
            continue
        in_total += len(x)
        exit_wall = f.stat().st_mtime
        dur = len(x) / SR

        gap: float | None = None
        trimmed = 0
        note = "first in run"
        if run is not None:
            trimmed = verbatim_overlap(run.tail, x)
            # Both invocations sit at the live edge by the time they exit, so the hole between
            # them is what is left after the second one's own audio is accounted for.
            gap = (exit_wall - dur) - prev_exit + trimmed / SR
            gaps.append(gap)
            if gap > GAP_SPLIT_SEC:
                note = f"BREAK: {gap:.0f}s missing - new run"
                run.close()
                run = None
                continues = False
            else:
                note = f"trimmed {trimmed / SR:.2f}s duplicate" if trimmed else "butt-joined"
                x = x[trimmed:]
                trimmed_total += trimmed
                if cap and run.samples >= cap:
                    run.close()
                    run = None
                    continues = True
                    note += ", split for length"

        if run is None:
            run = Run(dst, f"run_{f.name[5:20]}", continues, args.dry_run)
            runs.append(run)

        run.append(f.name, x, trimmed, gap if run.parts else None)
        if args.verbose:
            g = "-" if gap is None else f"{gap:+.2f}"
            print(f"{f.name:30s} {len(x)/SR:7.1f}s {g:>7s} {trimmed/SR:6.2f}s  {note}")
        prev_exit, prev_name = exit_wall, f.name

    if run is not None:
        run.close()

    out_total = sum(r.samples for r in runs)
    joins = sum(len(r.parts) - 1 for r in runs)
    breaks = sum(1 for r in runs[1:] if not r.continues)
    g = np.array(gaps) if gaps else np.zeros(1)
    print(f"input       {in_total/SR/3600:6.2f} h across {len(files) - skipped} segment(s)")
    print(f"duplicate   {trimmed_total/SR:6.2f} s removed")
    print(f"output      {out_total/SR/3600:6.2f} h across {len(runs)} file(s)"
          + (" (dry run - nothing written)" if args.dry_run else f" in {dst}"))
    if skipped:
        print(f"skipped     {skipped} file(s) - see above")
    print(f"joins       {joins} butt-joined, {breaks} real break(s) over {GAP_SPLIT_SEC:.0f}s")
    print(f"gap at join median {np.median(g):+.2f}s, p95 {np.percentile(g, 95):+.2f}s, "
          f"max {g.max():+.2f}s, total {g[g > 0].sum():.0f}s lost")
    if not args.dry_run:
        print(f"\nEach run has a .joins.txt marking the joins - import it in Audacity "
              f"(File > Import > Labels)\nso you can see them while marking ad boundaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
