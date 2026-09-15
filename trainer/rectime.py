#!/usr/bin/env python3
"""
When a recording was made, and therefore which recordings to keep.

Time of day matters more than it looks here. This station carries roughly ten times as much
advertising at eleven in the morning as at three in the morning, so "drop the night recordings"
is a thing we will want to say repeatedly - when splitting train from test, when deciding what is
worth labelling, and when asking whether a model trained on daytime transfers to evening.

Three naming conventions carry a timestamp, all of them local time:

    desk_20260910_184500     one ffmpeg segment, named when ffmpeg started
    run_20260910_184500      a stitched run, named after its first segment
    sess_20260914_084623     a phone session, named when the session opened

For phone sessions the .jsonl header also carries an explicit `started_utc`, which is preferred
when present because it does not depend on the filename surviving a copy.

A caveat that matters for the desktop recordings: the name is when *ffmpeg* started, and it joins
a live HLS playlist a little behind the live edge, so the broadcast is some tens of seconds older
than the name says. That is far below the resolution of any question about time of day, but it is
not zero, so do not use these timestamps to align two recordings - trainer/skew_test.py exists for
that.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

STAMP = re.compile(r"(?:desk|run|sess)_(\d{8})_(\d{6})")
HOP_S = 0.48


def started_at(path: str | Path) -> datetime | None:
    """Local wall-clock time the *content* was broadcast, or None if nothing says.

    Three sources, in order:

      <stem>.broadcast.json   an explicit override, when the two differ
      <stem>.jsonl            "started_utc", written by the phone when the session opened
      the filename            desk_/run_/sess_ + YYYYMMDD_HHMMSS, local time

    The override exists because *recorded at* and *broadcast at* are not the same thing. A phone
    session captures whatever the app is playing, and the listener may have scrolled back: the
    2026-09-14 session was recorded from 08:46 but plays the broadcast from 08:00. Everything
    that reasons about time here - which hours to train on, whether two recordings are the same
    broadcast - cares about when the audio went out, not when we happened to catch it.
    """
    p = Path(path)
    override = p.with_name(p.stem + ".broadcast.json")
    if override.exists():
        try:
            return datetime.fromisoformat(
                json.loads(override.read_text(encoding="utf-8"))["content_started_local"])
        except Exception:                                       # noqa: BLE001
            pass
    meta = p.with_suffix(".jsonl")
    if meta.exists():
        try:
            head = json.loads(meta.read_text(encoding="utf-8").splitlines()[0])
            if "started_utc" in head:
                return (datetime.fromisoformat(head["started_utc"].replace("Z", "+00:00"))
                        .astimezone())
        except Exception:                                       # noqa: BLE001
            pass                                                # fall through to the filename
    m = STAMP.search(p.name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").astimezone()


def frame_time(start: datetime, frame_index: int) -> datetime:
    """Wall-clock time of one embedding frame, for asking when a particular ad break ran."""
    return start + __import__("datetime").timedelta(seconds=frame_index * HOP_S)


def in_hours(t: datetime, lo: float, hi: float) -> bool:
    """Whether a local time falls in the window [lo, hi) given as decimal hours.

    Windows that wrap midnight are the normal case here - "the night" is 00:30 to 05:30 on one
    side and 23:00 to 06:00 on the other - so 23-6 means the small hours and not the other
    nineteen.
    """
    h = t.hour + t.minute / 60 + t.second / 3600
    return (lo <= h < hi) if lo <= hi else (h >= lo or h < hi)


def parse_window(spec: str) -> tuple[float, float]:
    """"9-17" or "22:30-6" into a pair of decimal hours."""
    def one(s: str) -> float:
        s = s.strip()
        if ":" in s:
            hh, mm = s.split(":", 1)
            return int(hh) + int(mm) / 60
        return float(s)
    lo, hi = spec.split("-", 1)
    return one(lo), one(hi)


def describe(path: str | Path, seconds: float | None = None) -> str:
    """A short human label for a recording: when it started, and when it ended if known."""
    t = started_at(path)
    if t is None:
        return "time unknown"
    if seconds is None:
        return t.strftime("%a %d %b %H:%M")
    end = frame_time(t, int(seconds / HOP_S))
    return f"{t.strftime('%a %d %b %H:%M')}-{end.strftime('%H:%M')}"


if __name__ == "__main__":
    import sys
    import wave
    for a in sys.argv[1:]:
        try:
            with wave.open(a) as w:
                secs = w.getnframes() / w.getframerate()
        except Exception:                                       # noqa: BLE001
            secs = None
        print(f"{Path(a).name:34s} {describe(a, secs)}")
