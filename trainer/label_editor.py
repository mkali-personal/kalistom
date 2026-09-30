#!/usr/bin/env python3
"""
A label editor for the recordings, in the browser, with nothing to install.

Audacity does the job but asks for five steps per file: open the audio, switch to spectrogram,
import the joins, import the draft, export under the right name. This does those implicitly - pick
a recording from the list and its spectrogram, its join markers and its best existing labels are
already there - so correcting a machine draft is only the correcting.

It is a small local web server (standard library plus numpy) and one page, label_editor.html.

    python trainer/label_editor.py            # opens http://localhost:8765
    python trainer/label_editor.py --port 9000 --no-browser

WHERE LABELS GO. It reads the best label file a recording has, in train.py's order - hand labels,
then the machine drafts - but it only ever WRITES <stem>.truth.txt, and keeps the previous one as
.truth.txt.bak. A draft is never overwritten, so a hand correction can always be compared with the
draft it started from (compare_labels.py), and train.py already prefers .truth.txt when both exist.

THE SPECTROGRAM is computed once per file and cached under captures/.labelcache: a 1024-point FFT
every 0.05 s, folded onto a mel-spaced frequency axis and stored as uint8, about 20 MB for two
hours. Each request cuts out the visible stretch and max-pools it to the width of the screen, so a
two-hour overview and a five-second close-up cost the same to draw. Max rather than mean because a
short loud event - a jingle's attack, a word - should stay visible when zoomed out.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import wave
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "label_editor.html"
GROUPS = {"stitched": ROOT / "captures" / "stitched", "sessions": ROOT / "captures" / "sessions"}
CACHE = ROOT / "captures" / ".labelcache"

# Same precedence as train.py's LABEL_ORDER, plus the parse step's default output. Kept literal
# rather than imported so the editor starts without pulling in the training stack.
LABEL_ORDER = (".truth.txt", ".gemini.txt", ".claude.txt", ".draft.txt")

SCORE_GLOBS = ("*.npz", "heads/oof_*.npz")      # under captures/: train.py --save-oof output
HEAD_WEIGHTS = ROOT / "app" / "src" / "main" / "assets" / "head_weights.json"
EMBEDDING_BYTES = 1024 * 2                       # one float16 embedding row in a .f16

SPEC_HOP = 0.05          # seconds per spectrogram column
N_FFT = 1024
N_MEL = 128              # before merging the low-frequency bins the FFT cannot resolve

_spec_lock = threading.Lock()
_spec_mem: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------- files

def resolve(name: str) -> tuple[Path, Path]:
    """'stitched/run_X' -> (directory, stem path). Anything else is refused: the name arrives from
    the network, and must not be able to walk out of the two recording directories."""
    m = re.fullmatch(r"(stitched|sessions)/([A-Za-z0-9_.-]+)", name or "")
    if not m or ".." in m.group(2):
        raise ValueError(f"bad name {name!r}")
    d = GROUPS[m.group(1)]
    return d, d / m.group(2)


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, OSError):
        return 0.0


def label_source(stem: Path) -> tuple[str, Path | None]:
    for suffix in LABEL_ORDER:
        p = stem.with_name(stem.name + suffix)
        if p.exists():
            return suffix.split(".")[1], p
    return "none", None


def read_label_file(path: Path) -> tuple[list[dict], list[float]]:
    """Audacity format, start<TAB>end<TAB>text. Point labels are markers, returned separately."""
    spans, points = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) < 2:
            continue
        try:
            a, b = float(p[0]), float(p[1])
        except ValueError:
            continue
        if b > a:
            spans.append({"s": a, "e": b, "text": p[2] if len(p) > 2 else "ad"})
        elif b == a:
            points.append(a)
    return sorted(spans, key=lambda x: x["s"]), sorted(points)


def list_files() -> list[dict]:
    out = []
    for group, d in GROUPS.items():
        for wav in sorted(d.glob("*.wav")):
            stem = wav.with_suffix("")
            out.append({"name": f"{group}/{stem.name}", "group": group, "stem": stem.name,
                        "seconds": round(wav_seconds(wav), 1),
                        "source": label_source(stem)[0]})
    return out


def load_labels(name: str) -> dict:
    d, stem = resolve(name)
    source, path = label_source(stem)
    spans = read_label_file(path)[0] if path else []
    joins_path = stem.with_name(stem.name + ".joins.txt")
    joins = read_label_file(joins_path)[1] if joins_path.exists() else []
    return {"name": name, "source": source, "labels": spans, "joins": joins,
            "seconds": wav_seconds(stem.with_suffix(".wav"))}


def save_labels(name: str, labels: list[dict]) -> tuple[str, int]:
    d, stem = resolve(name)
    rows = []
    for lab in labels:
        a, b = float(lab["s"]), float(lab["e"])
        if b <= a:
            continue
        text = str(lab.get("text") or "ad").replace("\t", " ").replace("\n", " ")
        rows.append((a, b, text))
    rows.sort()
    out = stem.with_name(stem.name + ".truth.txt")
    if out.exists():
        out.with_name(out.name + ".bak").write_bytes(out.read_bytes())
    out.write_text("".join(f"{a:.3f}\t{b:.3f}\t{t}\n" for a, b, t in rows), encoding="utf-8")
    return out.name, len(rows)


# ---------------------------------------------------------------- classifier scores

def f16_frames(stem: str) -> int | None:
    for d in GROUPS.values():
        p = d / f"{stem}.f16"
        if p.exists():
            return p.stat().st_size // EMBEDDING_BYTES
    return None


def recording_slices(doc) -> dict[str, slice] | None:
    """Which rows of a --save-oof file belong to which recording.

    Newer files say so outright (rec_names, rec_bounds). Older ones store only a fold per row and
    a name per fold, where a fold that held the same broadcast twice is named 'a + b'. The rows
    are still recoverable exactly: train.py writes recordings in order, each has as many rows as
    its .f16 has frames, and a merged fold lists its members in that same order - so each fold's
    rows are cut up by its members' frame counts. If the counts do not add up, the file was made
    from embeddings that have since changed, and None is returned rather than a guess.
    """
    if "rec_names" in doc.files:
        return {str(n): slice(int(a), int(b))
                for n, (a, b) in zip(doc["rec_names"], doc["rec_bounds"])}
    groups, names = doc["groups"], doc["names"]
    out: dict[str, slice] = {}
    for g, name in enumerate(names):
        rows = np.flatnonzero(groups == g)
        at = 0
        for member in str(name).split(" + "):
            n = f16_frames(member)
            if n is None:
                return None
            part = rows[at:at + n]
            if len(part) != n or (n and part[-1] - part[0] != n - 1):
                return None
            out[member] = slice(int(part[0]), int(part[0]) + n) if n else slice(0, 0)
            at += n
        if at != len(rows):
            return None
    return out


_score_lock = threading.Lock()
_score_index: dict = {"stamp": None, "by_rec": {}}


def score_index() -> dict[str, list[tuple[Path, slice]]]:
    """recording stem -> [(oof file, its rows)], rebuilt whenever a file appears or changes."""
    files = sorted(p for g in SCORE_GLOBS for p in (ROOT / "captures").glob(g))
    stamp = tuple((str(p), p.stat().st_mtime) for p in files)
    with _score_lock:
        if _score_index["stamp"] != stamp:
            by_rec: dict[str, list[tuple[Path, slice]]] = {}
            for p in files:
                try:
                    with np.load(p, allow_pickle=True) as doc:
                        sl = recording_slices(doc)
                except (OSError, ValueError, KeyError):
                    sl = None
                if sl is None:
                    print(f"  (skipping {p.name}: its rows no longer match the recordings)")
                    continue
                for rec, s in sl.items():
                    by_rec.setdefault(rec, []).append((p, s))
            _score_index.update(stamp=stamp, by_rec=by_rec)
        return _score_index["by_rec"]


def live_scores(stem: Path) -> np.ndarray | None:
    """The phone's own raw score per frame, for sessions recorded since the head shipped."""
    meta = stem.with_suffix(".jsonl")
    if not meta.exists():
        return None
    vals = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if '"ad"' in line and '"rms"' in line:
            vals.append(float(json.loads(line)["ad"]))
    return np.array(vals, np.float32) if vals else None


def score_sources(name: str) -> dict:
    d, stem = resolve(name)
    sources = [{"id": f"oof:{p.relative_to(ROOT / 'captures').as_posix()}",
                "label": f"{p.stem} (held out)"}
               for p, _ in score_index().get(stem.name, [])]
    meta = stem.with_suffix(".jsonl")
    if meta.exists() and '"ad"' in meta.read_text(encoding="utf-8")[:4000]:
        sources.append({"id": "live", "label": "phone (live)"})
    defaults = {"frames": 11, "on": 0.99, "off": 0.7}
    try:
        w = json.loads(HEAD_WEIGHTS.read_text(encoding="utf-8"))
        defaults = {"frames": int((w.get("smoothing") or {}).get("frames", 1)),
                    "on": float(w["on"]), "off": float(w["off"])}
    except (OSError, ValueError, KeyError):
        pass
    return {"sources": sources, "defaults": defaults}


def score_values(name: str, src: str) -> np.ndarray:
    d, stem = resolve(name)
    if src == "live":
        p = live_scores(stem)
        if p is None:
            raise ValueError("no live scores in this session")
        return p
    if not src.startswith("oof:"):
        raise ValueError(f"unknown source {src!r}")
    for path, s in score_index().get(stem.name, []):
        if src == f"oof:{path.relative_to(ROOT / 'captures').as_posix()}":
            with np.load(path, allow_pickle=True) as doc:
                return doc["p"][s].astype(np.float32)
    raise ValueError(f"{src} has no rows for {stem.name}")


# ---------------------------------------------------------------- spectrogram

def mel_edges(sr: int) -> np.ndarray:
    """FFT-bin boundaries of mel-spaced bands. At the bottom a mel band is narrower than one FFT
    bin, so duplicates are merged: the axis ends up with a little under N_MEL rows."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)                       # noqa: E731
    inv = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)                      # noqa: E731
    hz = inv(np.linspace(mel(40.0), mel(sr / 2), N_MEL + 1))
    edges = np.unique(np.round(hz / sr * N_FFT).astype(int))
    return edges[edges <= N_FFT // 2 + 1]


def compute_spectrogram(wav: Path) -> np.ndarray:
    """(columns, bands) uint8, low frequencies first."""
    with wave.open(str(wav), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{wav.name}: expected 16-bit mono")
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    hop = int(round(SPEC_HOP * sr))
    n = 0 if len(x) < N_FFT else (len(x) - N_FFT) // hop + 1
    edges = mel_edges(sr)
    win = np.hanning(N_FFT).astype(np.float32)
    db = np.empty((n, len(edges) - 1), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::hop]
    for k in range(0, n, 4096):
        chunk = frames[k:k + 4096].astype(np.float32) * win
        power = np.abs(np.fft.rfft(chunk, axis=1)) ** 2
        bands = np.add.reduceat(power, edges[:-1], axis=1) / np.diff(edges)
        db[k:k + 4096] = 10.0 * np.log10(bands + 1e-3)
    if n == 0:
        return np.zeros((0, len(edges) - 1), np.uint8)
    # A fixed range per file, from its own distribution: broadcast loudness varies between
    # recordings, and a global range would leave quiet ones nearly black.
    lo, hi = np.percentile(db, 5), np.percentile(db, 99.7)
    return np.clip((db - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def spectrogram(name: str) -> np.ndarray:
    d, stem = resolve(name)
    wav = stem.with_suffix(".wav")
    cached = CACHE / f"{name.replace('/', '__')}.npy"
    with _spec_lock:
        if name in _spec_mem:
            return _spec_mem[name]
        if cached.exists() and cached.stat().st_mtime >= wav.stat().st_mtime:
            S = np.load(cached)
        else:
            S = compute_spectrogram(wav)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(cached, S)
        _spec_mem.clear()                       # one file at a time is all the page ever shows
        _spec_mem[name] = S
        return S


def spectrogram_slice(name: str, t0: float, t1: float, cols: int) -> tuple[bytes, int, int]:
    S = spectrogram(name)
    a = max(int(np.floor(t0 / SPEC_HOP)), 0)
    b = min(int(np.ceil(t1 / SPEC_HOP)), len(S))
    part = S[a:b] if b > a else np.zeros((1, S.shape[1]), np.uint8)
    if len(part) > cols > 0:
        starts = (np.arange(cols) * len(part)) // cols
        part = np.maximum.reduceat(part, starts, axis=0)
    return np.ascontiguousarray(part).tobytes(), len(part), part.shape[1]


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):         # the page polls; the console should stay quiet
        pass

    def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/files":
                self._json(list_files())
            elif u.path == "/api/labels":
                self._json(load_labels(q["name"]))
            elif u.path == "/api/spec":
                body, cols, bands = spectrogram_slice(q["name"], float(q["t0"]), float(q["t1"]),
                                                      int(q.get("cols", 1000)))
                self._send(body, "application/octet-stream",
                           headers={"X-Cols": str(cols), "X-Bands": str(bands),
                                    "Access-Control-Expose-Headers": "X-Cols, X-Bands"})
            elif u.path == "/api/scores":
                self._json(score_sources(q["name"]))
            elif u.path == "/api/score":
                self._send(score_values(q["name"], q["src"]).tobytes(),
                           "application/octet-stream")
            elif u.path == "/audio":
                self._audio(resolve(q["name"])[1].with_suffix(".wav"))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, OSError) as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except (ConnectionError, BrokenPipeError):
            pass                                # the browser abandoned a request; nothing to do

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path != "/api/labels":
                raise KeyError(u.path)
            n = int(self.headers.get("Content-Length", 0))
            doc = json.loads(self.rfile.read(n))
            written, count = save_labels(q["name"], doc["labels"])
            self._json({"saved": written, "count": count})
        except (KeyError, ValueError, OSError) as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    def _audio(self, path: Path):
        """The WAV with byte-range support. Without it the browser cannot seek in a two-hour file
        - http.server's stock handler ignores Range and sends the whole thing every time."""
        size = path.stat().st_size
        rng = self.headers.get("Range")
        a, b = 0, size - 1
        status = HTTPStatus.OK
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    a = int(m.group(1))
                    b = int(m.group(2)) if m.group(2) else size - 1
                else:                              # suffix form: the last N bytes
                    a = max(size - int(m.group(2)), 0)
                b = min(b, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
        if a > b:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}/"
    print(f"label editor on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
