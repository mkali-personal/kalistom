#!/usr/bin/env python3
"""
Transcribes stitched recordings to Hebrew text with word-level timestamps.

This exists to make labelling cheap. A language model asked to listen to audio and report when
an ad ran is being asked two things at once, and it is far better at the first than the second -
its sense of elapsed time drifts badly over a long file. Splitting the job fixes that: speech
recognition says *when* each word was spoken, to within a few tens of milliseconds, and a language
model then only has to say *which* words are advertising. Every timestamp we end up with comes
from the recogniser.

That the text carries the signal at all is not an assumption. In the first five minutes tried, a
sponsorship break announced itself as "התחזית בחסות" and "השעה בחסות" - the forecast, the hour,
brought to you by - and the ads themselves closed with the standard consumer boilerplate
("כפוף לתנאי השימוש", "יש לעיין בעלון לצרכן לפני השימוש"). Those phrases do not occur in news
copy.

MODEL. ivrit-ai publishes Hebrew fine-tunes of Whisper; the large-v3-turbo CTranslate2 build runs
on CPU at about 1.6x realtime on this laptop with int8 weights, so a night's recording takes about
five hours. Batched inference was measured and gives nothing here - CPU inference is compute-bound
rather than latency-bound, so the batching that helps a GPU does not help us.

Outputs, written next to each input:
    <base>.words.json      segments and words with start/end times and confidence
    <base>.transcript.txt  readable timestamped text, and what the labelling step reads

Usage:
    python trainer/transcribe.py                       # all of captures/stitched
    python trainer/transcribe.py --limit-minutes 5     # try one file's first 5 minutes
    python trainer/transcribe.py FILE.wav [FILE.wav …]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
SR = 16000


def hms(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def duration(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / w.getframerate()


def transcribe_one(model, path: Path, args) -> dict:
    src = str(path)
    clip = None
    if args.limit_minutes:
        clip = {"start": 0, "end": args.limit_minutes * 60}

    segs, info = model.transcribe(
        src,
        language="he",
        word_timestamps=True,
        vad_filter=True,
        beam_size=args.beam,
        # Long unattended runs are the case where feeding the previous text back in goes wrong:
        # one bad segment can send the decoder into a repetition loop that then feeds itself for
        # minutes. Coherence across segment boundaries is worth less to us than never losing an
        # hour of a night's transcript to a loop.
        condition_on_previous_text=args.condition,
        clip_timestamps=[clip["start"], clip["end"]] if clip else "0",
    )

    out_segs = []
    words = 0
    last_report = time.time()
    total = args.limit_minutes * 60 if args.limit_minutes else duration(path)
    for s in segs:
        ws = [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
               "p": round(w.probability, 3)} for w in (s.words or [])]
        words += len(ws)
        out_segs.append({"start": round(s.start, 3), "end": round(s.end, 3),
                         "text": s.text.strip(), "words": ws})
        if time.time() - last_report > 30:
            done = s.end
            print(f"    {hms(done)} / {hms(total)}  ({100 * done / max(total, 1):4.1f}%)  "
                  f"{words} words", flush=True)
            last_report = time.time()

    return {"file": path.name, "seconds": round(total, 3), "model": args.model,
            "language": info.language, "segments": out_segs}


def write_outputs(path: Path, doc: dict) -> tuple[Path, Path]:
    jf = path.with_suffix(".words.json")
    tf = path.with_suffix(".transcript.txt")
    jf.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"[{hms(s['start'])}] {s['text']}" for s in doc["segments"]]
    tf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jf, tf


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="WAVs to transcribe (default: captures/stitched)")
    ap.add_argument("--in", dest="src", default=str(root / "captures" / "stitched"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--compute-type", default="int8", help="int8 (CPU) or float16 (GPU)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--beam", type=int, default=1, help="1 = greedy; higher is slower")
    ap.add_argument("--condition", action="store_true",
                    help="feed previous text back to the decoder (risks repetition loops)")
    ap.add_argument("--limit-minutes", type=float, default=0.0,
                    help="transcribe only the first N minutes of each file (for trying it out)")
    ap.add_argument("--force", action="store_true", help="redo files that already have output")
    args = ap.parse_args()

    files = ([Path(f) for f in args.files] if args.files
             else sorted(Path(args.src).glob("run_*.wav")))
    if not files:
        print(f"nothing to transcribe in {args.src}")
        return 1

    todo = [f for f in files if args.force or not f.with_suffix(".words.json").exists()]
    done_already = len(files) - len(todo)
    total_sec = sum(min(duration(f), args.limit_minutes * 60 or 1e9) for f in todo)
    print(f"{len(files)} file(s), {done_already} already transcribed, {len(todo)} to do")
    print(f"{total_sec / 3600:.2f} h of audio; at ~1.6x realtime expect "
          f"~{total_sec / 3600 / 1.6:.1f} h\n")
    if not todo:
        return 0

    from faster_whisper import WhisperModel        # imported late: loading it is slow
    print(f"loading {args.model} ({args.compute_type}, {args.threads} threads)"
          " - first run downloads ~1.6 GB", flush=True)
    t0 = time.time()
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type, cpu_threads=args.threads)
    print(f"loaded in {time.time() - t0:.0f}s\n", flush=True)

    run0 = time.time()
    audio_done = 0.0
    for i, f in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {f.name}", flush=True)
        t0 = time.time()
        try:
            doc = transcribe_one(model, f, args)
        except Exception as e:                                   # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        el = time.time() - t0
        audio_done += doc["seconds"]
        jf, tf = write_outputs(f, doc)
        nw = sum(len(s["words"]) for s in doc["segments"])
        rate = doc["seconds"] / max(el, 1e-9)
        left = (total_sec - audio_done) / max(audio_done / (time.time() - run0), 1e-9)
        print(f"    {doc['seconds'] / 60:.1f} min -> {len(doc['segments'])} segments, "
              f"{nw} words in {el / 60:.1f} min ({rate:.1f}x realtime)")
        print(f"    wrote {tf.name}; {left / 3600:.1f} h remaining\n", flush=True)

    print(f"done: {audio_done / 3600:.2f} h transcribed in "
          f"{(time.time() - run0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
