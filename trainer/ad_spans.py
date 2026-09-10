#!/usr/bin/env python3
"""
Turns transcripts into draft ad labels for Audacity.

Three subcommands, meant to be used in that order:

  scan    Finds the phrases that announce or close an ad break, with no model involved at all.
          Kan's sponsorship breaks introduce themselves - "התחזית בחסות", "השעה בחסות" - and the
          spots close with consumer boilerplate that news copy never uses. This costs nothing and
          already anchors most break *starts*. It is deliberately a keyword scan and nothing more:
          it says "an ad begins near here", never where one ends.

  prompt  Writes paste-ready prompts, one per chunk, for a chat model. The model is asked only
          which lines are advertising, and to copy the timestamps it is given rather than compute
          any - it is good at the first and bad at the second.

  parse   Reads the model's reply back, snaps every boundary onto the nearest word timestamp from
          the recogniser, and writes an Audacity label track. After this the times in the label
          file are the recogniser's to within a few tens of milliseconds, whatever the model said.

These are drafts. You correct them in Audacity; you do not trust them. trainer/compare_labels.py
measures how far a draft sits from a hand-marked truth, and that number is what decides how much
correcting a draft still needs.

Usage:
    python trainer/ad_spans.py scan
    python trainer/ad_spans.py prompt --out prompts/
    python trainer/ad_spans.py parse run_20260909_235255 reply.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Seeded from what showed up in the first transcripts, not from a general model of advertising.
# Weight is how strongly the phrase implies an ad; "בחסות" is close to decisive on this station,
# whereas a phone number merely leans that way. Revisit these once compare_labels.py has run
# against hand-marked truth - at that point they can be checked rather than guessed at.
MARKERS: list[tuple[str, float, str]] = [
    ("בחסות", 0.95, "sponsorship announcement"),
    # The presenters announce the break out loud - "we are going to a short break, commercials".
    # Found by reading transcripts, not by guessing: this is the only marker that reliably lands
    # at the START of a break, where all the legal boilerplate lands at the end of each spot.
    ("פרסומות", 0.95, "presenter announces the break"),
    ("הפסקה קצרה", 0.9, "presenter announces the break"),
    ("הפסקה קטנה", 0.9, "presenter announces the break"),
    ("מיד חוזרים", 0.8, "presenter going into the break"),
    ("מיד נחזור", 0.8, "presenter going into the break"),
    # Match the stable part of a legal phrase, never the whole wording: the recogniser rendered
    # "יש לעיין בעלון לצרכן לפני השימוש" as "יש לעיין בלון הצרכן לפני השימוש", so a marker for the
    # full phrase found nothing while "יש לעיין" finds it every time.
    ("יש לעיין", 0.9, "consumer leaflet boilerplate"),
    ("לפני השימוש", 0.8, "consumer leaflet boilerplate"),
    ("כפוף לתנאי", 0.9, "terms and conditions"),
    ("בכפוף לתקנון", 0.9, "terms and conditions"),
    ("כפוף לתקנון", 0.9, "terms and conditions"),
    ("אינו מהווה תחליף", 0.85, "medical disclaimer"),
    ("ט.ל.ח", 0.85, "errors and omissions excepted"),
    ("המבצע מוגבל", 0.8, "offer boilerplate"),
    ("בכפוף למלאי", 0.8, "offer boilerplate"),
    ("לתעריפון", 0.8, "price-list boilerplate"),
    ("לפרטים נוספים", 0.6, "call to action"),
    ("להזמנות", 0.6, "call to action"),
    ("חייגו", 0.6, "call to action"),
    ("סניפים", 0.5, "retail copy"),
    ("רשת סופרמרקטים", 0.5, "retail copy"),
]
PHONE = re.compile(r"\*\d{3,4}\b")            # Israeli shortcodes: *2000, *6800
WEB = re.compile(r"\b(www\.|\S+\.co\.il)\b")


def hms(t: float) -> str:
    h, r = divmod(max(t, 0.0), 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def parse_hms(s: str) -> float:
    parts = [float(p) for p in s.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load(base: Path) -> dict:
    jf = base if base.suffix == ".json" else base.with_suffix(".words.json")
    if not jf.exists():
        raise SystemExit(f"no transcript: {jf} (run trainer/transcribe.py first)")
    return json.loads(jf.read_text(encoding="utf-8"))


def all_words(doc: dict) -> list[dict]:
    return [w for s in doc["segments"] for w in s["words"]]


# ---------------------------------------------------------------- scan

def scan_doc(doc: dict) -> list[dict]:
    hits = []
    for s in doc["segments"]:
        text = s["text"]
        for phrase, weight, why in MARKERS:
            i = text.find(phrase)
            if i < 0:
                continue
            # Locate the phrase in time by finding the word it starts in, so the hit points at the
            # phrase rather than at the start of a segment that may run for a minute and a half.
            at = s["start"]
            run = ""
            for w in s["words"]:
                run += w["w"]
                if len(run) >= i:
                    at = w["s"]
                    break
            hits.append({"t": at, "weight": weight, "why": why, "phrase": phrase,
                         "text": text[max(0, i - 40):i + 60]})
        for rx, why in ((PHONE, "phone shortcode"), (WEB, "web address")):
            if rx.search(text):
                hits.append({"t": s["start"], "weight": 0.6, "why": why,
                             "phrase": rx.search(text).group(0), "text": text[:100]})
    hits.sort(key=lambda h: h["t"])
    return hits


def speech_coverage(doc: dict) -> float:
    """Fraction of the recording the recogniser produced words for. Low means music rather than
    failure - an overnight song programme transcribes as sparse lyrics - but either way a
    transcript that covers a quarter of the audio cannot be labelled from."""
    total = sum(s["end"] - s["start"] for s in doc["segments"])
    return 100.0 * total / max(doc["seconds"], 1e-9)


def cmd_summary(args) -> int:
    """One line per recording: is there speech, and is there advertising."""
    files = sorted(Path(args.src).glob("*.words.json"))
    if not files:
        print(f"no transcripts in {args.src} - run trainer/transcribe.py first")
        return 1
    print(f"{'file':28s} {'min':>5s} {'speech%':>7s} {'hits':>5s} {'strong':>6s}  "
          f"breaks near (min)")
    ads = strong_all = 0
    hours_all = 0.0
    for jf in files:
        doc = load(jf)
        hits = scan_doc(doc)
        strong = [h for h in hits if h["weight"] >= 0.85]
        strong_all += len(strong)
        cov = speech_coverage(doc)
        # Strong hits within two minutes of each other are one break, not several.
        cl: list[list[float]] = []
        for h in strong:
            if cl and h["t"] - cl[-1][-1] <= 120:
                cl[-1].append(h["t"])
            else:
                cl.append([h["t"]])
        ads += len(cl)
        hours_all += doc["seconds"] / 3600
        note = "  <-- music: too sparse to label from text" if cov < 40 else ""
        print(f"{jf.name[:-11]:28s} {doc['seconds'] / 60:5.1f} {cov:7.1f} "
              f"{len(hits):5d} {len(strong):6d}  "
              f"{', '.join(f'{c[0] / 60:.0f}' for c in cl)}{note}")
    print()
    per_hour = ads / max(hours_all, 1e-9)
    print(f"{strong_all} strong hit(s) in about {ads} break(s) across {len(files)} file(s)")
    print(f"{hours_all:.2f} h of audio, {per_hour:.2f} break(s) per hour")
    print()
    print("A file with zero hits is usually a real answer rather than a failure. Measured over")
    print("11.72 h of this station on 2026-09-10: 4.29 breaks/hour between 07:30 and 12:10,")
    print("3.50 between 05:30 and 07:30, and 0.41 between 00:30 and 05:30. Record daytime.")
    return 0


def cmd_scan(args) -> int:
    files = sorted(Path(args.src).glob("*.words.json"))
    if not files:
        print(f"no transcripts in {args.src} - run trainer/transcribe.py first")
        return 1
    grand = 0
    for jf in files:
        doc = load(jf)
        hits = scan_doc(doc)
        grand += len(hits)
        strong = [h for h in hits if h["weight"] >= 0.85]
        print(f"\n=== {doc['file']}  ({doc['seconds'] / 60:.1f} min) ===")
        print(f"{len(hits)} marker hit(s), {len(strong)} strong")
        for h in (hits if args.all else strong):
            print(f"  {hms(h['t'])}  {h['weight']:.2f}  {h['why']:28s} {h['phrase']}")
        if not args.dry_run:
            lab = jf.with_name(jf.name.replace(".words.json", ".markers.txt"))
            lab.write_text("".join(
                f"{h['t']:.3f}\t{h['t']:.3f}\t{h['phrase']} ({h['weight']:.2f})\n"
                for h in hits), encoding="utf-8")
            print(f"  -> {lab.name}")
    print(f"\n{grand} hit(s) across {len(files)} file(s)")
    return 0


# ---------------------------------------------------------------- prompt

INSTRUCTIONS = """\
Below is a timestamped transcript of Israeli public radio (Kan Reshet Bet), in Hebrew.

Your task: identify every stretch that is ADVERTISING rather than programme content.

Count as advertising:
  - commercial spots for products, shops, banks, insurance, medicines
  - sponsorship announcements ("בחסות ...") and the sponsored item's tag line
  - promotional trailers for the broadcaster's own television or radio programmes
  - jingles and station idents that sit inside a commercial break

Do NOT count as advertising:
  - news, interviews, analysis, weather, traffic, sport
  - the presenters introducing an item or each other
  - headlines and time checks that are not attached to a sponsor

Output rules, which matter more than anything else here:
  - Output ONLY a TSV table, no commentary before or after it.
  - Four columns: START<TAB>END<TAB>CONFIDENCE<TAB>WHAT
  - START and END must be COPIED VERBATIM from the [HH:MM:SS.ss] stamps in the transcript.
    Do not calculate, adjust or interpolate a time. If a break starts partway through a line,
    use that line's own stamp.
  - CONFIDENCE is high, medium or low.
  - WHAT is a few words naming the advertiser or the kind of spot.
  - One row per continuous advertising stretch. Merge adjacent spots that run back to back into
    a single row only if nothing else comes between them.

Transcript follows.
"""


def cmd_prompt(args) -> int:
    files = sorted(Path(args.src).glob("*.words.json"))
    if not files:
        print(f"no transcripts in {args.src} - run trainer/transcribe.py first")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for jf in files:
        doc = load(jf)
        stem = doc["file"].replace(".wav", "")
        lines = [f"[{hms(s['start'])}] {s['text']}" for s in doc["segments"]]
        chunks, cur, size = [], [], 0
        for ln in lines:
            if size + len(ln) > args.max_chars and cur:
                chunks.append(cur)
                cur, size = [], 0
            cur.append(ln)
            size += len(ln) + 1
        if cur:
            chunks.append(cur)
        for i, ch in enumerate(chunks, 1):
            suffix = f"_{i}of{len(chunks)}" if len(chunks) > 1 else ""
            p = out / f"{stem}{suffix}.prompt.txt"
            p.write_text(INSTRUCTIONS + "\n" + "\n".join(ch) + "\n", encoding="utf-8")
            n += 1
    print(f"wrote {n} prompt file(s) to {out}")
    print("Paste one into Claude, ChatGPT or Gemini; save the TSV reply, then:")
    print("  python trainer/ad_spans.py parse <run stem> <reply.txt>")
    return 0


# ---------------------------------------------------------------- parse

# Be generous about what a row looks like. The prompt tells the model to copy timestamps verbatim
# from the "[HH:MM:SS.ss]" stamps in the transcript, and a model that does exactly as it is told
# returns them still wrapped in brackets - so the brackets must be optional rather than a parse
# failure. Markdown pipe tables and comma separators are accepted for the same reason: the reply
# comes out of a chat window, and rejecting a correct answer over its punctuation wastes the
# user's evening, not ours.
_T = r"\[?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*\]?"
_SEP = r"(?:\s*[\t|,]\s*|\s{2,})"
ROW = re.compile(r"^\s*\|?\s*" + _T + _SEP + _T + _SEP + r"?(.*)$")


def split_rest(rest: str) -> tuple[str, str]:
    """Confidence and description out of whatever followed the two times."""
    parts = [c.strip() for c in re.split(r"[\t|]", rest) if c.strip()]
    if len(parts) < 2 and "," in rest:
        # Comma-separated: only the FIRST comma divides the columns, since a description like
        # "Machsanei Chashmal, BTB, AIG" carries commas of its own.
        head, _, tail = rest.partition(",")
        parts = [head.strip(), tail.strip()]
    if not parts:
        return "?", "ad"
    known = parts[0].lower() in ("high", "medium", "low")
    return (parts[0] if known else "?"), (parts[-1] if len(parts) > 1 or not known else "ad")


def snap(t: float, edges: list[float]) -> float:
    """Moves a time onto the nearest word edge. The model's times are only ever as good as the
    line it copied them from; the recogniser's are good to tens of milliseconds, so we keep those
    and discard the model's precision entirely."""
    if not edges:
        return t
    i = min(range(len(edges)), key=lambda k: abs(edges[k] - t))
    return edges[i]


def cmd_parse(args) -> int:
    base = Path(args.src) / args.stem
    doc = load(base)
    words = all_words(doc)
    starts = [w["s"] for w in words]
    ends = [w["e"] for w in words]

    reply = Path(args.reply).read_text(encoding="utf-8")
    spans, bad = [], 0
    for line in reply.splitlines():
        m = ROW.match(line.replace("‏", ""))
        if not m:
            if line.strip() and not line.lstrip().startswith(("START", "#", "-", "|--")):
                print(f"  ignored (not a table row): {line.strip()[:90]}")
                bad += 1
            continue
        a, b = parse_hms(m.group(1)), parse_hms(m.group(2))
        conf, what = split_rest(m.group(3))
        if b <= a:
            print(f"  ignored (end is not after start): {line.strip()[:90]}")
            bad += 1
            continue
        spans.append((snap(a, starts), snap(b, ends), conf, what))

    if not spans:
        # Overwriting a good label file with an empty one, having said nothing useful about why,
        # is the worst thing this could do. Refuse, and show what could not be read.
        print(f"\nNo usable rows in {args.reply} - nothing was written.")
        print("Expected each row to be: START<TAB>END<TAB>CONFIDENCE<TAB>WHAT")
        print("with times like 00:04:05.12. Square brackets, | tables and commas are all fine.")
        return 1

    spans.sort()
    merged: list[list] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < args.merge_gap:
            merged[-1][1] = max(merged[-1][1], s[1])
            merged[-1][3] += f"; {s[3]}"
        else:
            merged.append(list(s))

    # Deliberately not ".labels.txt": stitch.py owns that name for join markers, and a draft
    # that overwrote them would destroy the only record of where the audio is discontinuous.
    lab = Path(args.out) if args.out else base.with_suffix(".draft.txt")
    lab.write_text("".join(f"{a:.3f}\t{b:.3f}\tad ({c}) {w}\n" for a, b, c, w in merged),
                   encoding="utf-8")
    total = sum(b - a for a, b, _, _ in merged)
    print(f"{len(spans)} span(s) parsed, {len(merged)} after merging, {bad} line(s) ignored")
    print(f"{total / 60:.1f} min of ads in {doc['seconds'] / 60:.1f} min "
          f"({100 * total / max(doc['seconds'], 1):.1f}%)")
    print(f"-> {lab}")
    print("Import it in Audacity (File > Import > Labels) and correct it.")
    return 0


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(root / "captures" / "stitched"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="keyword scan, no model needed")
    s.add_argument("--all", action="store_true", help="show weak hits too")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_scan)

    m = sub.add_parser("summary", help="one line per recording: speech and advertising found")
    m.set_defaults(fn=cmd_summary)

    p = sub.add_parser("prompt", help="write paste-ready prompts for a chat model")
    p.add_argument("--out", default=str(root / "captures" / "prompts"))
    p.add_argument("--max-chars", type=int, default=40000)
    p.set_defaults(fn=cmd_prompt)

    q = sub.add_parser("parse", help="turn a model's TSV reply into Audacity labels")
    q.add_argument("stem", help="e.g. run_20260909_235255")
    q.add_argument("reply", help="file holding the model's TSV reply")
    q.add_argument("--out", default="", help="write labels here instead of <stem>.draft.txt")
    q.add_argument("--merge-gap", type=float, default=5.0,
                   help="join spans closer together than this many seconds")
    q.set_defaults(fn=cmd_parse)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
