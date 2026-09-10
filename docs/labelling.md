# Labelling ad breaks

The phone records; all labelling happens here, on the desktop. This describes the loop.

## The files a stitched run has

Everything for one recording shares a stem, `run_<timestamp>`, in `captures/stitched/`:

| File | Written by | What it is |
|---|---|---|
| `.wav` | `stitch.py` | 16 kHz mono audio, contiguous |
| `.json` | `stitch.py` | which source segments went in, at what offset, with what gap |
| `.joins.txt` | `stitch.py` | Audacity point labels marking every join |
| `.words.json` | `transcribe.py` | segments and words with times and confidence |
| `.transcript.txt` | `transcribe.py` | readable timestamped Hebrew |
| `.markers.txt` | `ad_spans.py scan` | Audacity point labels where an ad phrase was found |
| `.draft.txt` | `ad_spans.py parse` | the machine's proposed ad spans |
| `.truth.txt` | **you, in Audacity** | the hand-marked spans |

`.draft.txt` and `.truth.txt` are separate on purpose, and so is `.joins.txt`: a draft that
overwrote the join markers would destroy the only record of where the audio is discontinuous.

## Marking by hand

1. Open the `.wav` in Audacity.
2. Switch the track to **Spectrogram** view from the track's dropdown menu. Ad breaks are visible
   before they are audible — commercials are dynamic-range compressed, so they appear as a denser,
   flatter, louder band than speech either side of them.
3. **File > Import > Labels** and choose the `.joins.txt`. Those vertical lines are the seams
   where the recorder restarted and about a second of audio is missing. A boundary you draw across
   one is not trustworthy; move it clear of the seam or leave that break out.
4. Select each advertising region and press **Ctrl+B**, then type `ad`.
5. **File > Export > Export Labels**, saving next to the `.wav` as `<stem>.truth.txt`.

Boundary precision has a floor and a ceiling. Frames are 0.96 s and the state machine has seconds
of hysteresis, so nothing finer than about half a second matters. What does matter is being
*consistent*: always start the label at the same point relative to the break, or the boundary model
learns your inconsistency rather than the signal.

### The convention, which is a choice and not a fact

Currently the broadcaster's promotion of its own programmes **counts as advertising** — both the
tag that opens a break and the promo reel that closes it. The case for is that it is what plays
during the break and what you would want silenced. The case against is that a promo reel can run
straight into real programming, so an over-eager boundary clips content. Either rule works; a rule
that changes halfway through the dataset does not.

## Getting a draft first

Never mark from a blank track twice. Two sources of draft, and they do different jobs:

    python trainer/ad_spans.py scan

finds the phrases that announce or close a break, with no model involved. On the first
half-hour tested, all seven hits fell inside the one true break — but every hit was closing
boilerplate, so the scan clusters in the middle of a break and identifies neither edge, and it
missed a promo reel entirely, because a broadcaster advertising itself has no small print to
recite. It is a reliable break detector and a useless boundary finder.

    python trainer/ad_spans.py prompt --out captures/prompts

writes a paste-ready prompt per run. Paste it into Claude, ChatGPT or Gemini, save the TSV reply,
and:

    python trainer/ad_spans.py parse run_20260909_235255 reply.txt

Every boundary the model returns is snapped onto the nearest word timestamp from the recogniser,
so the times in `.draft.txt` are the recogniser's whatever the model said. That split is the whole
design: a model asked to listen and report *when* drifts badly over a long file, so it is only ever
asked *which*.

## The gate

Before trusting drafts at scale, mark one recording by hand **without looking at the draft**, then:

    python trainer/compare_labels.py run_X.truth.txt run_X.draft.txt

It reports coverage in the units the app is judged in — ad-seconds heard per hour, content-seconds
muted per hour — and, separately, how far apart the edges are for breaks both tracks found. Those
fail differently: a draft that finds every break but is four seconds late everywhere is a fixable
prompt problem, while one that misses breaks outright is not.

Read the recall and precision, not the agreement figure. In the tool's own self-test, a draft that
misses a third of one break and invents another still scores 94.4 % agreement over the recording.
Accuracy is the wrong metric here and always will be, because most of any recording is not an ad.
