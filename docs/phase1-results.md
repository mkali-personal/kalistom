# Phase 1 results — 2026-09-09

The recorder works and its central invariant is proven, not assumed. Phase 2 (labelling) is
unblocked. Two sessions recorded from Kan on a Galaxy S23 / Android 16.

## Alignment: verified

| Session | Duration | Frames | Samples | rms match (median / worst) |
|---|---|---|---|---|
| `sess_20260909_224402` | 214.1 s | 444 | 3,425,280 | 0.0002 dB / 0.0005 dB |
| `sess_20260909_225653` | 111.8 s | 231 | 1,789,440 | 0.0003 dB / 0.0005 dB |

Both match `(samples − WINDOW)/HOP + 1` exactly. The check is an exact proof rather than a
plausibility test: `trainer/ingest.py --verify-alignment` recomputes each frame's RMS from the WAV
using the documented mapping and compares it to what the device wrote. Agreement at 0.0002 dB is
floating-point noise. A dropped frame would make the error explode and stay large, which is why the
tool distinguishes a persistent offset from an isolated glitch.

## Embedding sanity

- No NaN, no Inf, no all-zero frames; values in [0, 4.84] (post-ReLU, expected).
- Adjacent-frame cosine 0.638 vs 0.428 at 19 s apart — the embeddings vary meaningfully over time
  rather than collapsing, which is what makes them learnable.
- YAMNet top-1 over 444 frames: **63.7 % Speech, 35.6 % Music**, remainder negligible. Plausible for
  talk radio with ad breaks. Whether the speech/music split itself carries ad signal (ads usually
  run over a music bed) is a Phase 3 hypothesis, not a finding.

## Capture is independent of the media volume

Stepped the media volume 15 → 10 → 5 → 1 → 0 during a live recording
(`tools/volume_test.sh`), then read the per-frame RMS trace back.

| Stage | Window | Frames | Median rms | Δ vs full |
|---|---|---|---|---|
| 15/15 | 10–25 s | 32 | −23.98 dBFS | — |
| 10/15 | 25–40 s | 31 | −24.47 dBFS | −0.49 dB |
| 5/15 | 40–55 s | 31 | −23.84 dBFS | +0.14 dB |
| 1/15 | 55–70 s | 31 | −23.64 dBFS | +0.34 dB |
| **0/15** | 70–85 s | 32 | **−23.91 dBFS** | **+0.07 dB** |

At volume 0 the signal is unchanged; had the tap sat after the volume stage those frames would be
near −120 dBFS. `AudioPlaybackCapture` reads **before** the stream-volume stage.

### Why this matters twice

1. **Data collection can be silent.** Record for days without hearing anything.
2. **The detector will keep seeing the audio while it is muting** — which the design quietly
   depended on. If capture had followed the volume, then the moment Phase 4 muted an ad the model
   would have been fed silence and could never detect the ad *ending*: it would mute once and stay
   stuck until a timeout. That failure mode is now ruled out for the volume path.

### Session-0 attenuation was tested too — it also sits after the capture tap

The Phase 4 actuator is an `AudioEffect` on session 0. If that had attenuated the capture path it
would have reintroduced exactly the blindness above. It does not.

Run in-process (as Phase 4 will), with markers written into the session so stage boundaries come
from sample offsets rather than wall clock:

| Stage | Frames | Median rms | Δ vs baseline |
|---|---|---|---|
| baseline | 22 | −23.49 dBFS | — |
| `LoudnessEnhancer` −40 dB | 22 | −23.91 dBFS | **−0.42 dB** |
| off (control) | 22 | −23.67 dBFS | −0.18 dB |
| `DynamicsProcessing` −60 dB | 23 | −24.45 dBFS | **−0.96 dB** |
| off (control) | 22 | −23.77 dBFS | −0.28 dB |
| off (control) | 6 | −24.23 dBFS | −0.74 dB |

The control stages drift −0.18 to −0.74 dB on programme material alone, so −0.42 and −0.96 dB are
indistinguishable from doing nothing. A −40 dB attenuation of the capture path would have been
unmistakable.

**The audible control was confirmed by the user: the output dropped out twice during the run.**
Without that, "capture unaffected" would have been trivially true and meaningless — the effects
could simply have done nothing at all.

So both attenuation paths, volume and session-0 effect, leave the captured signal intact. The
detector keeps watching through its own mute and can therefore detect the ad *ending*.

`Attenuator` is committed as the real actuator, not scaffolding.

**Still owed before Phase 4 ships:**
- A **watchdog**. The effect attenuates everything — alarms, navigation, calls. `releaseQuietly()`
  is called from `stopAll()` and the test's `finally`, but nothing yet force-releases on a stuck
  state, process death, or a mute that has run implausibly long.
- **Route changes** (Bluetooth connect/disconnect, headphone unplug) untested.

## Deviations from the roadmap

- **WAV, not Opus.** Opus needs FFmpeg inside Audacity to open; FLAC needs hand-assembled container
  headers from MediaCodec's `csd-0`. Neither risk belongs in the phase whose reliability is
  load-bearing. Cost: 115 MB/h rather than ~11, against 24 GB free, with a 6 GB rotating budget.
  Compression can move into desktop ingest later at no risk.
- **No resampler.** `AudioRecord` was asked for 16 kHz mono directly and the framework honoured it
  (`capture: 16000 Hz, 1 ch`), so there is no hand-rolled DSP and no opportunity for drift.

## Bugs found and fixed

- **Framing.** `WINDOW` (15600) is not a multiple of `HOP` (7680). The first shift-then-append loop
  dropped most of each read from the analysis window while still writing it to the WAV — embeddings
  would have desynchronised with no visible symptom. Replaced with a buffer that advances by exactly
  HOP per frame; verified by simulation that frame *k* starts at exactly *k*×HOP.
- **Per-frame RMS** was taken from the just-read chunk, which lies mostly outside the frame's own
  window. Now computed over the window itself — which is also what makes the exact alignment check
  possible.
- **`.gitignore`** pattern `assets/*.tflite` is anchored to the repo root and never matched
  `app/src/main/assets/`, so the 16 MB model was staged. Now `*.tflite`, with
  `tools/fetch_model.sh` to re-download.
- **Double Start** granted a MediaProjection consent that was then discarded (the service correctly
  ignored the second start). Now reports "already recording" instead of prompting.

## Still uncharacterised

MediaProjection lifecycle on Android 16: screen-off, process restart, reboot. This matters more now
than it did in Phase 0, because Phase 2 expects days of unattended recording and the consent token
is single-use.

## Desktop stream recording (added after Phase 1)

Recording the same broadcast on the laptop lets data accumulate overnight instead of requiring the
phone be carried around, and it avoids the projection-consent and battery questions entirely. Two
things were measured before trusting it.

**ffmpeg quits early on this live stream, intermittently.** One run stopped after 44 s of a
requested 300 s and still exited with status 0 — nothing errored, ffmpeg simply decided the live
playlist had ended. The `-reconnect` options do not cover that, since there is no error to recover
from. `tools/record_stream.sh` therefore runs each segment as its own ffmpeg invocation under a
supervisor loop, so an early exit costs the tail of one segment rather than the night, and reports
how many segments came back short.

**Consecutive segments do not overlap; each restart loses about a second.** An earlier note here
claimed roughly 8 seconds of overlap per restart, inferred from a `skew_test.py` alignment score.
That was wrong, and the correction matters because it reverses what the stitching step has to do.
Measured over the 148 restarts of the 2026-09-10 overnight run: no two consecutive segments share
a single byte of audio, tested by searching each segment's first 0.25 s verbatim within the
previous segment's last 120 s. Correlation had suggested otherwise only because ordinary speech
similarity produces peaks in the 0.4–0.66 range, well short of proof; a quarter-second exact byte
run cannot occur by chance, so the verbatim test settles it.

What restarts actually cost is a small hole. Estimating each gap as
`(exit_time_B − duration_B) − exit_time_A` — both invocations have converged to the stream's live
edge by the time they exit — gives a median of +0.10 s, a 5th–95th percentile range of −1.8 s to
+2.5 s, a maximum of +3.4 s, and 108 s lost across the whole 7.92-hour night, which is 0.4 %.

So there is no duplication to trim, and no risk of the same audio landing in both a training and a
test split. There is instead a seam at every join. `trainer/stitch.py` butt-joins the segments,
records each join and its estimated gap in a per-run JSON manifest, and writes an Audacity label
track marking the joins, so a boundary drawn across one is visible as suspect rather than
invisible. A gap over 5 s closes the run and starts a new file, because contiguity that cannot be
shown should not be implied.

The overnight run reduced this way from 149 segments to 15 files of about half an hour each,
covering 7.92 hours with 134 joins and no break over 5 s.

**Desktop and phone audio are interchangeable — measured, not assumed.** A phone session and a
desktop segment covering the same five minutes of broadcast were aligned (envelope correlation
0.998, offset 347.5 s, refined to 1.2 ms) and compared on identical content:

| Band | Phone | Desktop | Diff |
|---|---|---|---|
| 0–300 Hz | 64.3 % | 64.3 % | +0.0 |
| 300–1000 Hz | 27.1 % | 27.1 % | +0.0 |
| 1000–3400 Hz | 8.0 % | 8.0 % | +0.0 |
| 3400–6000 Hz | 0.5 % | 0.5 % | −0.0 |
| 6000–8000 Hz | 0.0 % | 0.0 % | −0.0 |

Waveform correlation after alignment: **r = 0.9986**. The phone app and ffmpeg are receiving the
same rendition, so what reaches YAMNet is the same signal either way. Desktop recordings are valid
training data for a model that will run on the phone.

This retracts an earlier suspicion. A first comparison suggested the phone carried far more energy
above 3.4 kHz (3.3 % against 0.7 %), but those were recordings of different moments — talk versus
music — and on identical content the difference vanishes entirely.

**Remaining difference: where the embeddings are computed.** The audio matches, but embeddings
generated on the laptop would come from a different TFLite runtime than the phone's. The clean fix
is an offline mode in the app that reads a WAV and emits the same three-file session it would have
produced live, so desktop recordings are processed by the exact code path that will run in
production. That also makes every past recording reprocessable if the model or hop size changes.

## Offline processing on the phone (proven equivalent to live recording)

Laptop audio matches phone audio (r = 0.9986), but embeddings computed on the laptop would come
from a different TFLite runtime. So the app gained an offline mode: WAVs are copied to an inbox on
the device, processed through the phone's own YAMNet, and only the `.f16` and `.jsonl` come back,
named after the input. Ingest pairs them with the laptop's copy of the audio — which means the
verifier's rms check also proves the phone processed exactly the file we think it did.

To rule out any drift between the two paths, the framing logic was extracted into a single
`FrameEmitter` used by both live capture and offline processing. It was then tested by feeding the
phone two WAVs it had recorded itself and comparing the result against the embeddings it produced
at the time:

| Session | Frames | Live bytes | Offline bytes | Result |
|---|---|---|---|---|
| `sess_20260909_232448` | 119 | 243,712 | 243,712 | **byte-identical** |
| `sess_20260909_232603` | 85 | 174,080 | 174,080 | **byte-identical** |

So offline processing is not merely close to live recording, it is the same computation. The same
test also confirms the `FrameEmitter` refactor left the recorder's behaviour unchanged.

Two consequences worth keeping in mind. Any recording ever made can be reprocessed if the model or
the hop size changes, rather than being wasted. And the laptop never needs a TFLite runtime
installed, which is convenient, since it does not have one.

### Delay between the two sources

The laptop stream lags the phone app by about **347 seconds** — nearly six minutes. This is why
`trainer/skew_test.py` aligns before it compares; a naive comparison of the two on wall-clock time
would show almost no correlation and imply, wrongly, that they carry different audio.

## Overnight recording is the wrong shift (2026-09-10)

The first full night, 23:52 to 07:48, gave 7.92 hours across 15 stitched files. Transcribed and
scanned for the phrases that mark a commercial break, the advertising turns out to be almost
entirely absent:

| Hours | Files | Ad breaks found |
|---|---|---|
| 23:52-00:23 | 1 | 1 |
| 00:23-05:52 | 10 | **0** |
| 05:52-07:48 | 4 | 6 |

This is the broadcast, not a failure of the scan. `כפוף` - the single strongest indicator, the
opening of the terms-and-conditions boilerplate - occurs 7 times in the 23:52 file, **zero times in
all ten files between 00:23 and 05:52**, and returns from 05:52 onward. `בחסות` occurs only after
06:52. (High counts of `רשת` are a false friend: it is the station's own name, `רשת ב׳`.)

So the night yielded roughly 7 breaks, about 18 minutes of advertising in 475 minutes, or 3.8 % -
against the 10-20 % the plan assumed. As a training set that is badly short of positive examples,
and the shortfall is not fixable by recording more nights. **Record daytime instead**, where
commercial density is highest: the morning drive and the afternoon drive.

Two further findings from the same pass:

- **Some overnight programming is music, and cannot be labelled from text.** Speech coverage - the
  fraction of the recording the recogniser produced words for - runs 83-92 % on talk programmes but
  falls to 25 % on the 04:15 file and 55-60 % on two others, which are song programmes transcribing
  as sparse lyrics. `ad_spans.py summary` flags anything under 40 %. Those hours would need marking
  by ear; they happen to contain no advertising, so nothing is lost this time.
- **The keyword scan finds breaks but never their edges.** On the one file labelled in full, all
  seven hits fell inside the true break, so precision was perfect - but every hit was *closing*
  boilerplate, so the hits cluster mid-break, and the scan missed a station promo reel entirely,
  because a broadcaster advertising its own programmes recites no small print. Break detection and
  boundary finding are separate jobs; the model pass does the second.

### Daytime confirms it, measured (2026-09-10)

A second recording covering 07:22 to 12:07 was added, bringing the corpus to 11.72 h. Counting ad
breaks by time of day settles the question:

| Period | Hours | Breaks | Per hour |
|---|---|---|---|
| 23:50-00:30 | 1.07 | 2 | 1.86 |
| 00:30-05:30 | 4.92 | 2 | **0.41** |
| 05:30-07:30 | 2.00 | 7 | 3.50 |
| 07:30-12:10 | 3.73 | 16 | **4.29** |
| total | 11.72 | 27 | 2.30 |

Daytime carries **ten times** the ad density of the small hours. An hour recorded at 10:00 is worth
about ten hours recorded at 03:00, so labelling effort and disk should both go to daytime.

The count rose from the 7 breaks first reported partly because the corpus grew and partly because
the scan improved: reading transcripts turned up that presenters announce breaks out loud
("יוצאים להפסקה קטנה, פרסומות"), which is the only marker that lands at the *start* of a break -
every piece of legal boilerplate lands at the end of a spot instead.

**A limitation of transcript-based labelling, found the same way.** In several breaks the
recogniser produces nothing for 30 to 70 seconds while the audio continues at a normal -24 dBFS.
That is not silence; it is jingles, stings and sung sponsor beds carrying no intelligible speech.
So a transcript can say that a break is happening and roughly where it ends, but it cannot see
inside those stretches, and a label drawn across one is partly guesswork. Two rebroadcast
programmes show the mirror image: the host says "פרסומות" but only a station ident follows,
because a rebroadcast carries the cue without the advertising.

## First model that works (2026-09-10)

Twelve recordings, 31 labelled ad breaks, 6.14 h, leave-one-recording-out. Every held-out fold is
a different hour of broadcast with different presenters and a partly different pool of
advertisements, so a fold's score reflects generalisation rather than jingle recognition.

| | 4 recordings, 7 breaks | 12 recordings, 31 breaks |
|---|---|---|
| pooled precision | 57.6 % | **83.6 %** |
| pooled recall | 61.1 % | **90.6 %** |
| best trade (saved per second lost) | 1.8 | **12.3** |
| content-seconds lost at that point | 79 | **26** |

At `on=0.99 / off=0.98` the model removes 334 of the 481 ad-seconds in an average hour while
wrongly muting 26 seconds of programme. That is still above the plan's 10 s/h target, but it is a
usable trade rather than the roughly one-for-one exchange the smaller dataset produced.

**The gain came from labels, not from modelling.** Nothing about the head changed between the two
columns except the optimiser, which only made it faster. Quadrupling the labelled breaks did all
of it - which is the plan's cold-start prediction, measured.

**A split in the results that needs explaining before it is trusted:**

| Group | Best trade | Content-seconds lost |
|---|---|---|
| the 4 recordings labelled first | 3.8 | 41 |
| the 8 daytime recordings labelled later | **17.7** | **21** |

Two explanations fit. The early recordings may simply be harder - they are the sparse ones, with a
single break each, at hours when the station barely advertises. Or the labelling convention drifted
between the first batch and the second: the later files were marked after the presenter-announcement
markers were found and after many transcripts had been read, so the boundaries may be drawn more
consistently. Those have opposite remedies, and nothing in the data distinguishes them.

**This is what the human gate is for.** Every number above rests on labels produced by a model, and
their error has never been measured. Marking one recording by hand, without looking at the draft
first, and running `trainer/compare_labels.py` would settle both the general question and this
specific one.
