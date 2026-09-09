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

**Consecutive segments overlap by about 8 seconds**, measured with `trainer/skew_test.py`'s
alignment (scores 0.995–0.999, so this is certain rather than inferred). Each restart resumes from
the stream's live edge, which lags real time by the buffer depth, so the new segment re-delivers
audio the previous one already had. At 30-minute segments that is under 0.5 % duplication, but
identical audio appearing in both a training and a test split would inflate results, so ingest
should trim each segment's head against the previous segment's tail.

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
