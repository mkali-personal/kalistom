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

### Open, and important for Phase 4

Whether a **session-0 `AudioEffect`** (the actuator chosen in Phase 0) also sits after the capture
tap is **untested**. If that effect *did* attenuate the captured signal it would reintroduce exactly
the blindness described above, and the app-agnostic actuator would have to be abandoned in favour of
audio-focus ducking. Cheap to test: run the effect probe during a recording and read the rms trace.
**Do this before building anything on the session-0 actuator.**

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
