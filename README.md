# Kalistom

A private, sideloaded Android app that detects commercials in the phone's **playback** audio and
drops the volume until they end. Personal use only — never published to Play.

Design review, issues and full roadmap: see the plan at
`~/.claude/plans/in-android-app-for-starry-lampson.md`, derived from
`Android App for Commercial Detection.pdf`.

## Where the project stands

Findings: [`docs/phase0-results.md`](docs/phase0-results.md), [`docs/phase1-results.md`](docs/phase1-results.md).

| Phase | What | Status |
|---|---|---|
| 0a | Toolchain + zero-code manifest probe | **PASSED** - nothing opted out; Spotify opts in |
| 0b | Capture feasibility spike | **PASSED** - Kan captured cleanly on Android 16 |
| 1 | Session recorder (`.wav` + `.f16` + `.jsonl`) | **DONE** - alignment verified to 0.0002 dB |
| 2 | Audacity labelling + offline eval harness | next |
| 3 | Model + HMM smoothing | not started |
| 4 | Actuator (duck / fade / skip) | not started |
| 5 | Fingerprints + MediaSession metadata | not started |
| 6 | Hardening | not started |

**The Phase 0 gate has passed** (2026-09-09, Galaxy S23 / Android 16). Phase 1 is unblocked.

Two findings that changed the design:

- An unprivileged app *can* attenuate the global output mix via an `AudioEffect` on session 0,
  which displaces `setStreamVolume` as the planned actuator.
- **Capture is independent of the media volume** - measured flat to +0.07 dB with the slider at 0.
  So you can record training data with the phone silent, and the detector keeps seeing audio while
  it mutes. The session-0 effect was tested the same way and also leaves capture intact (-0.4 dB
  against -0.2..-0.7 dB control drift), so the app-agnostic actuator is confirmed viable.

## Setting up on another computer

The desktop side needs very little, and what it needs depends on what that machine is for.

**To record the live stream and nothing else** — the case for a spare machine left running
overnight — you need `ffmpeg` on the PATH and a clone of this repository. Nothing else at all:
no Python packages, no model, no Android SDK.

```bash
git clone https://github.com/mkali-personal/kalistom.git
cd kalistom
tools/record_stream.sh 8                       # 8 hours into captures/desktop
```

Check first that the machine will not sleep, hibernate or install updates overnight. A machine
that suspends at 2 a.m. leaves you a broken final segment and nothing after it, and you will not
find out until morning. Press Ctrl+C to stop rather than closing the window, so ffmpeg finalises
the file; if it is killed instead, `python trainer/repair_wav.py captures/desktop/*.wav` recovers
the audio, which is all still on disk.

**To also stitch, transcribe and label** on that machine:

```bash
pip install --user -r trainer/requirements.txt
python trainer/stitch.py                       # 149 segments -> ~15 half-hour files
python trainer/transcribe.py                   # Hebrew ASR with word-level timestamps
```

The Hebrew Whisper model is not in the repository and does not need to be fetched by hand:
`faster-whisper` downloads it on first use, about 1.6 GB, into `~/.cache/huggingface`. Expect
roughly 1.2x realtime on a laptop CPU, so a night's recording takes about six hours.

**Recordings do not travel through git** — `captures/` is ignored, and a night is about 900 MB.
Move them on a USB stick, over a network share, or with Syncthing pointed at the output directory.
Converting to FLAC first (`ffmpeg -i in.wav -c:a flac out.flac`) is lossless, roughly halves the
size, and Audacity opens FLAC natively.

**To build or run the Android app** you additionally need the Android SDK and one download that is
kept out of git:

```bash
tools/fetch_model.sh                           # YAMNet, 16 MB, into app/src/main/assets/
```

Labelling in Audacity: see [`docs/labelling.md`](docs/labelling.md).

## Recording a session

```bash
tools/fetch_model.sh                              # once, downloads YAMNet (16 MB)
python trainer/ingest.py --pull --verify-alignment # pull + prove alignment
```

On the phone: start playback, open **ads-filter**, tap **Start recording**, grant the projection
prompt. A session opens when audio starts and closes 30 s after it stops. **Mark** drops a timestamped
marker. Volume can be at zero throughout.

## The Phase 0 gate

Everything depends on one unknown: **`AudioPlaybackCapture` is opt-out-able.** An app is capturable
only if it uses `USAGE_MEDIA` / `USAGE_GAME` / `USAGE_UNKNOWN` and has not disabled capture via
`android:allowAudioPlaybackCapture="false"` or `AudioAttributes.ALLOW_CAPTURE_BY_NONE`. Streaming
apps have every incentive to opt out. If Spotify and YouTube both refuse, scope shrinks to
podcast/radio apps — worth knowing on day one rather than month three.

### One-time device setup (Galaxy S23)

1. *Settings → About phone → Software information →* tap **Build number** ×7.
2. *Settings → Developer options →* enable **USB debugging** and **Wireless debugging**.
   (Wireless matters: Phase 1 has you carrying the phone around recording for days.)
3. Connect USB, unlock the phone, accept the **Allow USB debugging?** prompt (tick *Always allow*).
   Without that tap `adb devices` reports `unauthorized`.
4. Note the **Android version** — it determines the MediaProjection/FGS rules (issue I6).

### Step 1 — manifest probe (~2 min, no code runs on the phone)

```bash
tools/probe_capture_optout.sh                 # defaults to the usual suspects
tools/probe_capture_optout.sh com.some.app    # or specific packages
```

Reading the result:

- **`false`** — definitively not capturable. Rule that app out.
- **`true`** — explicitly opted in. Very likely fine.
- **`absent`** — the default, and **proves nothing**: an app can still refuse per-track at runtime
  via `ALLOW_CAPTURE_BY_NONE`, which never shows up in a manifest.

So this is a fast *negative* test only. Survivors go to step 2.

### Step 2 — capture spike (~30 min)

```bash
tools/spike.sh install     # build + install + launch
tools/spike.sh watch       # live log in another terminal
```

Then, per app under test:

1. Start audio playing **in the app under test** first.
2. In the spike app, type its name in the label box.
3. **Start capture** → grant the projection prompt → let it run ~20 s → **Stop**.
4. Read the `VERDICT` line.

```bash
tools/spike.sh results     # every verdict recorded so far
tools/spike.sh pull        # WAVs + verdicts into ./captures/
```

The verdict is driven by per-second RMS: a stream that never exceeds −60 dBFS while audio is
audibly playing means the app opted out. Listen to the pulled WAV to confirm before believing it.

Also record, while you're here: what happens on screen-off, on process restart, and after a reboot.
The consent token is single-use on Android 14+, so expect a re-consent tap each start.

**Gate:** at least the podcast/radio apps must be capturable.

## Layout

```
spike/                 Phase 0b throwaway spike (Kotlin, minSdk 29, targetSdk 35)
  app/src/main/java/com/adsfilter/spike/
    MainActivity.kt      label box, start/stop, live log
    CaptureService.kt    FGS(mediaProjection) -> AudioPlaybackCapture -> WAV + verdict
tools/
  probe_capture_optout.sh   Phase 0a manifest probe
  spike.sh                  build / install / watch / pull / results
captures/              pulled recordings (gitignored)
```

## Toolchain notes

Already present on this machine: Android Studio, SDK platform android-35, build-tools 35.0.0/36.0.0,
Gradle 8.12 (cached), `adb` 35.0.2. Two gotchas:

- Neither `adb` nor `gradle` is on `PATH`; the scripts locate them explicitly.
- **`apkanalyzer` is not installed** (cmdline-tools was never set up), so the manifest probe uses
  `aapt2` from build-tools instead.

Build directly if you prefer:

```bash
cd spike && "$HOME/.gradle/wrapper/dists/gradle-8.12-bin/"*/gradle-8.12/bin/gradle :app:assembleDebug
```
