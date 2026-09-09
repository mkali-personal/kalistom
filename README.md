# ads-filter

A private, sideloaded Android app that detects commercials in the phone's **playback** audio and
drops the volume until they end. Personal use only — never published to Play.

Design review, issues and full roadmap: see the plan at
`~/.claude/plans/in-android-app-for-starry-lampson.md`, derived from
`Android App for Commercial Detection.pdf`.

## Where the project stands

See [`docs/phase0-results.md`](docs/phase0-results.md) for the Phase 0 findings.

| Phase | What | Status |
|---|---|---|
| 0a | Toolchain + zero-code manifest probe | **PASSED** - nothing opted out; Spotify opts in |
| 0b | Capture feasibility spike | **PASSED** - Kan captured cleanly on Android 16 |
| 1 | Session recorder (`.opus` + `.f16` + `.jsonl`) | not started |
| 2 | Audacity labelling + offline eval harness | not started |
| 3 | Model + HMM smoothing | not started |
| 4 | Actuator (duck / fade / skip) | not started |
| 5 | Fingerprints + MediaSession metadata | not started |
| 6 | Hardening | not started |

**The Phase 0 gate has passed** (2026-09-09, Galaxy S23 / Android 16). Phase 1 is unblocked.

A bonus finding from the same session: an unprivileged app *can* attenuate the global output mix
via an `AudioEffect` on session 0, which displaces `setStreamVolume` as the planned actuator.
Details and caveats in the results doc.

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
