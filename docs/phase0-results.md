# Phase 0 results — 2026-09-09

**Gate: PASSED.** Playback capture works on the target device for the primary target (Kan), and a
better-than-expected actuator is available. Phase 1 is unblocked.

## Device

| | |
|---|---|
| Model | Samsung SM-S911U1 (Galaxy S23, US Snapdragon) |
| Android | **16 (API 36)**, security patch 2026-07-05 |
| Bootloader | `flash.locked=1`, `verifiedbootstate=green`, `oem_unlock_supported` empty |
| Build type | `user` |

Android 16 is newer than the roadmap assumed (it was written against 14/15). The spike targets
SDK 35 and runs fine; no new MediaProjection restriction was hit.

## 0a — Manifest probe (`allowAudioPlaybackCapture`)

| Package | Flag | targetSdk | Meaning |
|---|---|---|---|
| `com.spotify.music` | **`true`** | 37 | explicitly opted IN |
| `com.applicaster.il.ch1` (Kan) | absent | 36 | default → confirmed capturable in 0b |
| `com.google.android.youtube` | absent | 37 | default → untested at runtime |
| `com.google.android.apps.youtube.music` | absent | 36 | default → untested at runtime |
| `com.apple.android.music` | absent | 35 | default → untested at runtime |

**Nothing is opted out.** Issue I1 — the biggest project risk — is largely retired. Spotify
explicitly setting `true` is the strongest possible manifest-level signal.

Not installed, so not probed: AntennaPod, Pocket Casts, Google Podcasts.

Still untested at runtime: YouTube, YouTube Music, Apple Music. `absent` is the default, not a
guarantee — an app can still refuse per-track via `ALLOW_CAPTURE_BY_NONE`.

## 0b — Runtime capture spike, Kan (`com.applicaster.il.ch1`)

```
VERDICT [kan]: AUDIO CAPTURED -> this app is capturable
duration=21s  loud_seconds=21/21  peak=-7.8 dBFS
```

Offline analysis of `cap_kan_20260909_221842.wav`:

| Property | Value | Reading |
|---|---|---|
| Format | 48 kHz / 16-bit / stereo, 21.75 s | as requested |
| RMS | −24.7 dBFS (both channels) | healthy programme level |
| Peak | −7.8 dBFS, **0 clipped samples** | clean, no limiter artefacts |
| DC offset | −0.00003 | none |
| L/R correlation | **1.0000** | dual mono |
| Digital silence | 0.47% of frames | normal |
| Energy < 3.4 kHz | 98.0% | speech-dominated |
| Roll-off | ~10.5 kHz | bandwidth-limited broadcast feed |

### Consequences for the build

1. **Downmix to mono is lossless** (correlation 1.0), so the recorder can halve its work immediately.
2. **Resampling 48 k → 16 k costs ~0.3% of energy** for this source. Resampler quality matters far
   less than the plan assumed — a decent polyphase filter is plenty, no need to agonise.
3. Speech-dominated and narrowband strengthens improvement **B** ("is a host talking?") over a
   generic ad/content classifier, and means YAMNet's AudioSet priors are being used well outside
   the music-heavy regime they're usually demoed on.

## Bonus — global output-mix actuator (Phase 4, pulled forward)

Can an **unprivileged** app attenuate the whole output mix via an `AudioEffect` on session 0?
**Yes, on Android 16.** All three attached, enabled, and stayed attached:

| Effect | Attenuation set | Accepted |
|---|---|---|
| `LoudnessEnhancer` | −4000 mB (−40 dB) | yes, gain read back un-clamped |
| `Equalizer` | 5 bands at −1500 mB floor | yes (−15 dB — a duck, not a mute) |
| `DynamicsProcessing` | −60 dB input gain | yes |

User confirmed the drop was **audible**. Only `MODIFY_AUDIO_SETTINGS` (a normal permission) is
needed.

This displaces `setStreamVolume` as the intended actuator (issue I4): no visible volume-slider
movement, no Bluetooth absolute-volume interaction, no dependency on the playing app exposing a
`MediaSession`, and it is app-agnostic. `LoudnessEnhancer` is the leading candidate — deepest
attenuation for the simplest API.

**Caveats to carry into Phase 4:**
- It attenuates *everything*, alarms and navigation prompts included. The state machine must be
  provably unable to get stuck attenuating; needs a watchdog and a release on process death.
- It is still a **mute, not look-ahead**. It does nothing for the 3–4 s leak in issue I2.
- Untested: behaviour across Bluetooth/headphone route changes, and during phone calls.

## Why full audio interception is not on the table

Asked whether a private app could intercept all speaker audio outright. No, and permissions are not
the limiting factor:

- `CAPTURE_AUDIO_OUTPUT` is `signature|privileged|role` on this device — it requires the platform
  signing key or `/system/priv-app` placement. It is not a runtime permission, so `pm grant` cannot
  grant it.
- Root would be the route, and this device's bootloader is permanently locked with no OEM-unlock
  option (US Snapdragon Samsung).
- Even with root, `AudioPlaybackCapture` is a **tap**, not an inline filter. There is no supported
  point in app → AudioFlinger → HAL where an app holds audio and decides whether to pass it.

The probe results make it moot: nothing is opted out.

## Open items

- Runtime-test YouTube / YouTube Music / Apple Music (manifest says default, unverified).
- MediaProjection lifecycle on Android 16 not yet characterised: screen-off, process restart, reboot.
  The consent token is single-use, so a re-consent tap per start is expected.
- Confirm whether a session-0 effect survives a Bluetooth route change.
