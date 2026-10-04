# Kalistom

A private, sideloaded Android app that detects commercials in the phone's **playback** audio and
drops the volume until they end. Personal use only — never published to Play.

Design review, issues and full roadmap: see the plan at
`~/.claude/plans/in-android-app-for-starry-lampson.md`, derived from
`Android App for Commercial Detection.pdf`.

## Everyday commands

Everything below is typed in a terminal opened in the project folder. In Command Prompt that is:

```
cd C:\Users\michaeka\git-projects\ads-filter
```

The commands are the same in Command Prompt and in Git Bash. The only exception is the `tools/*.sh`
scripts, which need Git Bash. Each script also prints its own help when run with `--help`, for
example `python trainer/update.py --help`.

### Label recordings

```
python trainer/label_editor.py
```

This opens the label editor in your browser at http://localhost:8765/. Pick a recording from the
list, correct the ad spans, and save. It writes your labels next to the audio as
`<recording>.truth.txt`, and training always prefers those over machine drafts. The editor keeps
running until you press Ctrl+C in the terminal. If it says the port is in use, an editor is already
running; open the address above, or start a second one with `--port 9000`.

### After recording: pull, retrain, and put the new head on the phone

Plug the phone in, stop any recording in the app, and run:

```
python trainer/update.py --ship
```

This runs the whole loop:

1. It copies new sessions from the phone and deletes them there once they have arrived intact.
2. It lists recordings that still have no labels. Those are left out until you label them, so a
   good habit is to label first and then run this.
3. It trains twice on this machine. The first model is trained without the newest two hours or so
   and tested on them, which tells you how the head does on audio it has not heard. The second is
   trained on everything and is the one that goes to the phone.
4. It prints the test result next to every earlier run, so you can see whether things improved.
5. It rebuilds the app with the new head and installs it. The thresholds on the phone stay as they
   were.

Without `--ship`, steps 1 to 4 run and nothing on the phone changes; the last line tells you how to
install the new head later. Other useful variations:

| Command | What it does |
|---|---|
| `python trainer/update.py --no-pull` | Use what is already on the computer; do not touch the phone |
| `python trainer/update.py --no-train` | Only print the report from the last run again |
| `python trainer/update.py --full` | The careful test: every broadcast day held out in turn, on a free Kaggle GPU. Use it before changing thresholds or trying a different kind of head |
| `python trainer/update.py --full --backend local` | The same, on this machine, which is much slower |

The quick test in step 3 covers only a couple of hours, so its numbers jump around from run to run.
It is good for catching a model that has broken; it is not good enough for choosing thresholds.

### Change the thresholds

The head gives every 0.48 s of audio a score between 0 and 1. The phone starts muting when the
score reaches the **on** threshold (now 0.999) and stops muting when it falls below the **off**
threshold (now 0.2). To change one, for example the on threshold to 0.99:

```
python trainer/ship.py --on 0.99
```

This edits `app/src/main/assets/head_weights.json`, rebuilds the app and installs it. It refuses
to install while the phone is recording, because installing ends the session. Use `--off 0.3` for
the other threshold, or give both. Every change saves the previous file under
`captures/heads/shipped/`, and the command prints how to undo it, which looks like this:

```
python trainer/ship.py --restore captures/heads/shipped/head_weights_<date>_<time>.json
```

What a change buys, measured on held-out days for the 07:00-11:00 programme (seconds per hour of
listening, with off at 0.2):

| on | Ad seconds muted | Programme seconds muted by mistake | Delay before muting starts |
|---|---|---|---|
| 0.9999 | 392 | 14 | 6.2 s |
| **0.999 (now)** | **448** | **23** | **2.9 s** |
| 0.995 | 472 | 30 | 1.9 s |
| 0.99 | 478 | 35 | 1.4 s |
| 0.95 | 489 | 47 | 0.5 s |

To see the full trade as a picture, draw the ROC curves; `--show` opens the plot in a window:

```
python trainer/roc_threshold.py captures/heads/oof_day_mlp16.npz --dir captures/stitched --dir captures/sessions --hours 07-11 --show
```

After changing the thresholds, `python trainer/update.py --no-train --full` reprints the last
careful test judged at the new values.

### Other commands worth keeping nearby

| Command | What it does |
|---|---|
| `python trainer/ship.py` | Rebuild the app and install it, changing nothing |
| `python trainer/ship.py --head captures/heads/head_mlp16.json` | Install a head trained earlier, keeping the phone's thresholds |
| `python trainer/ingest.py --pull` | Only copy sessions off the phone, and check them |
| `python trainer/ingest.py --pull --keep-on-phone` | Copy without deleting them from the phone |
| `python trainer/repair_wav.py captures/sessions/*.wav` | Fix recordings whose length shows as 0, from a session the app did not close. Healthy files are skipped |
| `%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe devices` | Check that the computer sees the phone (Command Prompt). It should list one device |
| `tools/record_stream.sh --at 07:00 4` | Record the radio stream on this computer from 07:00 to 11:00 (Git Bash). See "Setting up on another computer" |

Where things are kept:

| Path | What is there |
|---|---|
| `captures/sessions/` | Recordings pulled from the phone, and their labels |
| `captures/stitched/` | Recordings of the stream made on a computer, and their labels |
| `captures/heads/history.jsonl` | One line per training run, the source of the history table |
| `captures/heads/head_mlp16.json` | The newest trained head, before it is shipped |
| `captures/heads/shipped/` | Every head file the phone had before a change, for undoing |
| `app/src/main/assets/head_weights.json` | What the phone runs: the head and its thresholds |

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
tools/record_stream.sh 8                       # 8 hours starting now
tools/record_stream.sh --at 07:00 4            # sleep until the next 07:00, then record 07:00-11:00
```

`--at` waits for the next occurrence of that time - today if it is still ahead, tomorrow
otherwise - so the morning programme is captured whether or not anyone is awake to start it. The
end time is absolute: `--at 07:00 4` means 07:00 to 11:00, so a machine that suspends past the
start records the remainder rather than sliding the whole window later.

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

## Training the classifier

The everyday loop is `python trainer/update.py`, described under "Everyday commands" above. This
section explains what it does underneath.

**Folds are broadcast days, not recordings.** The same advertisement airs many times in one
morning, so with one recording held out, its ads were usually in training an hour earlier, from
another recording. Holding out a whole day (04:00 to 04:00) removes those repeats. `update.py
--full` holds out every day in turn; with `--fold-cache` a new recording retrains only its own day.
`train.py --fold-by recording` restores the old split. Numbers from different splits are not
comparable, so compare heads only on the same split.

**The quick split** (`train.py --test-newest-hours 2`, the default in `update.py`) holds out only
the newest days, trains once on the rest and once on everything. It keeps `--on`/`--off` exactly
as given rather than sweeping them, because a sweep on two hours of audio would chase noise.

The steps below still work on their own. There are two ways to train, and they produce the same kind of result file.

**Locally, with nothing but numpy and scipy.** `trainer/train.py` is the reference: it builds
the pool from every labelled recording, trains one model per held-out recording, and writes the
out-of-fold predictions.

```bash
python trainer/train.py --dir captures/stitched --dir captures/sessions --head mlp \
    --hidden 16 --save-oof captures/heads/oof_mlp16.npz --fold-cache captures/heads/folds_mlp16
```

It is slow on a laptop: 40 recordings take hours per head. `--fold-cache` saves each fold as it
finishes, so a run that is interrupted resumes instead of starting over.

**On a free Kaggle GPU, driven from this machine.** `trainer/torch_train.py` fits the same heads
with PyTorch, and `trainer/kaggle_run.py` runs it on a Kaggle T4. A fold that takes minutes locally
takes seconds there.

```bash
python trainer/kaggle_run.py --export --run "k40_lr head=lr l2=1e-3" \
                                      --run "k40_mlp16 head=mlp hidden=16 wd=1e-2"
```

The driver works in four steps:

1. It exports the pool with `train.py --export`, using the same labels, joins and folds.
2. It uploads the pool as a private Kaggle dataset, and only when the pool has changed. The upload
   is embeddings and labels, not audio.
3. It pushes a private script and waits for it to finish.
4. It copies the results into `captures/heads/`, where `smooth.py` and the label editor pick them
   up.

Kaggle needs a one-time setup:

1. Create an account and verify a phone number, which Kaggle requires before it allows GPUs.
2. Install the command-line tool with `pip install kaggle`.
3. Create an API token under **Settings** and save it to `~/.kaggle/access_token`.

`torch_train.py` also runs locally on the exported pool wherever PyTorch is installed, using the
CPU if there is no GPU:

```bash
python trainer/torch_train.py captures/kaggle/dataset/pool.npz --head mlp --out oof.npz
```

The two implementations agree. On the same folds, the logistic regression matches `train.py`
exactly (correlation 1.000). The network agrees as closely as two trainings of one network from
different random starts can (correlation 0.96), because the two share no random-number generator.

## Recording a session

```bash
tools/fetch_model.sh                              # once, downloads YAMNet (16 MB)
python trainer/ingest.py --pull --verify-alignment # pull + prove alignment
```

Pulling moves the sessions rather than copying them. Live inference never reads old sessions, so
once a session is on the computer it is deleted from the phone. A session is deleted only after
every one of its files hashes the same on both sides, and the session still being recorded is
copied but left in place until a later pull. Add `--keep-on-phone` to copy without deleting.

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
