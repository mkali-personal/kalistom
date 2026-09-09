#!/usr/bin/env bash
# Does the phone's media volume affect what AudioPlaybackCapture sees?
#
# Steps the media volume down in stages while the recorder runs, so the per-frame rms trace in
# the session's .jsonl can be read back afterwards. If the tap sits BEFORE the stream-volume
# stage the trace stays flat; if AFTER, it steps down with each stage and collapses at 0.
#
# Why it matters: if capture is volume-independent, training data can be collected with the phone
# silent - which makes long passive recording actually livable.
#
# Run this WHILE a recording is in progress:
#     tools/volume_test.sh [seconds_per_stage]
set -uo pipefail
export MSYS_NO_PATHCONV=1

ADB="${ADB:-$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe}"
STAGE="${1:-15}"
STREAM=3          # STREAM_MUSIC

vol_get() { "$ADB" shell cmd media_session volume --stream $STREAM --get 2>&1 | tr -d '\r' \
            | sed -n 's/.*volume is \([0-9]*\) in range \[0\.\.\([0-9]*\)\].*/\1 \2/p'; }
vol_set() { "$ADB" shell cmd media_session volume --stream $STREAM --set "$1" >/dev/null 2>&1; }

read -r START MAX <<<"$(vol_get)"
[ -n "${START:-}" ] || { echo "could not read volume" >&2; exit 1; }
echo "starting volume $START/$MAX; ${STAGE}s per stage"
echo "restore with: $ADB shell cmd media_session volume --stream 3 --set $START"
echo

trap 'echo; echo "restoring volume to $START"; vol_set "$START"' EXIT

for v in "$MAX" $(( MAX * 2 / 3 )) $(( MAX / 3 )) 1 0; do
  vol_set "$v"
  ts=$(date +%H:%M:%S)
  echo "$ts  volume -> $v/$MAX   (hold ${STAGE}s)"
  sleep "$STAGE"
done

echo
echo "done - stop the recording, then:"
echo "  python trainer/ingest.py --pull"
echo "  python trainer/volume_report.py"
