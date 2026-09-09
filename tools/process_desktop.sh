#!/usr/bin/env bash
# Runs desktop-recorded WAVs through the phone's own YAMNet.
#
# The audio from the laptop was measured interchangeable with the phone's capture (r = 0.9986), but
# embeddings computed on the laptop would come from a different TFLite runtime. Processing them on
# the device removes that last difference between training data and production.
#
# The WAV never leaves the laptop's copy: the phone returns only .f16 and .jsonl, named after the
# input, and ingest pairs them with the local audio. That also means ingest's rms check doubles as
# proof the phone processed exactly the file we think it did.
#
#   tools/process_desktop.sh push [dir]   copy WAVs to the phone's inbox (default captures/desktop)
#   tools/process_desktop.sh pull [dir]   fetch .f16/.jsonl back beside those WAVs
#   tools/process_desktop.sh status       what is waiting and what is done
#
# Between push and pull, tap "Process inbox" in the app. It needs the model loaded, so start a
# recording once first if the app has just been launched.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe}"
PKG=com.adsfilter
REMOTE_IN="/sdcard/Android/data/$PKG/files/inbox"
REMOTE_OUT="/sdcard/Android/data/$PKG/files/processed"

need_device() {
  [ -n "$("$ADB" devices | sed -n '2p')" ] || { echo "No device attached." >&2; exit 1; }
}

case "${1:-status}" in
  push)
    need_device
    DIR="${2:-$ROOT/captures/desktop}"
    [ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }
    "$ADB" shell mkdir -p "$REMOTE_IN" >/dev/null 2>&1
    n=0
    for f in "$DIR"/*.wav; do
      [ -e "$f" ] || continue
      base=$(basename "$f")
      # skip anything already processed
      if [ -f "$DIR/${base%.wav}.f16" ]; then
        echo "skip (already processed): $base"
        continue
      fi
      echo "push $base ($(du -h "$f" | cut -f1))"
      "$ADB" push "$(cygpath -w "$f")" "$REMOTE_IN/" >/dev/null || { echo "  push failed" >&2; continue; }
      n=$(( n + 1 ))
    done
    echo
    echo "$n file(s) in the inbox. Now tap \"Process inbox\" in the app, then:"
    echo "  tools/process_desktop.sh pull $DIR"
    ;;

  pull)
    need_device
    DIR="${2:-$ROOT/captures/desktop}"
    mkdir -p "$DIR"
    "$ADB" pull "$REMOTE_OUT" "$(cygpath -w "$(dirname "$DIR")")/_processed_tmp" >/dev/null 2>&1
    tmp="$(dirname "$DIR")/_processed_tmp"
    if [ -d "$tmp" ]; then
      moved=0
      for f in "$tmp"/*; do
        [ -e "$f" ] || continue
        mv -f "$f" "$DIR/" && moved=$(( moved + 1 ))
      done
      rmdir "$tmp" 2>/dev/null
      echo "$moved file(s) -> $DIR"
    else
      echo "nothing to pull"
    fi
    echo
    echo "verify with: python trainer/ingest.py --dir $DIR --verify-alignment"
    ;;

  status)
    need_device
    echo "=== inbox (waiting) ==="
    "$ADB" shell "ls -1 $REMOTE_IN 2>/dev/null | head -20" | tr -d '\r' || echo "(empty)"
    echo "=== processed (ready to pull) ==="
    "$ADB" shell "ls -1 $REMOTE_OUT 2>/dev/null | head -20" | tr -d '\r' || echo "(empty)"
    ;;

  *)
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
