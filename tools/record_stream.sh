#!/usr/bin/env bash
# Records the Kan Reshet Bet live stream to 16 kHz mono WAV segments, unattended.
#
# This is the same broadcast the phone app plays - the site labels this player
# "כאן חדשות ברשת ב'", matching what the phone showed while recording - but it reaches us by a
# different route, since ffmpeg decodes the AAC HLS stream directly rather than going through the
# phone's decoder and Android's audio pipeline. Whether that difference matters to YAMNet is what
# trainer/skew_test.py measures. Do not treat desktop and phone audio as interchangeable until it
# has been run.
#
# WHY THE SUPERVISOR LOOP: ffmpeg intermittently treats this live playlist as finished and exits
# cleanly mid-run - one observed run stopped after 44 s of a requested 300 s and still reported
# success. The -reconnect flags do not cover that case, because nothing errored. So each segment is
# a separate ffmpeg invocation and the loop simply starts the next one, which caps the damage from
# any single drop at the tail of one segment.
#
# Usage:
#   tools/record_stream.sh [hours] [out_dir]
#   tools/record_stream.sh 8                    # record 8 h starting now
#   tools/record_stream.sh --at 07:00 4         # sleep until the next 07:00, then record 4 h
#
# --at exists so the morning programme is captured whether or not anyone is awake to start it.
# The end time is absolute: ask for 07:00 and four hours and you get 07:00-11:00, so a machine
# that suspends past the start records the remainder rather than sliding the window to 09:00.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --at) AT="$2"; shift 2 ;;
    --at=*) AT="${1#--at=}"; shift ;;
    *) break ;;
  esac
done
HOURS="${1:-8}"
OUT="${2:-$ROOT/captures/desktop}"
SEGMENT_MIN="${SEGMENT_MIN:-30}"

# Resolved from data-player-hls-src on https://www.kan.org.il/radio/ (this is the CDN's own
# redirect target; using it directly avoids re-resolving the redirect on every restart).
STREAM="${KAN_STREAM:-https://r.il.cdn-redge.media/livehls/oil/kancdn-live/live/radio/kan_reshet_bet/live.livx/playlist.m3u8?renditions}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
mkdir -p "$OUT"

total=$(python -c "print(int(float('$HOURS')*3600))")
seg=$(( SEGMENT_MIN * 60 ))

if [ -n "$AT" ]; then
  # The next occurrence of HH:MM, today if it is still ahead of us and tomorrow otherwise.
  start=$(python -c "
import datetime, sys
hh, mm = (sys.argv[1].split(':') + ['0'])[:2]
now = datetime.datetime.now()
t = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
if t <= now:
    t += datetime.timedelta(days=1)
print(int(t.timestamp()))" "$AT")
  deadline=$(( start + total ))
  echo "waiting until $(date -d "@$start" '+%a %H:%M' 2>/dev/null || echo "$AT") to record ${HOURS}h into $OUT"
  echo "the machine must stay awake until then - check that sleep and hibernate are off"
  echo
  while :; do
    now=$(date +%s)
    left=$(( start - now ))
    [ "$left" -le 0 ] && break
    # Print a countdown once a minute so it is obvious this is alive and not wedged, and sleep in
    # short steps so a laptop that suspends and resumes notices quickly rather than an hour late.
    if [ $(( left % 60 )) -lt 20 ] || [ "$left" -lt 60 ]; then
      printf "  %02d:%02d:%02d until %s   " $(( left/3600 )) $(( left%3600/60 )) $(( left%60 )) "$AT"
    fi
    sleep $(( left < 15 ? left : 15 ))
  done
  echo
  # The deadline is absolute, not "now plus HOURS". If the machine slept through the start time we
  # record whatever is left of the window rather than sliding the whole thing later - the point of
  # asking for 07:00 is to capture the 07:00 programme, not to capture some h hours.
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    echo "the $AT window has already passed; nothing to record"
    exit 0
  fi
  [ $(( now - start )) -gt 60 ] &&     echo "starting $(( (now - start) / 60 )) min late - recording the rest of the window"
else
  deadline=$(( $(date +%s) + total ))
fi

echo "recording until $(date -d "@$deadline" '+%H:%M' 2>/dev/null || echo "+${HOURS}h") into $OUT"
echo "segments of ${SEGMENT_MIN} min, 16 kHz mono, restarting on early exit"
echo

short=0
while :; do
  now=$(date +%s)
  left=$(( deadline - now ))
  [ "$left" -le 10 ] && break
  want=$(( left < seg ? left : seg ))

  f="$OUT/desk_$(date +%Y%m%d_%H%M%S).wav"
  ffmpeg -hide_banner -loglevel error -y \
    -http_persistent 0 -m3u8_hold_counters 1000 \
    -reconnect 1 -reconnect_streamed 1 -reconnect_on_network_error 1 -reconnect_delay_max 30 \
    -t "$want" -i "$STREAM" \
    -ac 1 -ar 16000 -c:a pcm_s16le "$f" 2>/dev/null

  if [ ! -s "$f" ]; then
    echo "$(date +%H:%M:%S)  no data - stream unreachable, retrying in 15s"
    rm -f "$f"
    sleep 15
    continue
  fi

  # cygpath first: the path here is an MSYS /c/... one, which Windows Python cannot open, so this
  # check silently reported 0 s for every segment and the log claimed a short segment every time.
  # ffmpeg was never affected - Git Bash converts a bare argument, but not one inside a quoted
  # string.
  got=$(python -c "
import wave,sys
try:
    w=wave.open(sys.argv[1],'rb'); print(int(w.getnframes()/w.getframerate()))
except Exception: print(0)" "$(cygpath -w "$f" 2>/dev/null || echo "$f")")

  if [ "$got" -lt $(( want * 9 / 10 )) ]; then
    short=$(( short + 1 ))
    echo "$(date +%H:%M:%S)  short segment: ${got}s of ${want}s (restarting; $short so far)"
    sleep 3
  else
    echo "$(date +%H:%M:%S)  segment ok: ${got}s -> $(basename "$f")"
  fi
done

echo
echo "finished. $short short segment(s)."
ls -la "$OUT" | tail -20
echo
echo "next: python trainer/skew_test.py <phone.wav> <desktop.wav>   (once, to validate)"
