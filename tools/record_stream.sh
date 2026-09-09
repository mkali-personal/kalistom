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
#   tools/record_stream.sh 8            # record overnight into captures/desktop
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
deadline=$(( $(date +%s) + total ))

echo "recording ${HOURS}h into $OUT"
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

  got=$(python -c "
import wave,sys
try:
    w=wave.open(r'$f','rb'); print(int(w.getnframes()/w.getframerate()))
except Exception: print(0)")

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
