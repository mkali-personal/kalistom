#!/usr/bin/env bash
# Records the Kan Reshet Bet live stream to 16 kHz mono WAV segments, unattended.
#
# This is the same broadcast the phone app plays (the site labels this player
# "כאן חדשות ברשת ב'", which matches what the phone showed while recording), but it reaches us by a
# different route: ffmpeg decodes the 202 kbps AAC HLS stream directly instead of going through the
# phone's decoder and audio pipeline. Whether that difference matters to YAMNet is measured by
# tools/skew_test.sh - do not assume desktop and phone audio are interchangeable until it has run.
#
# Segments are written at SEGMENT_MIN intervals so a crash costs one segment, not the night.
#
# Usage:
#   tools/record_stream.sh [hours] [out_dir]
#   tools/record_stream.sh 8                 # record overnight into captures/desktop
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOURS="${1:-8}"
OUT="${2:-$ROOT/captures/desktop}"
SEGMENT_MIN="${SEGMENT_MIN:-30}"

# Resolved from the player element on https://www.kan.org.il/radio/ (data-player-hls-src).
STREAM="${KAN_STREAM:-https://kancdn.medonecdn.net/livehls/oil/kancdn-live/live/radio/kan_reshet_bet/live.livx/playlist.m3u8?renditions}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
mkdir -p "$OUT"

secs=$(python -c "print(int(float('$HOURS')*3600))")
echo "recording ${HOURS}h (${secs}s) -> $OUT"
echo "segments of ${SEGMENT_MIN} min, 16 kHz mono"
echo "stream: $STREAM"
echo

# -reconnect keeps a dropped stream from ending the run; a gap inside one segment is acceptable
# because each segment is verified independently before labelling.
ffmpeg -hide_banner -loglevel warning \
  -reconnect 1 -reconnect_streamed 1 -reconnect_on_network_error 1 -reconnect_delay_max 30 \
  -t "$secs" -i "$STREAM" \
  -ac 1 -ar 16000 -c:a pcm_s16le \
  -f segment -segment_time $(( SEGMENT_MIN * 60 )) -segment_format wav -strftime 1 \
  "$OUT/desk_%Y%m%d_%H%M%S.wav"

echo
echo "done. segments:"
ls -la "$OUT" | tail -20
