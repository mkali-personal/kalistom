#!/usr/bin/env bash
# Downloads the YAMNet TFLite model into app/src/main/assets/.
# Kept out of git (16 MB binary); run once after cloning.
#
# This is the TF Hub build, which exposes THREE outputs:
#   Identity   [frames, 521]   class scores
#   Identity_1 [frames, 1024]  embeddings   <- what the model trains on
#   Identity_2 [frames, 64]    log-mel spectrogram
# The MediaPipe "audio_classifier" build exposes only scores, so don't swap it in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/app/src/main/assets/yamnet.tflite"
URL="https://www.kaggle.com/api/v1/models/google/yamnet/tfLite/tflite/1/download"

if [ -f "$DEST" ]; then
  echo "already present: $DEST ($(du -h "$DEST" | cut -f1))"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "downloading YAMNet..."
curl -sL --max-time 300 -o "$tmp/dl" "$URL"
tar xzf "$tmp/dl" -C "$tmp"
found="$(find "$tmp" -name '*.tflite' | head -1)"
[ -n "$found" ] || { echo "no .tflite in download" >&2; exit 1; }
mv "$found" "$DEST"
echo "-> $DEST ($(du -h "$DEST" | cut -f1))"
