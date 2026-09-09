#!/usr/bin/env bash
# Phase 0a: zero-code check of whether an app has opted out of AudioPlaybackCapture.
#
# Reads android:allowAudioPlaybackCapture from each installed app's manifest.
#
#   INTERPRETING THE RESULT
#     false   -> definitively NOT capturable. Rule the app out.
#     true    -> opted in explicitly. Very likely capturable.
#     absent  -> DEFAULT (capturable for targetSdk >= 29 media usage) BUT PROVES NOTHING:
#                the app can still refuse per-AudioTrack at runtime via ALLOW_CAPTURE_BY_NONE,
#                which never appears in a manifest. Only the Phase 0b spike settles it.
#
# Usage:  tools/probe_capture_optout.sh [package ...]
set -uo pipefail

ADB="${ADB:-$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe}"
AAPT2="${AAPT2:-$LOCALAPPDATA/Android/Sdk/build-tools/35.0.0/aapt2.exe}"
WORK="${TMPDIR:-/tmp}/capture-probe.$$"

DEFAULT_PKGS=(
  com.spotify.music
  com.google.android.youtube
  com.google.android.apps.youtube.music
  de.danoeh.antennapod
  au.com.shiftyjelly.pocketcasts
  com.google.android.apps.podcasts
)

PKGS=("$@")
[ ${#PKGS[@]} -eq 0 ] && PKGS=("${DEFAULT_PKGS[@]}")

[ -x "$ADB" ]   || { echo "adb not found at $ADB" >&2; exit 1; }
[ -x "$AAPT2" ] || { echo "aapt2 not found at $AAPT2" >&2; exit 1; }

if [ -z "$("$ADB" devices | sed -n '2p')" ]; then
  echo "No device. Enable USB debugging and accept the RSA prompt on the phone." >&2
  exit 1
fi

echo "device: $("$ADB" shell getprop ro.product.model | tr -d '\r') / Android $("$ADB" shell getprop ro.build.version.release | tr -d '\r') (API $("$ADB" shell getprop ro.build.version.sdk | tr -d '\r'))"
echo

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

printf '%-42s %-10s %-8s %s\n' PACKAGE FLAG TARGET NOTE
printf '%-42s %-10s %-8s %s\n' "------" "----" "------" "----"

for pkg in "${PKGS[@]}"; do
  path=$("$ADB" shell pm path "$pkg" 2>/dev/null | tr -d '\r' | grep -m1 'base.apk' | sed 's/^package://')
  if [ -z "$path" ]; then
    printf '%-42s %-10s %-8s %s\n' "$pkg" "-" "-" "not installed"
    continue
  fi

  local_apk="$WORK/$pkg.apk"
  if ! "$ADB" pull "$path" "$local_apk" >/dev/null 2>&1; then
    printf '%-42s %-10s %-8s %s\n' "$pkg" "?" "?" "pull failed"
    continue
  fi

  tree=$("$AAPT2" dump xmltree "$local_apk" --file AndroidManifest.xml 2>/dev/null)
  flag=$(printf '%s' "$tree" | grep -i 'allowAudioPlaybackCapture' | head -1 |
         sed -n 's/.*=\(.*\)$/\1/p' | tr -d ' ')
  target=$(printf '%s' "$tree" | grep -i 'targetSdkVersion' | head -1 |
           sed -n 's/.*=\(.*\)$/\1/p' | tr -d ' ')

  case "$flag" in
    *0xffffffff|*true|*"(type 0x12)0xffffffff") verdict="opted IN"; flag="true"  ;;
    *0x0|*false)                                verdict="OPTED OUT - rule it out"; flag="false" ;;
    "")                                         verdict="default; spike required"; flag="absent" ;;
    *)                                          verdict="unparsed: $flag" ;;
  esac

  printf '%-42s %-10s %-8s %s\n' "$pkg" "$flag" "${target:-?}" "$verdict"
done

echo
echo "'absent' does NOT mean capturable - run the Phase 0b spike app to confirm."
