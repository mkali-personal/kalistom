#!/usr/bin/env bash
# Phase 0b helper: build / install / watch / collect the capture spike.
#
#   tools/spike.sh build      compile the debug APK
#   tools/spike.sh install    build + install on the connected device
#   tools/spike.sh watch      tail the spike's log (verdict lines appear here)
#   tools/spike.sh pull       copy captured WAVs + verdicts to ./captures/
#   tools/spike.sh results    print every verdict recorded so far
set -uo pipefail

# Git Bash (MSYS) rewrites /sdcard/... into a Windows path before adb sees it.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe}"
GRADLE="${GRADLE:-$HOME/.gradle/wrapper/dists/gradle-8.12-bin/cetblhg4pflnnks72fxwobvgv/gradle-8.12/bin/gradle}"
export JAVA_HOME="${JAVA_HOME:-/c/Program Files/Android/Android Studio/jbr}"

PKG=com.adsfilter.spike
APK="$ROOT/spike/app/build/outputs/apk/debug/app-debug.apk"
REMOTE="/sdcard/Android/data/$PKG/files/spike"

die() { echo "$*" >&2; exit 1; }
need_device() {
  [ -n "$("$ADB" devices | sed -n '2p')" ] || die "No device attached."
}

case "${1:-}" in
  build)
    cd "$ROOT/spike" && "$GRADLE" :app:assembleDebug --no-daemon
    ;;
  install)
    cd "$ROOT/spike" && "$GRADLE" :app:assembleDebug --no-daemon || die "build failed"
    need_device
    "$ADB" install -r "$APK" || die "install failed"
    "$ADB" shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
    echo "installed and launched."
    ;;
  watch)
    need_device
    "$ADB" logcat -c
    echo "watching SpikeCapture (Ctrl+C to stop)..."
    "$ADB" logcat -s SpikeCapture:I
    ;;
  pull)
    need_device
    mkdir -p "$ROOT/captures"
    # remote path must stay POSIX; local path must be Windows-native for adb.exe
    "$ADB" pull "$REMOTE" "$(cygpath -w "$ROOT/captures")" && echo "-> $ROOT/captures"
    ;;
  results)
    need_device
    echo "=== verdicts on device ==="
    # shellcheck disable=SC2016
    "$ADB" shell 'for f in '"$REMOTE"'/*.verdict.txt; do [ -e "$f" ] && { echo "--- $f"; cat "$f"; }; done' 2>/dev/null \
      || echo "(none yet)"
    ;;
  *)
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
