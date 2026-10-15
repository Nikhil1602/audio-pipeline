#!/usr/bin/env bash
# Linux capture on real hardware (M6): run from the repository root on a Linux desktop
# (an Ubuntu live USB is enough) with the speakers on and a microphone connected.
#
#   tools/linux/hw-check.sh [path/to/sdctl]
#
# sdctl: the CI artifact "sdctl-linux-x86_64" (no build needed), or target/release/sdctl.
# Needs the desktop's sound server (PipeWire or PulseAudio; nothing to install on Ubuntu) and
# network access once, to fetch the pinned models. Writes everything to target/hw-check/ and
# prints a summary to paste back.
set -uo pipefail

sdctl="${1:-target/release/sdctl}"
chmod +x "$sdctl" 2>/dev/null || true
out=target/hw-check
mkdir -p "$out"
log="$out/summary.txt"
: > "$log"
say() { echo "$@" | tee -a "$log"; }

say "== system"
say "$(uname -srm); $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
say "$(pactl info 2>/dev/null | grep -E 'Server Name|Server Version|Default Sink|Default Source')"

say "== models"
"$sdctl" models fetch >"$out/models.log" 2>&1 && say "ok" || say "FAILED (see $out/models.log)"

say "== devices"
"$sdctl" devices 2>&1 | tee -a "$log"

say "== 1. loopback: plays 8 s of quiet noise and checks the system-audio capture"
say "   (keep other audio stopped)"
"$sdctl" loopback-test 2>&1 | tee "$out/loopback.log" | grep -E "capture:|drift|verdict" | tee -a "$log"

say "== 2. microphone + system audio for 60 s with processing"
say "   Please talk for the first 30 s, then stay quiet. A test tone plays from 35 s."
rm -rf "$out/rec"
( sleep 35; "$sdctl" play "$out/tone.wav" >/dev/null 2>&1 ) &
python3 - "$out/tone.wav" <<'PY'
import math, struct, sys, wave
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
    w.writeframes(b"".join(struct.pack("<hh", v, v) for v in
        (int(3000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(48000 * 10))))
PY
"$sdctl" record --seconds 60 --process --wav --out "$out/rec" 2>&1 | tee "$out/record.log" \
  | grep -vE "^\[ *[0-9.]+s\]" | tail -12 | tee -a "$log"

say "== 3. device change: unplug/replug headphones or switch the output in Settings during"
say "   the next 40 s (optional; press Enter to start, or Ctrl+C to skip)"
read -r _ || true
rm -rf "$out/switch"
"$sdctl" record --seconds 40 --out "$out/switch" 2>&1 | tee "$out/switch.log" \
  | grep -E "Switching|Started|NoDevice|verdict|journal" | tee -a "$log"

say "== done: please send $log (and $out/*.log if something failed)"
