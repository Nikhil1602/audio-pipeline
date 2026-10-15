#!/bin/bash
# SonicDiar: Mac check (real hardware).
#
# A guided test of microphone and system-audio capture, device changes and (optionally)
# transcription on a real Mac. It takes about 10 minutes, asks before every step that plays
# sound or records, and writes the results to one zip on the Desktop to send back.
#
# Nothing is installed: everything stays in this folder (models are downloaded into ./models)
# and in the results folder on the Desktop. Delete both afterwards if you like.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here" || exit 1
# A browser download marks files as quarantined, which stops macOS from running them.
xattr -dr com.apple.quarantine . 2>/dev/null || true
chmod +x ./sdctl 2>/dev/null || true

stamp=$(date +%Y%m%d-%H%M)
name="sonicdiar-check-$stamp"
out="$HOME/Desktop/$name"
mkdir -p "$out"
log="$out/summary.txt"
tty=/dev/tty

say() { echo "$*" | tee -a "$log"; }
ask() { local a; read -r -p "$1 " a < "$tty"; echo "$a"; }
step() { echo; echo "------------------------------------------------------------"; say "== $*"; }

say "SonicDiar Mac check, $(date)"
say "$(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion)), $(uname -m), $(sysctl -n hw.model)"
if [ "$(uname -m)" != arm64 ]; then
  say "(Intel Mac: everything works; transcription runs slower than on Apple silicon.)"
fi

step "1/6 Audio devices"
./sdctl devices 2>&1 | tee -a "$log"
# Device names and how they connect (built-in, USB, Bluetooth), nothing else.
system_profiler SPAudioDataType 2>/dev/null \
  | grep -E "^        [^ ].*:$|Default (Input|Output|System Output) Device|Transport|Input Source|Output Source" \
  >> "$log"

step "2/6 Download the speech-activity model (about 46 MB, once)"
if ./sdctl models fetch > "$out/models.log" 2>&1; then say "ok"; else say "FAILED (see models.log)"; fi

step "3/6 System audio"
echo "This plays 8 seconds of quiet noise through your speakers and checks that the Mac's"
echo "system audio is captured. macOS will ask to allow Terminal to record system audio:"
echo "please click Allow. (If you clicked Don't Allow earlier: System Settings > Privacy &"
echo "Security > Screen & System Audio Recording > turn on Terminal, then run this again.)"
ask "Set the volume to a normal level, stop any other audio, then press Enter."
./sdctl loopback-test > "$out/loopback.log" 2>&1
grep -E "capture:|drift|verdict|ermission|Retrying|retrying|did not start" "$out/loopback.log" | tee -a "$log"

step "4/6 Microphone and system audio together (60 seconds)"
echo "When the recording starts:"
echo "  1. talk normally for about 20 seconds,"
echo "  2. then play a YouTube video with someone talking, through the speakers, for 20 seconds,"
echo "  3. then stay quiet until it ends."
echo "macOS may ask to allow Terminal to use the microphone: please click Allow."
ask "Press Enter to start recording."
echo "Recording for 60 seconds ..."
./sdctl record --seconds 60 --process --out "$out/rec" > "$out/record.log" 2>&1
grep -vE "^\[ *[0-9.]+s\]" "$out/record.log" | tail -15 | tee -a "$log"
grep -E "PermissionNeeded|Switching|Retrying|NoDevice|Stalled" "$out/record.log" | head -20 >> "$log"

step "5/6 Device changes (40 seconds, optional)"
echo "During the next 40 seconds, two or three times: connect or disconnect headphones or"
echo "AirPods, or switch the output and input in Control Center > Sound."
a=$(ask "Press Enter to start, or type s and Enter to skip.")
if [ "$a" != s ]; then
  echo "Recording for 40 seconds ..."
  ./sdctl record --seconds 40 --out "$out/switch" > "$out/switch.log" 2>&1
  grep -E "Switching|Started|NoDevice|PermissionNeeded|verdict|journal" "$out/switch.log" | tee -a "$log"
else
  say "skipped"
fi

step "6/6 Live transcription (optional; downloads a 490 MB speech model once)"
a=$(ask "Test it? Type y and Enter for yes, or just Enter to skip.")
if [ "$a" = y ]; then
  if ./sdctl models fetch --only whisper-small > "$out/models-asr.log" 2>&1; then
    echo "Talk for 30 seconds, in any language, while it records."
    ask "Press Enter to start."
    echo "Recording for 30 seconds ..."
    ./sdctl record --seconds 30 --process --asr-model models/ggml-small.bin \
      --out "$out/asr" > "$out/asr.log" 2>&1
    grep -E "^ *> |ASR:" "$out/asr.log" | tail -20 | tee -a "$log"
  else
    say "model download FAILED (see models-asr.log)"
  fi
else
  say "skipped"
fi

echo
a=$(ask "Include the recordings themselves (your voice) in the results? Type y for yes, Enter for no.")
if [ "$a" != y ]; then
  rm -rf "$out/rec" "$out/switch" "$out/asr"
  rm -f "$out"/*.wav
  say "(recordings not included)"
fi
(cd "$HOME/Desktop" && zip -qr "$name.zip" "$name")
echo
echo "============================================================"
echo "Done, thank you! Please send this file back:"
echo "   ~/Desktop/$name.zip"
echo "============================================================"
open -R "$HOME/Desktop/$name.zip" 2>/dev/null || true
