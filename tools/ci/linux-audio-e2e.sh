#!/usr/bin/env bash
# Linux capture end to end on a real sound server with virtual devices (CI, or a Linux desktop
# you don't mind reconfiguring briefly).
#
#   tools/ci/linux-audio-e2e.sh pulseaudio   # starts a PulseAudio daemon
#   dbus-run-session -- tools/ci/linux-audio-e2e.sh pipewire   # starts PipeWire + pipewire-pulse
#   tools/ci/linux-audio-e2e.sh existing     # uses the server already running
#
# Creates the devices the tests expect (see crates/sd-capture/tests/linux_e2e.rs), runs the
# tests, and removes the devices again.
set -euo pipefail

server="${1:?usage: linux-audio-e2e.sh pulseaudio|pipewire|existing}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-$(id -u)}"
mkdir -p -m 0700 "$XDG_RUNTIME_DIR"

pids=()
cleanup() {
  for m in "${modules[@]:-}"; do [ -n "$m" ] && pactl unload-module "$m" 2>/dev/null || true; done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  if [ "$server" = pulseaudio ]; then pulseaudio --kill 2>/dev/null || true; fi
}
modules=()
trap cleanup EXIT

case "$server" in
  pulseaudio)
    pulseaudio --daemonize=yes --exit-idle-time=-1 --log-target=stderr
    ;;
  pipewire)
    # Their logs go to files: they warn a lot without a real-time kit, which is harmless.
    pipewire 2>/tmp/pipewire.log & pids+=($!)
    sleep 1
    wireplumber 2>/tmp/wireplumber.log & pids+=($!)
    pipewire-pulse 2>/tmp/pipewire-pulse.log & pids+=($!)
    ;;
  existing) ;;
  *) echo "unknown server: $server" >&2; exit 2 ;;
esac

for _ in $(seq 1 50); do
  pactl info >/dev/null 2>&1 && break
  sleep 0.2
done
pactl info | grep -E "Server Name|Server Version|Default Sample" || { echo "e2e: no sound server"; exit 1; }

# Report where the script stops (it runs with -e), so CI annotations show the reason.
trap 'echo "e2e: setup failed at line $LINENO: $BASH_COMMAND"' ERR

# Loads a module, retrying: right after start-up a server (PipeWire) may not be ready for
# modules yet, and a monitor appears only after its sink does.
load() {
  local out
  for _ in $(seq 1 25); do
    if out=$(pactl load-module "$@" 2>&1); then
      modules+=("$out")
      return 0
    fi
    sleep 0.4
  done
  echo "e2e: pactl load-module $* failed: $out"
  return 1
}

load module-null-sink sink_name=e2e_speaker rate=48000 channels=2 \
  sink_properties=device.description=e2e_speaker
load module-null-sink sink_name=e2e_mic_feed rate=48000 channels=1 \
  channel_map=mono sink_properties=device.description=e2e_mic_feed
load module-remap-source master=e2e_mic_feed.monitor source_name=e2e_mic \
  source_properties=device.description=e2e_mic
# PipeWire creates the nodes asynchronously.
for _ in $(seq 1 50); do
  pactl list short sources | grep -q e2e_mic && break
  sleep 0.2
done
pactl set-default-sink e2e_speaker
pactl set-default-source e2e_mic
echo "--- sinks"; pactl list short sinks
echo "--- sources"; pactl list short sources

SD_CAPTURE_E2E=1 SD_PULSE_DIAG=1 cargo test --release --locked -p sd-capture --test linux_e2e -- \
  --test-threads=1 --nocapture

# Full stack: `sdctl record` (capture -> rings -> journal, processing with AEC/VAD, device
# restarts) for SD_E2E_RECORD_SECONDS while noise plays into the speaker and the virtual mic.
record_s="${SD_E2E_RECORD_SECONDS:-0}"
if [ "$record_s" -gt 0 ]; then
  cargo build --release --locked -p sdctl
  sdctl=target/release/sdctl
  "$sdctl" models fetch
  python3 - <<'PY'
import random, struct, wave
for name, seed, ch in [("target/e2e_sys.wav", 1, 2), ("target/e2e_mic.wav", 2, 1)]:
    rnd = random.Random(seed)
    with wave.open(name, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(48000)
        frames = bytearray()
        for _ in range(48000 * 10):
            v = int(rnd.gauss(0, 0.03) * 32767)
            frames += struct.pack("<h", max(-32768, min(32767, v))) * ch
        w.writeframes(bytes(frames))
PY
  ( while true; do "$sdctl" play target/e2e_sys.wav --device e2e_speaker >/dev/null 2>&1; done ) &
  pids+=($!)
  ( while true; do "$sdctl" play target/e2e_mic.wav --device e2e_mic_feed >/dev/null 2>&1; done ) &
  pids+=($!)
  rm -rf target/e2e-rec
  set +e
  "$sdctl" record --seconds "$record_s" --process --restart-every 60 --out target/e2e-rec \
    > target/e2e-rec.log 2>&1
  status=$?
  set -e
  grep -vE "^\[ *[0-9.]+s\]" target/e2e-rec.log | tail -25 | sed 's/^/record: /'
  switches=$(grep -c "Switching" target/e2e-rec.log || true)
  resyncs=$(grep -o "reference_resyncs: [0-9]*" target/e2e-rec.log | grep -o "[0-9]*$" | tail -1)
  echo "record: switch events $switches, echo-reference resyncs ${resyncs:-?}"
  # Each restart may resync the echo reference; jittery timestamps would resync constantly.
  if [ -n "$resyncs" ] && [ "$resyncs" -gt $((10 + 5 * switches)) ]; then
    echo "e2e: too many echo-reference resyncs ($resyncs for $switches switch events)"
    status=1
  fi
  exit $status
fi
