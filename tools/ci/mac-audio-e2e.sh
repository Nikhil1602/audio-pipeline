#!/usr/bin/env bash
# macOS capture end to end with virtual devices (CI runners, or a Mac you don't mind
# reconfiguring briefly). Installs BlackHole 2ch/16ch (virtual loopback devices, MIT) and
# switchaudio-osx, tries to grant the microphone and system-audio permissions to the CI
# runner, and runs crates/sd-capture/tests/mac_e2e.rs.
#
#   tools/ci/mac-audio-e2e.sh            # CI: installs and configures
#   tools/ci/mac-audio-e2e.sh existing   # a Mac that already has BlackHole 2ch and 16ch
set -uo pipefail

mode="${1:-ci}"
echo "e2e: $(sw_vers -productName) $(sw_vers -productVersion) ($(uname -m))"

if [ "$mode" = ci ]; then
  brew install --quiet blackhole-2ch blackhole-16ch switchaudio-osx >/dev/null 2>&1 \
    || echo "e2e: brew install failed"
  # The HAL loads new drivers when coreaudiod restarts.
  sudo killall coreaudiod 2>/dev/null || true
  for _ in $(seq 1 30); do
    SwitchAudioSource -a -t output 2>/dev/null | grep -q "BlackHole 2ch" && break
    sleep 1
  done
  # Best effort: grant Microphone and System Audio Recording to every process TCC might
  # attribute the tests to: the runner's provisioner, each ancestor of this script, and the test
  # binary. The user database is writable on hosted runners; the system one needs SIP off, so
  # failures are only reported. tccd is restarted to read the changes.
  test_bin=$(cargo test --release --locked -p sd-capture --test mac_e2e --no-run \
    --message-format=json 2>/dev/null | python3 -c 'import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if m.get("executable"): print(m["executable"])' | tail -1)
  clients=("/usr/local/opt/runner/provisioner/provisioner" "$test_bin")
  pid=$$
  while [ -n "$pid" ] && [ "$pid" -gt 1 ]; do
    exe=$(ps -o comm= -p "$pid" 2>/dev/null)
    [ -n "$exe" ] && clients+=("$exe")
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  done
  now=$(date +%s)
  granted=0
  for db in "$HOME/Library/Application Support/com.apple.TCC/TCC.db" \
            "/Library/Application Support/com.apple.TCC/TCC.db"; do
    for service in kTCCServiceMicrophone kTCCServiceAudioCapture; do
      for client in "${clients[@]}"; do
        [ -z "$client" ] && continue
        sudo sqlite3 "$db" "INSERT OR REPLACE INTO access VALUES('$service', '$client', \
          1,2,4,1,NULL,NULL,0,'UNUSED',NULL,0,$now,NULL,NULL,'UNUSED',$now);" 2>/dev/null \
          && granted=$((granted + 1))
      done
    done
  done
  echo "e2e: granted $granted permission rows for ${#clients[@]} clients: ${clients[*]}"
  sudo killall tccd 2>/dev/null || true
fi

echo "--- outputs"; SwitchAudioSource -a -t output || true
echo "--- inputs"; SwitchAudioSource -a -t input || true

SD_CAPTURE_E2E=1 SD_COREAUDIO_DIAG=1 cargo test --release --locked -p sd-capture --test mac_e2e -- \
  --test-threads=1 --nocapture || exit 1

# Full stack: `sdctl record` (capture -> rings -> journal, processing with AEC/VAD, device
# restarts) for SD_E2E_RECORD_SECONDS. System audio is the tap (default output: BlackHole 2ch);
# the mic is BlackHole 16ch, fed with a different noise.
record_s="${SD_E2E_RECORD_SECONDS:-0}"
status=0
if [ "$record_s" -gt 0 ]; then
  SwitchAudioSource -t output -s "BlackHole 2ch" >/dev/null
  SwitchAudioSource -t input -s "BlackHole 16ch" >/dev/null
  cargo build --release --locked -p sdctl || exit 1
  sdctl=target/release/sdctl
  "$sdctl" models fetch >/dev/null || echo "e2e: models fetch failed"
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
  ( while true; do "$sdctl" play target/e2e_sys.wav --device BlackHole2ch_UID >/dev/null 2>&1; done ) &
  p1=$!
  ( while true; do "$sdctl" play target/e2e_mic.wav --device BlackHole16ch_UID >/dev/null 2>&1; done ) &
  p2=$!
  rm -rf target/e2e-rec
  "$sdctl" record --seconds "$record_s" --process --restart-every 60 --out target/e2e-rec \
    > target/e2e-rec.log 2>&1
  status=$?
  kill "$p1" "$p2" 2>/dev/null
  grep -vE "^\[ *[0-9.]+s\]" target/e2e-rec.log | tail -25 | sed 's/^/record: /'
  switches=$(grep -c "Switching" target/e2e-rec.log || true)
  resyncs=$(grep -o "reference_resyncs: [0-9]*" target/e2e-rec.log | grep -o "[0-9]*$" | tail -1)
  echo "record: switch events $switches, echo-reference resyncs ${resyncs:-?}"
  if [ -n "$resyncs" ] && [ "$resyncs" -gt $((10 + 5 * switches)) ]; then
    echo "e2e: too many echo-reference resyncs ($resyncs for $switches switch events)"
    status=1
  fi
fi
exit $status
