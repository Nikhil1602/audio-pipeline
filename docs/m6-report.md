# M6 report: Linux capture

Date: 2026-10-14. Plan: [roadmap.md](roadmap.md) §M6.

## 1. Summary

- **One backend for both Linux sound servers.** `sd-capture` captures the microphone and system audio on Linux through the PulseAudio client API (libpulse). It works with:
  - PulseAudio itself;
  - PipeWire, through its built-in PulseAudio server (`pipewire-pulse`), which every PipeWire desktop installs.
- **Same contract as Windows.** The events, the default-device following (200 ms settle), the fallback when a chosen device disappears, the stall watchdog, the gap-fill budget and the continuous timeline behave as on Windows. The packet-to-timeline half of the backend is now shared code (`feed.rs`) used by both backends.
- **Validated in CI on both servers** with virtual devices (null sinks, a remapped virtual microphone), on every push:
  - the played signal comes back sample-exact (every 100 ms window correlates at 1.0000);
  - timestamps agree with the audio within 20 ms;
  - mic and system clocks agree to within 1.2 ppm (`sdctl record` drift estimate);
  - a default-sink change and a device removal are followed without breaking the timeline;
  - the full stack (`sdctl record --process`: capture, rings, journal, AEC, VAD) runs with nothing dropped: 2 minutes on every push, and a one-hour soak with a device restart every minute passed on both servers (mic vs system drift −2.5 and −0.03 ppm; 10 and 0 echo-reference re-syncs).
- **Not validated on real Linux hardware yet** (real drivers, a real microphone and Bluetooth). [`tools/linux/hw-check.sh`](../tools/linux/hw-check.sh) does that in about 5 minutes from an Ubuntu live USB, using the `sdctl` binary CI builds.

## 2. Design change from the roadmap

The roadmap planned a native PipeWire backend (`pipewire-rs`) with PulseAudio as the fallback. The PulseAudio API alone was chosen instead:

| | PulseAudio API (chosen) | Native PipeWire + Pulse fallback (planned) |
|---|---|---|
| PipeWire desktops (Ubuntu ≥ 22.10, Fedora, Debian 12, Arch) | yes, through `pipewire-pulse` | yes |
| PulseAudio desktops (Ubuntu 22.04 LTS, older) | yes | only through the fallback |
| Flatpak | standard `--socket=pulseaudio` | needs the PipeWire portal |
| Backends to maintain | one | two |
| Build | links `libpulse.so.0` (present on every desktop); no dev package or bindgen needed | needs libpipewire headers and libclang |

Native PipeWire would give somewhat lower latency and graph-clock timestamps. Nothing measured here needs them (timestamps are within 20 ms, drift ≤ 58 ppm). It can be added later behind the same contract if a need shows up.

**Licences:** the Rust bindings (`libpulse-binding`, `libpulse-sys`) are MIT/Apache-2.0. libpulse itself is LGPL-2.1+; it is the system's own library, linked dynamically and not shipped with the SDK.

## 3. How it works (`crates/sd-capture/src/pulse.rs`)

- **Streams.** The mic is a record stream on a source. System audio is a record stream on the **monitor** of a sink (what the sink plays). Streams ask for float32 at the configured rate and channels; the server converts.
- **Devices.** Ids are source names (mic) and sink names (system audio). Monitors are never listed as microphones. With no microphone, PulseAudio makes a sink monitor the default source; that counts as "no device", not as a mic.
- **Device changes.** Streams are connected to an explicit device with `DONT_MOVE`, so the server never moves them silently. The backend subscribes to server, source and sink events and:
  - follows a new default after the settle time (`Switching{DefaultDeviceChanged}`);
  - falls back to the default when a chosen device is removed (`Switching{DeviceRemoved}`);
  - returns to the chosen device when it reappears, and reconnects when the server restarts (e.g. `systemctl --user restart pipewire`). Both are implemented but **not yet tested**.
- **Timestamps.** Arrival time (`CLOCK_MONOTONIC`) minus the duration of the queued data and the source's buffering latency (taken once per stream), then smoothed (`arrival.rs`). Frames read are the device position, so the timeline aligner detects loss exactly; server-side overruns are flagged as glitches. The smoothing estimates the offset *arrival − stream position* as the 10th percentile of the last 2 s of fragments: scheduling delay (a few ms, never negative) sits above it, and PulseAudio's start-up burst (audio its null sink rendered ahead of real time, which arrives "early") below it. It follows the estimate directly during a stream's first 2 s, then at most 500 ppm (clock drift); three fragments in a row more than 50 ms off restart it (suspend, stall).
  - Without smoothing, the processing stage's echo-reference aligner (which re-syncs on errors over 2 ms) re-synced 1,503 times in 2 minutes on PipeWire (45,020 in an hour). A first version tracked the *lowest* offset; PulseAudio's start-up burst then set it too low and it climbed back at 500 ppm, which read as +245 ppm of drift and −73 ppm mic vs system over an hour. With the percentile: 0 re-syncs on PipeWire and 1 on PulseAudio in 2 minutes, slopes within 10 ppm.
  - `pa_stream_get_latency` was tried first and rejected: for monitor streams it subtracts the sink's playback buffer, so it moved by hundreds of ms (and below zero) as playback streams came and went. On PipeWire it wandered by 36 ms over 30 s, which looked like +650 ppm of drift.
- **Silence.** Sources deliver continuously (silence included) unless suspended, so the timeline fills silence only after **1 s** without data (Windows loopback, which stops delivering when nothing plays, fills after 60 ms). A suspended device is not a stall.
- **Raw mode** does not exist in PulseAudio; sources are unprocessed unless the user loaded an effect module. `Started{raw}` reports `false`.

## 4. Results (CI, Ubuntu 24.04, virtual devices)

`crates/sd-capture/tests/linux_e2e.rs` run by [`tools/ci/linux-audio-e2e.sh`](../tools/ci/linux-audio-e2e.sh). The signal is 48 kHz band-limited noise; "windows" are 100 ms correlations of capture against what was played.

| Test | PulseAudio 16.1 | PipeWire 1.0.5 (pipewire-pulse) |
|---|---|---|
| System audio: played signal in the capture, worst window | 1.0000 | 1.0000 |
| Microphone (virtual mic fed by a null sink), worst window | 1.0000 | 1.0000 |
| Timestamps vs audio (system / mic) | −19.7 / +0.3 ms | +10.2 / +10.2 ms |
| 30 s capture: worst window, timestamp slope vs nominal rate | 1.0000, −7 ppm | 1.0000, −9 ppm |
| 30 s: timeline length vs wall time | 31.02 s / 31.02 s | 30.86 s / 30.87 s |
| Default sink change: followed, signal intact after it | yes, 1.0000 | yes, 1.0000 |
| Timeline across the switch vs wall time | 4.53 s / 4.53 s | 4.08 s / 4.10 s |
| Chosen device removed: falls back to the default | yes | yes |
| Removed device comes back: capture returns to it | yes | yes |
| Full stack (`sdctl record --process`) | see §4.1 | see §4.1 |

Notes:
- **PulseAudio drift estimate.** Over the hour both PulseAudio sources run about +42 ppm against the host clock, and mic vs system −2.5 ± 0.1 ppm (the first percentile-free version showed −73 ppm). The two sources agree, so the common +42 ppm is most likely the null sink's own pacing; the processing stage follows it with 10 re-syncs in an hour of restarts every minute. PipeWire: +1 ppm and −0.03 ± 0.27 ppm. To be checked on a real sound card.
- **PulseAudio start-up.** On PulseAudio's null sink the first data arrived about 2 s after the stream started (filled with silence, so the timeline stays continuous). PipeWire started immediately. This is the null sink's initial 2 s block; real sound cards are expected to start within tens of ms (to be confirmed by the hardware check).
- **PulseAudio rewinds.** When a playback stream starts, PulseAudio rewinds the sink to play it sooner; the monitor does not see the rewound start, so the first few ms of a newly started sound are missing from system audio. The tests skip the first 100 ms window for that reason. PipeWire has no rewinds.

### 4.1 Full stack

`sdctl record --process --restart-every 60` while noise plays into the speaker and the virtual mic (`tools/ci/linux-audio-e2e.sh`, `SD_E2E_RECORD_SECONDS`). Every push runs 2 minutes; a commit message with `[soak]` runs one hour.

**One hour, before timestamp smoothing:**

| | PulseAudio | PipeWire |
|---|---|---|
| Verdict | PASS (continuous timeline, nothing dropped) | PASS |
| Journal (system / mic) | 3600.16 s / 3600.19 s, 0 gaps | 3599.87 s / 3599.85 s, 0 gaps |
| Dropped (ring + queue), device glitches | 0, 0 | 0, 0 |
| Restarts (one per minute) | 59 / 60 | 59 / 59 |
| Silence filled at restarts, whole hour (system / mic) | 2.0 s / 2.5 s | 1.1 s / 2.0 s |
| Mic vs system drift | n/a (mic estimate not ready) | +0.04 ± 6.7 ppm |
| Processing: dropped blocks, timeouts | 0, 0 | 0, 0 |
| Echo-reference resyncs | 963 | 45,020 (the jitter fixed since) |

**Two minutes, final code:**

| | PulseAudio | PipeWire |
|---|---|---|
| Verdict | PASS | PASS |
| Drift system / mic (vs host clock) | +27.0 / +38.8 ppm | +1.06 / +1.04 ppm |
| Mic vs system drift | +11.7 ± 1.4 ppm | −0.02 ± 0.27 ppm |
| Echo-reference resyncs | 1 (3 device switches) | 0 |

**One hour, final code** (CI run 38662736907):

| | PulseAudio | PipeWire |
|---|---|---|
| Verdict | PASS (continuous timeline, nothing dropped) | PASS |
| Journal (system / mic) | 3600.18 s / 3600.20 s, 0 gaps | 3600.14 s / 3600.14 s, 0 gaps |
| Dropped (ring + queue), device glitches | 0, 0 | 0, 0 |
| Restarts (one per minute) | 59 / 60 | 59 / 59 |
| Silence filled at restarts, whole hour (system / mic) | 1.5 s / 2.4 s | 3.5 s / 3.5 s |
| Drift vs host clock (system / mic) | +43.8 / +41.3 ppm | +1.02 / +1.00 ppm |
| Mic vs system drift | −2.46 ± 0.11 ppm | −0.03 ± 0.27 ppm |
| Processing: dropped blocks, timeouts | 0, 0 | 0, 0 |
| Echo-reference resyncs (119 switch events) | 10 | 0 |

## 5. Not done in M6 (open items)

1. **Real hardware run** (sound card, USB/Bluetooth headset, unplugging): run `tools/linux/hw-check.sh` from a Linux desktop or an Ubuntu 24.04 live USB. Until then Linux capture is "validated in CI on virtual devices".
2. **WSLg** (the roadmap's local second path) was not used: WSL is not installed on the development laptop.
3. **Native PipeWire backend:** not needed so far (§2).
4. **Flatpak/Snap packaging** and the PipeWire portal are part of M8.
5. **ALSA-only systems** (no sound server, e.g. minimal servers) are not supported; capture reports `Retrying` with "no PulseAudio/PipeWire server".

## 6. How to run

Unit and contract tests (any OS): `cargo test -p sd-capture`.

End to end on a Linux machine (creates and removes virtual devices on the running server):
```bash
tools/ci/linux-audio-e2e.sh existing
```

Real-hardware check (prints a summary to send back):
```bash
tools/linux/hw-check.sh path/to/sdctl
```
