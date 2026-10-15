# M7 report: macOS capture

Date: 2026-10-15. Plan: [roadmap.md](roadmap.md) §M7.

## 1. Summary

- **macOS capture works** on Core Audio:
  - **microphone:** an IOProc on the input device;
  - **system audio:** a **Core Audio process tap** (macOS 14.2+), a private global stereo tap of every process's output, read through a private aggregate device clocked by the output device.
- **Same contract as Windows and Linux:** events, default-device following (200 ms settle), fallback when a chosen device disappears, stall watchdog, gap-fill budget and a continuous timeline. The packet-to-timeline code is shared with the other backends (`feed.rs`).
- **Validated in CI on real macOS** (GitHub's Apple-silicon runners, macOS 14.8 and 15.7) with BlackHole virtual devices, on every push:
  - the played signal comes back intact through the mic path (every 100 ms window correlates at 1.0000, timestamps +0.0 ms) and through the system-audio tap (0.998, timestamps +13 to +34 ms);
  - 30 s with no loss and no drift (±0 ppm);
  - a default input change and a default output change (the tap is rebuilt on the new device) keep the timeline continuous;
  - the full stack (`sdctl record --process`: capture, journal, AEC, VAD) ran for an hour with a device restart every minute on both versions: nothing dropped, 0 journal gaps (§4.2).
- **Permissions are handled, not assumed.** A denied microphone, and a system-audio tap waiting on its permission prompt, are reported as `CaptureEvent::PermissionNeeded`; capture keeps the timeline running and can always be stopped.
- **First real Mac (§8):** a MacBook Air on macOS 27 recorded mic and system audio for 60 s with a continuous timeline and nothing dropped. It also found two bugs in the check kit and one in the capture code (fixed). Bluetooth, AirPods switching and USB headsets are still untested on real hardware.

## 2. Design

### 2.1 What the roadmap planned, and what was built

| Planned | Built | Why |
|---|---|---|
| AUHAL for the mic | IOProc on the device (`AudioDeviceCreateIOProcID`) | AUHAL does not resample input; with an IOProc the backend receives the device's own format either way and converts once, the same way for mic and tap |
| Core Audio tap + aggregate device | as planned (global stereo tap, private aggregate) | |
| ScreenCaptureKit fallback (macOS 13) | **not built** | §5 |
| TCC checks with typed errors | microphone status from AVFoundation; tap permission inferred | macOS has no public query for System Audio Recording |

### 2.2 How it works (`crates/sd-capture/src/coreaudio.rs`)

- **Real-time thread.** The IOProc only copies samples and the input timestamp into lock-free rings (`rtrb`). Everything else happens on the capture thread.
- **Format.** Core Audio delivers each device's own rate and channel count (BlackHole 16ch: 16 channels; many Macs run at 44.1 kHz). `convert.rs` maps channels (mono = average of all; stereo = the first two) and resamples with WebRTC's sinc resampler (the one the processing stage already uses, BSD-3-Clause), carrying host times and a continuous position. Unit-tested on every OS: 44.1 kHz stereo → 48 kHz mono keeps a tone (correlation > 0.95) and 10 ms chunk timing.
- **Timestamps.** `mHostTime` of the IOProc's input time, converted to ns on the same clock as `HostTime::now` (`CLOCK_UPTIME_RAW`). Device positions come from `mSampleTime`; a jump (the HAL skipped a cycle) is flagged as a discontinuity so the timeline aligner measures the gap.
- **Device changes.** Property listeners on the default input/output device, the device list and the current device's "is alive" drive default following, removal (`DeviceRemoved`) and the return to a chosen device.
- **The process tap** is created with `CATapDescription initStereoGlobalTapButExcludeProcesses:@[]` (private, unmuted) and an aggregate device whose main sub-device is the output device, with drift compensation on the tap. `AudioHardwareCreateProcessTap` is looked up at run time (`dlsym`), so the same binary still runs, with mic capture only, on macOS before 14.2, where system audio reports that it needs 14.2.
  - The tap is **global**: it hears every process's output, whichever device it plays on. Choosing an output device selects the clock the tap runs on, not what it hears.
- **Permissions.**
  - **Microphone:** AVFoundation's authorization status is read before opening. Denied or restricted → `PermissionNeeded { confirmed: true }`, and capture keeps looking every 2 s. Not determined → the request is made (the user sees the prompt); audio is silent until they answer.
  - **System Audio Recording:** macOS blocks `AudioHardwareCreateProcessTap` until the user answers the prompt (found in CI, where nobody can answer it). The tap is therefore created on a helper thread; after 2 s of waiting the capture reports `PermissionNeeded { confirmed: false }`, keeps the timeline running, and can be stopped at any time. A tap delivering only exact zeros for 10 s is reported the same way (with no permission, macOS delivers silence).
- **Playback** (`play`, for test signals): an IOProc on the output device, filled from memory; it counts output cycles the device skipped (`PlaybackReport::underruns`).

## 3. CI on macOS (`tools/ci/mac-audio-e2e.sh`, job `mac capture e2e`)

GitHub's macOS runners are Apple-silicon virtual machines with one virtual sound device. The job:
1. installs BlackHole 2ch and 16ch (virtual loopback devices, MIT) and switchaudio-osx, and restarts `coreaudiod`;
2. grants Microphone and System Audio Recording (`kTCCServiceMicrophone`, `kTCCServiceAudioCapture`) by writing the TCC databases for every process the test could be attributed to (the runner, its ancestors, the test binary) and restarting `tccd`. Hosted runners allow this; a real Mac does not, and the user grants the permissions instead;
3. runs `crates/sd-capture/tests/mac_e2e.rs` and then the full-stack recording.

The test player detects cycles the output device skipped. On the macOS 15 VM, one run had the whole audio system stall for one cycle, which the capture flagged as a device glitch and the player as a skip. The played signal itself had a gap, so such runs are retried (at most three times) instead of failing.

## 4. Results

### 4.1 Capture tests (CI, BlackHole)

| Test | macOS 14.8 | macOS 15.7 |
|---|---|---|
| Mic (BlackHole loopback): worst 100 ms window, timestamps vs audio | 1.0000, +0.0 ms | 1.0000, +0.0 ms |
| System audio (tap): played signal captured | yes (peak 0.192) | yes (peak 0.192) |
| 30 s mic: worst window, drift, timeline vs wall | 1.0000, +0 ppm, 31.04 / 31.01 s | 1.0000, −0 ppm, 31.10 / 31.08 s |
| Default input change: followed, timeline vs wall | yes, 1.88 / 1.88 s | yes, 1.89 / 1.86 s |
| Default output change: tap rebuilt, audio after the switch, timeline vs wall | yes, 0.998, +12.7 ms, 4.74 / 4.74 s | yes, 0.998, +34.0 ms, 4.68 / 4.70 s |

Notes:
- **Tap correlation is 0.998, not 1.0000:** the tap is a mixdown made by the HAL (stereo, with its own gain staging) rather than the sample-exact loopback of the BlackHole input, which is expected.
- **Tap start-up:** the first tap of a run delivered audio about 0.4 s after the capture started; rebuilding it on another device took about 1.4 s (filled with silence, so the timeline stays continuous).
- **Tap timestamps** are 13–34 ms later than the audio; the mic path is exact. The processing stage measures and follows the echo delay, so this offset does not need to be exact.

### 4.2 Full stack

`sdctl record --process --restart-every 60` with the tap on BlackHole 2ch (default output) and the mic on BlackHole 16ch, while noise plays into both (`tools/ci/mac-audio-e2e.sh`, `SD_E2E_RECORD_SECONDS`). Every push runs 2 minutes; a commit message with `[soak]` runs one hour. **One hour** (CI run 38678415947):

| | macOS 14.8 | macOS 15.7 |
|---|---|---|
| Verdict | PASS (continuous timeline, nothing dropped) | PASS |
| Journal (system / mic) | 3600.54 s / 3600.53 s, 0 gaps | 3600.20 s / 3600.15 s, 0 gaps |
| Dropped (ring + queue) | 0 | 0 |
| Device glitches (system / mic), flagged and filled | 12 / 12 | 28 / 33 |
| Restarts (one per minute; the tap is rebuilt each time) | 59 / 59 | 59 / 59 |
| Silence filled, whole hour (system / mic) | 8.8 s / 3.7 s | 4.8 s / 3.2 s |
| Mic vs system drift | 0.00 ppm | 0.00 ppm |
| Processing: dropped blocks, timeouts | 0, 0 | 0, 0 |
| Echo-reference resyncs (118 switch events) | 0 | 15 |

- **Glitches** are cycles the virtual machine's audio system skipped (the HAL's sample time jumped); the capture flagged each one and the timeline aligner filled the gap. They are a property of the CI VMs (the test player saw the same stalls), to be compared with a real Mac.
- **Silence filled** is mostly the tap's start-up after each of the 59 restarts (about 0.1 s each).
- **Drift 0.00 ppm:** BlackHole and the tap run on the same clock here; a real Mac's built-in mic and output devices may differ by a few ppm, which the processing stage follows.

## 5. Not done in M7 (open items)

1. **A real Mac.** Built-in mic and speakers, a USB or Bluetooth headset, AirPods switching, sleep/wake, and the permission prompts as a user sees them. Until then macOS capture is "validated in CI on virtual devices".
2. **ScreenCaptureKit fallback** for macOS 13 – 14.1 (system audio there currently reports that it needs 14.2). It needs Screen Recording permission and an SCStream; worth adding only if macOS 13 support is required.
3. **Per-device system audio.** The tap hears all processes on every device; `initExcludingProcesses:andDeviceUID:withStream:` could limit it to one device if that is ever wanted.
4. **Aggregate inputs.** If the output device also has inputs (a USB headset), the aggregate lists them before the tap, and the backend skips that many buffers. In CI the output device (BlackHole) also has an input, which carries the same played signal through BlackHole's loopback. The captured audio matched the tap (0.998, a HAL mixdown) rather than the loopback input (sample-exact 1.0000), so the skip looks right. A USB headset on a real Mac will confirm it.
5. **Voice processing** (`kAUVoiceIOProperty`, Apple's own echo cancellation) is not used; the pipeline runs its own AEC.

## 6. Testing on a real Mac: the Mac check kit

For testers who are not developers, `tools/mac-check` is published as the release [`mac-check`](https://github.com/Nikhil1602/sonicdiar/releases/tag/mac-check) (workflow `mac-check-kit`, triggered by a `mac-check-v*` tag). A tester pastes one line into Terminal:

```bash
curl -fsSL https://github.com/Nikhil1602/sonicdiar/releases/download/mac-check/run.sh | bash
```

It downloads a universal `sdctl` (Apple silicon and Intel; no Rust, Xcode or GitHub account needed) and guides them through: devices, system audio (`loopback-test`), 60 s of mic + system audio with processing, optional device changes, optional transcription. The results end up in one zip on their Desktop; recordings are included only if they agree.

**Intel Macs:** ONNX Runtime stopped shipping Intel macOS builds after 1.23.2, so Intel Macs use 1.23.2 (pinned in `models/manifest.json`) while the other platforms keep 1.31; the pipeline requests ONNX Runtime's API 23, which both provide. The kit's Intel build is checked in CI under Rosetta (devices, runtime download, VAD).

## 7. How to run

Unit tests (any OS): `cargo test -p sd-capture`.

On a Mac with BlackHole 2ch and 16ch installed (changes the default devices while it runs):
```bash
tools/ci/mac-audio-e2e.sh existing
```

Recording with transcription on a Mac works as on Windows (`sdctl record --process ...`). The first run asks for Microphone and System Audio Recording permission (for the terminal app that runs it).

## 8. First real Mac: a tester's MacBook Air (2026-10-15)

**Machine:** MacBook Air (Mac17,3, Apple silicon), macOS 27.0.1 (26A434), built-in microphone (48 kHz mono) and speakers (48 kHz stereo). The tester ran kit `mac-check-v2` remotely and sent back the zip, without recordings.

**Results, step by step:**

| Step | Result | What the logs show |
|---|---|---|
| 1 Devices | ok | Mic and speakers listed with their formats and UIDs. |
| 2 Runtime download | ok | ONNX Runtime 1.31 (arm64) and Silero VAD fetched, SHA-256 verified. |
| 3 System audio (`loopback-test`) | FAIL, **a test bug** | See "Test bug 1". |
| 4 Mic + system, 60 s, processing | CHECK, **really a pass** | See "Step 4 in numbers" and "Test bug 2". |
| 5 Device changes, 40 s | PASS, **with a capture bug** | See "The capture bug". |
| 6 Transcription | skipped by the tester | |

**Step 4 in numbers:**
- Both journals continuous: 0 gaps, 0 dropped, 0 device glitches, 0 restarts.
- The tester's YouTube speech was captured as system audio (RMS about −14 dBFS) and segmented by the VAD.
- The echo canceller switched itself on 8.5 s in (echo correlation 0.86): the speakers' sound reached the mic.
- The mic and the system tap ran on one clock: +0.00 ± 0.00 ppm between them, both +0.35 ppm against the host clock.

**Test bug 1: the system-audio test played too early.**
- `loopback-test` started capture, waited 600 ms, then played its 8 s noise probe.
- On a first run, opening the tap waits for the user to answer the "System Audio Recording" prompt. Here the first captured block came 10.0 s after the probe began (a lag of −480,869 frames at 48 kHz, predicted from the timestamps). The probe was over before capture started, so there was nothing to correlate.
- **Fix:** `loopback-test` now waits up to 120 s for the system stream's `Started` event, printing the permission wait, before it plays.

**Test bug 2: ordinary events counted as problems.**
- `sdctl record` counted `SpeechStart` and `SpeechDiscarded` (normal VAD events) as problems, which turned the verdict into CHECK.
- **Fix:** they are printed as ordinary events.

**The capture bug: idle system audio was taken for a stall.**
- **What happened:** while nothing played, the tap's IOProc was not called at all: 0 packets in 26 s. CI never exercised this case: noise always played into the tap there.
- **The effect:** the 2 s stall watchdog took the silence for a hang and reopened the tap every 2 s, 13 times. The timeline stayed continuous (26.4 s of silence filled, PASS), but each reopen was reported as a `Recovery` switch.
- **Fix:** for a tap, a quiet IOProc counts as a stall only while some client runs the output device (`kAudioDevicePropertyDeviceIsRunningSomewhere`). The same condition guards the "device never called back" check. This matches Windows, where a loopback stall needs the endpoint's peak meter to show audio.
- **Not verified on this Mac yet;** the next kit run will show it.

**Not explained yet:**
- **A one-off gap-fill on the mic in step 4.** It filled 1.96 s (94,064 frames), all within the first second; there was none afterwards. In step 5, the same mic needed only 0.05 s.
- **No device changes in step 5.** The log shows no device switches, so device switching on a real Mac is still unverified.
