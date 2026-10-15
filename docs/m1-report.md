# M1 report: Windows capture

Date: 2026-10-13. Host: Windows 11 x64, i5-12450HX, 12 GB RAM, NVMe SSD. Audio: Realtek(R) Audio (built-in "Microphone Array", built-in "Speakers"). Rust 1.94.1.

## What was built

| Component | Where | Notes |
|---|---|---|
| Timeline aligner | `sd-core::timeline` | Turns device packet timestamps into a continuous timeline. Fills real gaps with flagged silence, synthesizes silence while loopback delivers nothing, trims overlaps, absorbs jitter and slow drift without spurious gaps. |
| Segmented drift estimator | `sd-core::drift` | Pooled within-segment regression with a standard error. Segments break at gaps and discontinuities. |
| WASAPI capture | `sd-capture` (Windows) | Event-driven shared mode with MMCSS "Pro Audio". Mic = capture endpoint; system = render endpoint with `LOOPBACK`. `AUTOCONVERTPCM` delivers a fixed format (48 kHz float, mono mic, stereo system) whatever the device. Timestamps from `qpcPosition`. Device events via `IMMNotificationClient`. |
| Device management | `sd-capture` | Enumeration with names, defaults and mix formats. Follows the OS default with a 200 ms settle. A removed or invalidated device triggers a switch; a missing preferred device falls back to the default and switches back when it returns. Exponential-backoff retry (100 ms … 5 s). The timeline keeps running with silence while no device exists. |
| Test playback | `sd-capture::play` | WASAPI render with device-clock timing (`IAudioClock`). Used to verify loopback now and to measure echo paths in M2. |
| Recording engine | `sd-engine::Recorder` | Capture → drain thread (levels, drift, timeline) → bounded 30 s queue → journal thread (write, fsync, disk-full retry). The drain thread never touches disk (fixes M0 finding 1). Also accepts caller-provided rings ("bring your own audio"). |
| CLI | `sdctl devices`, `record`, `loopback-test` | `record` prints live levels, drift and capture counters, and can force restarts (`--restart-every`). |

## Validation on real hardware

### Basic recording (60 s, default devices)

Continuous timelines, 0 dropped frames, 0 journal gaps.

### Restart / switch path (`record --seconds 40 --restart-every 5`)

| | mic | system |
|---|---|---|
| Forced restarts | 7 | 7 |
| Journal duration (40.24 s wall) | 39.98 s | 40.08 s |
| Gaps in journal / dropped frames | 0 / 0 | 0 / 0 |
| Audio replaced by silence per restart | ~0.5 s (WASAPI re-init + first-packet latency) | n/a (silent) |

### 30-minute soak (`record --seconds 1800`, default devices, normal laptop use)

| Metric | mic | system |
|---|---|---|
| Journal duration (1800.25 s wall) | 1799.98 s | 1800.09 s |
| Gaps in journal / dropped frames | 0 / 0 | 0 / 0 |
| Device glitches (WASAPI discontinuity flag) | 2 | 10 |
| Gap-filled silence | 0.116 s | 0.438 s |
| Real audio present | yes (peak −11.2 dBFS) | ~64 s (peak −8.4 dBFS) |

Process resources over the soak (sampled every 10 s):

| CPU | Working set | Private bytes | Handles |
|---|---|---|---|
| 35.3 s over 1783 s = **2.0 % of one core** (0.17 % of 12 threads) | **13.3–13.7 MB, flat** | 4.3–4.4 MB | 211–216, stable |

Journal size: 514 MB for 30 min (48 kHz mono + stereo, 16-bit), as predicted in M0.

### Loopback fidelity (`sdctl loopback-test`)

Plays stereo noise at −30 dBFS RMS on the default output while recording its loopback, then compares.

| Check | 8 s signal | 60 s signal |
|---|---|---|
| Lag constant across all 100 ms windows (no drop, duplication or drift) | yes (80 windows) | yes (600 windows) |
| Channel order | correct | correct |
| Gain | +1.00 dB | +1.00 dB |
| Full-band correlation (median) | 0.902 | 0.902 |
| Linear-fit residual above 1 kHz (128-tap NLMS) | −30.3 dB | −38.7 dB |
| Ring drops / journal gaps | 0 / 0 | 0 / 0 |

Offline spectral analysis (Welch coherence; script in the session scratchpad):

| Band | Gain | Coherence |
|---|---|---|
| 20–100 Hz | +0.9 dB | 0.78–0.80 |
| 100–300 Hz | +1.0 dB | 0.95 |
| 300–1000 Hz | +1.0 dB | 0.9991 |
| 1 kHz – 24 kHz | +1.00 dB | **0.9999** |

**Interpretation.** The capture path is sample-exact in time. Above 1 kHz the content equals the playback up to a flat +1 dB and a linear phase response. Below ~1 kHz the Realtek driver adds deterministic low-frequency energy (identical across runs), consistent with a bass/loudness enhancement applied *before* the loopback tap. For echo cancellation this is the correct reference, because it is what reaches the speaker. Full-band bit-exactness could not be demonstrated on this hardware; it needs "Audio enhancements" turned off for the endpoint.

The 8 s verdict sits right at its −30 dB threshold. The threshold was chosen after seeing this laptop's data, and any other sound playing during the test (e.g. a notification) can fail it. Of three 60 s runs, one failed for an undiagnosed reason (output not kept), one passed, and one's verdict was not captured.

### Clock drift

| Measurement | Result |
|---|---|
| Mic vs host clock, 30 min soak | +1.38 ppm. Other runs: +0.5, −0.48 ± 0.17, −8.6 (60 s window). Run-to-run spread exceeds the stated standard error. |
| Loopback vs host, 60 s continuous playback | +0.87 ppm, −1.27 ppm (two runs) |
| Loopback, sparse playback (4 × 8 s bursts in 150 s), single regression (before fix) | −118 ppm (wrong) |
| Same scenario, segmented regression (after fix) | −23.3 ppm, then −17.5 ± 6.9 ppm (still biased, ~2.5 σ) |

**Interpretation.** Both endpoints of this codec run within a few ppm of the host clock. Loopback timestamps shift when audio starts after silence, so drift measured from sparse loopback data is unreliable. The standard error assumes independent residuals and understates real uncertainty.

### Timestamps

Loopback packet timestamps (`qpcPosition`) place the audio **+83.2 ms later** than the render device clock (`IAudioClock`) says it was played. It ranged from 83.1 to 83.4 ms across 7 runs. The cause is not established; it may be how each API defines "position" (mix point vs. DAC). It does not affect mic/system alignment, which uses capture timestamps only, but M2 echo-path delay must be estimated from the signal, not assumed.

## Bluetooth headset (OnePlus Nord Buds 3r)

When connected, Windows exposes two endpoints: "Headphones" (A2DP, 48 kHz stereo output) and "Headset" (hands-free, **16 kHz mono** mic). Opening the headset mic switches the earbuds to the hands-free profile.

### Finding: timestamps from the hands-free driver are not reliable

The earbuds deliver exactly real-time audio (100 packets/s × 480 frames after conversion), but `qpcPosition` stamps wobble:
- **First recording:** a regular sawtooth, with eight packets stamped ~9 ms apart and then an ~18 ms resync. The driver also sets `DATA_DISCONTINUITY` at each resync (~11 per second) without losing audio.
- **Later recording:** about ±6 ms swings and bursts of packets 2 ms apart.

The M1 aligner originally placed packets by timestamp, so it read this jitter as gaps and overlaps:

| Aligner version | Mic timeline vs wall | Silence inserted | **Real audio trimmed** |
|---|---|---|---|
| timestamp re-anchor per packet (first M1 version) | 46.60 s / 45.04 s (+1.5 s) | 1.91 s | 0 |
| timestamp vs slew-limited baseline | 30.0 s / 30.2 s | 1.69 s per 30 s | **1.59 s per 30 s** |
| **device position (final)** | 30.00 s / 30.16 s | 0.10 s (start-up only) | **0** |

### Fix: device positions decide what was lost

WASAPI's per-packet device position counts real frames, but in the device's native rate even when `AUTOCONVERTPCM` delivers another rate. On the earbuds that's 160 per 480-frame packet, 16 kHz vs 48 kHz. Scaled by `stream_rate / device_rate` the position is exactly continuous. The aligner now:
- measures loss from position jumps (exact, immune to stamp jitter);
- uses timestamps only for the first packet of a stream, the first packet after synthesized silence, or when positions are implausible.

Validated on hardware:
- earbud mic 30 s: 0 trimmed, 0.10 s silence at start-up only;
- Realtek mic 30 s: 0 trimmed, 0 silence;
- switch Realtek → earbuds → Realtek in 45 s: 0 trimmed, silence only at the switch points (1.5 s for the Bluetooth profile change at 10 s, 0.2 s at 30 s), timeline 44.99 s for 45.03 s wall.

The drift estimator also no longer breaks segments on device `DISCONTINUITY` flags, only on inserted silence. Drift on the earbud mic remains imprecise (±20 ppm) because it is fitted from the jittery timestamps.

## Hot-plug (Bluetooth earbuds disconnected and reconnected during a 120 s recording)

Mic: Realtek (default). System audio: default output (the earbuds).

| Time | Event | System-audio capture |
|---|---|---|
| 14.6 s | earbuds disconnected | `DeviceRemoved` → **Speakers (Realtek)** immediately |
| 32.1 s | earbuds reconnected; Windows made them default | `DefaultDeviceChanged` → **earbuds** |
| 89.1 s | earbuds disconnected | `DeviceRemoved` → **Speakers (Realtek)** |

Both timelines were continuous, with 0 frames dropped and 0 trimmed.

### Finding: the *mic* loses audio when an *output* device changes, and its position counter hides it

At each of the three events the Realtek mic packet carried `DATA_DISCONTINUITY` and a timestamp jump of +25, +25 and +38 ms, while its device position stayed continuous. The audio engine dropped mic audio while reconfiguring; positions count only delivered frames. The position-only aligner therefore inserted nothing, and the mic timeline fell **102 ms** behind real time (and behind system audio).

**Fix.** Positions still decide by default, but a device-flagged packet whose timestamp jumps more than 15 ms past the baseline is treated as real loss, and the timestamp-measured gap is inserted. The 15 ms threshold sits above the Bluetooth hands-free driver's flagged-resync jitter (about ±8 ms, no loss).

**Validation by replay** (`sdctl debug replay`, which re-runs the aligner over recorded packet timestamps and flags):
- **Hot-plug recording:** gaps of 24.7 / 24.9 / 37.9 ms recovered (3 timestamp overrides). The timeline ends **−1.8 ms** from real time instead of −102 ms.
- **Earbud recording made with the position aligner (bt4):** 0 overrides, only the two real switch gaps, ends +1.9 ms from real time.
- **Earlier earbud recordings (bt1–bt3):** made with older aligners whose trims and inserts are baked into the journal, so their replays are not faithful and are not counted.

**Live re-test with the fix (150 s, earbuds connected at ~21 s and disconnected at ~79 s):**

| Time | Event | System audio | Mic (Realtek) |
|---|---|---|---|
| 20.9 s | earbuds connected; Windows made them default | `DefaultDeviceChanged` → earbuds | lost 23.8 + 19.2 ms, filled live |
| 78.6 s | earbuds disconnected | `DeviceRemoved` → Speakers (Realtek), within 0.1 s | lost 24.4 ms, filled live |

The mic journal was 149.98 s for 150.04 s wall (the normal start-up offset), so it stayed in step with real time. Mic drift held at −0.15 ± 0.23 ppm (the old build jumped to −1500 ppm). Dropped and trimmed were both 0 on both sources.

**Also validated:** a capture asking for a device that is missing (the earbuds after they were disconnected) fell back to the default mic, with a clean timeline.

## Automated tests

40 passing (`cargo test --workspace`), clippy clean with `-D warnings`. New in M1:
- aligner: jitter, 300 ppm drift over 10 min with no spurious gaps, glitch gap, loopback silence and resume, overlap trimming, silent-from-start, reset;
- segmented drift with bursty offsets, with the error covered by the reported standard error;
- engine: synthetic sources → journal with exact levels (−6.02 dBFS) and 100 ppm drift recovered within 1 ppm;
- capture backoff.

CI runners have no audio devices, so hardware tests are manual (commands above).

## Open items (carried into M2 or later)

1. **Hot-plug: done.** Removing, adding and default change were validated live, including mic loss during output reconfiguration. Untested: a USB device, and unplugging the device that is *being captured* as the mic.
2. **Drift correction itself is M2.** M1 measures drift; resampling to align mic and loopback belongs with echo cancellation. Rule for M2: correct only from long continuous runs, skip the first ~0.5 s after each gap, and require a small standard error.
3. **Journal size (~1 GB/h)**: compress (FLAC) and/or journal at 16 kHz. Decide by measurement in M2.
4. **Per-restart audio loss (~0.5 s mic)** is inherent to WASAPI re-initialisation on this device. A faster hand-over (open the new device before closing the old) is possible later.
5. **Mic level on this laptop is very low** (typical RMS −80 to −100 dBFS, peaks −11 to −72 dBFS). This is likely the Windows input level or the mic privacy/mute setting rather than the pipeline. Worth checking before M2 noise/AGC tuning.
6. **Exclusive-mode conflicts, Bluetooth hands-free profile switches, and per-process loopback** were not tested.
