# M2 report: audio processing (in progress)

Date: 2026-10-13. Same host as M1. This report covers echo cancellation, voice activity detection, ONNX Runtime loading, the real-time processing stage with drift tracking, AGC and journal compression. Noise suppression is available but off by default; its effect on recognition will be measured by WER in M3.

## Echo cancellation engine choice: sonora vs the original WebRTC C++

**Candidates:**
- [`sonora`](https://crates.io/crates/sonora) 0.2.0: pure-Rust port of the WebRTC audio processing module (M145), BSD-3.
- `webrtc-audio-processing` 2.1: the original C++ via meson.

**Building the C++ on Windows/MSVC took four workarounds:**
- Windows' 260-character path limit hit the abseil sources (worked around with `subst`);
- `CXXFLAGS=/std:c++20` was needed for designated initializers;
- the Rust wrapper crate is not MSVC-compatible (GCC flags, Unix symbol-renaming tools), so a small C++ harness links the library directly: [bench/webrtc-cpp/](../bench/webrtc-cpp/);
- the harness links meson's build-tree library rather than the renamed installed copy.

**Same input to both:** real speech (Windows TTS voices, `tools/testdata/make_tts.ps1`) over a synthetic echo path (30 ms delay plus about 60 ms of decaying reflections), with near-end speech added from 12 s. Echo cancellation only, scored by `crates/sd-dsp/src/tests.rs::compare_sonora_with_webrtc_cpp`.

| ERL | Engine | Single-talk ERLE | Double talk: near-end gain | Double talk: SI-SDR (raw → out) |
|---|---|---|---|---|
| 6 dB | sonora | 68.3 dB | −3.1 dB | 6.3 → 4.5 |
| 6 dB | C++ | 67.7 dB | −9.5 dB | 6.3 → −7.1 |
| 10 dB | sonora | 65.0 dB | −2.6 dB | 10.2 → 5.5 |
| 10 dB | C++ | 64.5 dB | −8.8 dB | 10.2 → −6.6 |
| 20 dB | sonora | 55.2 dB | −1.8 dB | 20.2 → 7.3 |
| 20 dB | C++ | 55.3 dB | −7.7 dB | 20.2 → −5.7 |

**Comparison:**
- Single-talk echo removal is equivalent between the two engines, and AEC-path latency is identical (128 samples at 16 kHz).
- Both duck the near-end in double talk, so this is inherent to AEC3's nonlinear suppressor and not a port defect. The C++ library ducks more.
- The two are different WebRTC versions and their outputs differ in detail (cross SI-SDR about −5 dB).

**Decision: sonora.** It matches C++ on echo removal, is better in double talk, and builds on every platform with no C++ toolchain.

## Linear vs full AEC output (`Aec16k`)

AEC3 cancels in two stages: a linear adaptive filter, then a nonlinear suppressor. `sd_dsp::aec16k::Aec16k` runs AEC3 at 16 kHz (single band, cheaper) through `sonora-aec3`'s block processor and returns both outputs.

**Findings while building it:**
- AEC3 expects WebRTC's "FloatS16" scale (±32768). Feeding ±1.0 floats left it silently inactive.
- The two outputs have different latencies: **linear 64 samples, full 128 samples** at 16 kHz. Both are locked by tests and must be compensated separately.

**Real speech, synthetic echo path, matched references:**

| ERL | Output | Single-talk ERLE | Double-talk SI-SDR (raw → out) |
|---|---|---|---|
| 6 dB | full | 68.1 dB | 6.3 → 4.6 |
| 6 dB | **linear** | 21.7 dB | 6.3 → **12.6** |
| 10 dB | full | 64.5 dB | 10.2 → 5.5 |
| 10 dB | **linear** | 22.4 dB | 10.2 → **12.8** |
| 10 dB | hybrid@20 | **60.3 dB** | 10.2 → **8.9** |
| 20 dB | full | 55.2 dB | 20.2 → 7.5 |
| 20 dB | linear | 22.1 dB | 20.2 → 13.1 |

**Synthetic independent talkers (CI test):** double-talk SI-SDR raw 8.0 → full 3.5 → **linear 22.8 dB**; single-talk ERLE full 39.8, linear 22.2 dB.

**The hybrid output** (`AsrMixer`) uses the full output on frames where the suppressor removes more than the threshold (echo-only frames) and the linear output elsewhere, with 10 ms crossfades. The threshold trades single-talk echo against double-talk quality:
- **10 dB ERL:** threshold 10 → 64.4 dB / 6.7 dB; 20 → 60.3 / 8.9; 30 → 49.3 / 10.8 (single-talk ERLE / double-talk SI-SDR).

**The output used for transcription will be chosen by word error rate in M3** (the default is hybrid@20).

## Real speaker → microphone path (this laptop)

`sdctl echo-test` plays speech on the speakers while recording, aligns mic and loopback on host time, and measures.

| Mic stream | Echo in mic | Mic–loopback coherence | Our AEC |
|---|---|---|---|
| default (processed by the Realtek driver) | none detectable: 52 dB below loopback, coherence 0.001 in every band | 0.001 | nothing to cancel. Running it anyway **degraded clean near-end speech from 41.6 to 9–14 dB SI-SDR** |
| raw (`--raw-mic`, `AUDCLNT_STREAMOPTIONS_RAW` accepted) | −27 dBFS, 7.5 dB below loopback, arriving 10.5 ms after it | 0.13–0.64 by band | ERLE 3–6 dB |

**Findings:**
1. **This laptop's mic driver cancels speaker echo itself, and very effectively.** The speakers were confirmed clearly audible during the test. Many laptops do this in their default capture path.
2. **With the raw stream, the echo path is mostly nonlinear.** The low coherence caps any linear canceller at a few dB, and AEC3's suppressor removed little more. The likely cause is speaker-protection or amplifier DSP after the loopback tap. The driver's AEC can use the true post-DSP reference; we cannot.
3. **Running AEC without echo harms the near-end.** Hence the echo gate below.

## Echo gate and production mic chain (`MicProcessor`)

**`EchoGate`** correlates the mic's and the reference's 10 ms loudness envelopes (best lag 0–300 ms) over the far-end-active frames of a 10 s window:

| Case | Correlation |
|---|---|
| device-cancelled echo | 0.00–0.11 |
| independent talkers (double talk) | up to 0.22 |
| real echo | 0.46–0.60 (synthetic: 0.99) |

- **Thresholds:** on above 0.35, off after about 10 s below 0.15.
- **Auto mode starts with AEC off.** Switching on takes about 5 s once echo is evident; switching off takes about 15 s.

**Chain:** `Aec16k` → hybrid mixer → crossfade with a bypass path (same resampler and HPF, delayed to the same 128-sample latency) → optional NS / AGC2 at 16 kHz.

**On the real recordings:**
- **Default mic:** AEC never engaged, and double talk stayed at 41.6 dB SI-SDR.
- **Raw mic:** engaged at 5.0 s, with ERLE 4.3 dB (the nonlinear-path limit) and double talk −2.7 → −0.5 dB.

## Voice activity detection (`sd-vad`)

**Model:** Silero VAD **v6.2.3** (MIT; the repository's LICENSE file was checked because a README badge says CC BY-NC), file `silero_vad_16k_op15.onnx`, 1.29 MB.
- It is pinned by URL, size and SHA-256 in `models/manifest.json` and fetched with `sdctl models fetch`.
- Weights are not committed.
- It runs on ONNX Runtime (`ort` 2.0.0-rc.14).

**Interface,** checked against the model's declared inputs and outputs: 512-sample chunks at 16 kHz, each prefixed with the previous 64 samples; recurrent state `[2,1,128]`; `sr` = 16000.

**Streaming segmenter** with Silero's reference defaults: threshold 0.5, hysteresis 0.35, minimum speech 250 ms, minimum silence 300 ms, 100 ms padding.

**Accuracy** on real speech (both TTS voices, 63 s) with low-pass noise added. Ground truth is the 32 ms frames of the clean speech within 40 dB of the peak.

| SNR | Precision | Recall | Frame F1 |
|---|---|---|---|
| clean | 0.959 | 0.994 | **0.976** |
| 20 dB | 0.959 | 0.987 | 0.973 |
| 10 dB | 0.956 | 0.981 | 0.968 |
| 5 dB | 0.953 | 0.978 | 0.965 |
| 0 dB | 0.935 | 0.966 | **0.950** |

**Other checks:**
- Silence: every chunk has probability below 0.2.
- White noise: fewer than 5 % of chunks flagged.
- Results are identical however the audio is split into pushes.

**Cost:** 119 µs per 32 ms chunk on one thread (0.0037× real time), so about 0.4 % of a core per stream.

**Toolchain note:** ONNX Runtime ships no prebuilt binaries for `x86_64-pc-windows-gnu`. This machine's Rust defaulted to the GNU host, so local builds now use the MSVC toolchain (directory override), which is the standard Windows target and the one CI uses.

## ONNX Runtime loading (production fix)

**The problem.** The first VAD integration linked ONNX Runtime through `ort`'s prebuilt binaries. Those import `DirectML.dll`, which the build installs as a **symlink** into a user cache. In the user's terminal environment Windows could not resolve it, and `sdctl` died before `main` with `0xC0000135 STATUS_DLL_NOT_FOUND`, silently and including audio capture. A missing optional ML component must never stop the audio engine.

**The fix.**
- `ort` is now built with `load-dynamic`. The executable has no link-time dependency on ONNX Runtime or DirectML (verified from its import table).
- The official Microsoft ONNX Runtime **1.31.0** CPU build (2026-10-09, matching `ort`'s API level 31) is pinned per platform in `models/manifest.json`, with the archive's SHA-256 and the extracted library's SHA-256. `sdctl models fetch` installs it. The Windows `onnxruntime.dll` depends only on system DLLs and the VC++ runtime.
- **Lookup order:** explicit setting, then `SD_ORT_DYLIB`, then next to the executable, then `models/`.
- If the library or model is missing, the processing stage emits a `Warning` event and keeps recording and processing without VAD.

VAD results through the official runtime are identical to before (F1 0.976 / 0.973 / 0.968 at clean / 20 / 10 dB).

## Processing stage (`sd-engine::processing`)

**Data flow:** drain thread → bounded queue (copies of blocks; the journal is unaffected) → processing thread. The thread aligns system audio to each mic frame by **host time**, then runs:
- `MicProcessor` (echo gate, AEC, AGC2) → 16 kHz mic;
- system audio → 16 kHz mono;
- Silero VAD and the segmenter on both.

Outputs go to a bounded `ProcessedChunk` channel and as `Speech` / `EchoCanceller` / `Warning` events.

**Drift.** A phase-locked fractional read pointer follows the host-time target with 4-point cubic interpolation. Errors beyond 2 ms resync instead.

| Synthetic test (band-limited far end, 30 ms echo, system clock offset) | ERLE | Reference rate tracked | Resyncs / timeouts |
|---|---|---|---|
| no drift, system starts 37 ms after mic | 33.0 dB | 0 ppm | 0 / 0 |
| system clock +200 ppm, 30 s | **31.6 dB** at the end | **+199.96 ppm** | 0 / 0 |
| same, earlier integer sample slips (replaced) | 20.0 dB | — | 0 / 0 |

**Bugs found and fixed during integration:**
- mic frames that predate the system stream were mapped onto the system's first samples, a 37 ms misalignment;
- reference waiting was based on mic backlog instead of host-time lag, so bursts looked like stalls;
- system output was tied to the AEC reference pointer, so a timeout would have dropped system audio.

**Offline processing** (`OfflineProcessor`, `sdctl process <journal>`) runs the same stage over a journal at **0.031× real time** (60 s in 1.9 s). It is for catch-up after overload or crash, for reprocessing, and for evaluation. **Replay reproduces the live output bit for bit** (see "Live vs offline replay" below).

## Live run (`sdctl record --process`, 60 s; TTS speech played on the speakers at about 10–46 s)

**System audio:**
- 15 speech segments between 10.0 and 46.0 s, following the TTS sentences.
- The echo gate never engaged, which is correct: this laptop's driver cancels the echo.
- 0 dropped input/output, 0 reference resyncs or timeouts.
- Mic/system clock drift 0.11 ± 0.78 ppm, not significant.

**Mic:**
- **No speech detected at first.** This laptop's driver-processed mic is extremely quiet: −64 to −90 dBFS RMS, against about −20 to −30 for normal speech capture.
- **AGC2 was found inactive.** `GainController2::default()` leaves adaptive digital gain off (WebRTC's default), so it is now enabled explicitly.
- Reprocessed with AGC: mic output rose −67.1 → −51.9 dBFS RMS (capped by the −50 dBFS output-noise limit), and VAD found mic speech at 25.7–27.0 s and 57.3–57.9 s.
- **Unconfirmed:** whether those are the user's speech. A scripted mic test is still to do.

**Scripted mic test** (`record --process --agc`, 30 s; the user spoke from about 9.4 to 18 s and was silent otherwise):
- **VAD:** speech at **9.44–11.11 s and 11.36–18.18 s** (the gap is a pause in speaking), with no false segments in the silent parts.
- **Mic level:** while speaking the capture read −22 to −31 dBFS RMS, peaking at −10.9 dBFS, which is a normal level. **The earlier "too quiet" reading was silence, not a level problem:** this driver path idles at about −85 dBFS, and after speech the Realtek driver's own noise suppression outputs exact digital zeros (−180 dBFS) for seconds at a time.
- **Processed with AGC2:** speech −20.6 dBFS RMS, silence before speech −66 dBFS, after speech −99 dBFS. Without AGC: −30.4 dBFS RMS overall.
- **Limiting at speech onset:** 3 of 482,720 samples reached full scale and 11 reached at least 32,000, all in the first 0.7 s of speech. They are AGC2's limiter at its ceiling, not wrapped samples. The cause is AGC2's 15 dB initial gain, applied before its level estimate settles. Re-running the same journal offline with different initial gains:

  | initial gain | samples ≥ −0.3 dBFS | first 1 s of speech | last 3 s of speech (settled) |
  |---|---|---|---|
  | 15 dB (WebRTC default) | 16 | −14.5 dBFS | −25.2 dBFS |
  | **10 dB (new default)** | **0** | −17.4 dBFS | −25.2 dBFS |
  | 6 dB | 0 | −21.4 dBFS | −25.2 dBFS |
  | 0 dB | 0 | −27.3 dBFS | −25.2 dBFS |

  The settled level does not depend on the initial gain. Only the onset overshoot does. The default is now 10 dB (`MicConfig::agc_initial_gain_db`). A very quiet mic converges about 1 s later than with 15 dB, because gain changes are limited to 6 dB/s. This was measured on one mic and one speaker.
- 0 dropped, 0 reference resyncs or timeouts.

## Journal compression (`sd-journal::codec`)

Raw 16-bit journaling at 48 kHz (mic mono + system stereo) costs **1.04 GB per hour**. M1 left open whether to compress, journal at 16 kHz, or both. The journal must keep the original capture (it is the recovery source, and the ASR input rate may change), so the rate stays 48 kHz and the journal is compressed losslessly.

**Design.** Each block (normally 10 ms) is compressed on its own, so a torn write or a flipped bit damages only that block, and the per-block CRC covers exactly the stored bytes. The scheme is FLAC's "fixed" subframe:
- per channel, the best polynomial predictor of order 0–3;
- the residual Rice-coded in 128-sample partitions;
- for stereo, `right − left` is stored instead of `right` when that is smaller;
- digital silence costs 1 byte per channel, and incompressible audio is stored verbatim, so a block never expands by more than 3 bytes.

It is about 300 lines of Rust with no new dependencies. A full FLAC stream was not used because it needs a stream header and seek structure, which works against independent crash-safe blocks.

**Format.** The journal format is now **version 2**: each segment header records its codec, and each index record stores the block's byte length. Version 1 (raw PCM) journals still read; all the M1/M2 recordings were read and re-encoded. `JournalConfig::codec` selects `Lossless` (the default) or `PcmS16le`. `sdctl journal recode <in> <out> [--codec pcm|lossless]` rewrites a journal and verifies it block for block.

**Robustness tests.**
- Lossless round trips: random signals, full-scale noise, ±32767 alternation, 1–3 channels and 0–600 frames.
- 6,000 corrupted or random inputs to the decoder: each must be rejected or decode, and must never panic.
- Torn writes at many offsets of data and index, with both codecs: the reader returns exactly the whole blocks before the tear.
- A flipped byte inside a compressed block stops that segment at the block, and later segments survive.

**Real recordings** (every one re-encoded with `journal recode` and verified identical):

| Recording | Content | Size vs raw PCM |
|---|---|---|
| hotplug1 / hotplug2 (4–5 min) | mic room noise, system mostly idle | 2.9 % / 3.0 % |
| mictest1 (1 min) | mic speech + silence, system idle | 4.3 % |
| live2 (2 min) | TTS played on speakers, mic hears it | 9.3 % |

Those are mostly quiet, so dense audio was also measured in 10 ms blocks (`cargo run --release -p sd-journal --example codec_ratio`):

| Signal | Level | Size vs PCM | Bitrate |
|---|---|---|---|
| Mic, live speech only (mictest1, 9.4–18.2 s, 48 kHz mono) | −24.7 dBFS | 33.5 % | 257 kbit/s |
| TTS speech files (48 kHz mono) | −20.5 dBFS | 32 % | 245 kbit/s |
| System loopback during TTS playback (live2, 48 kHz stereo) | −22.0 dBFS | 10.3 % | 159 kbit/s |
| Mic, mostly room noise (live2) | −66.8 dBFS | 7.3 % | 56 kbit/s |

So continuous speech on both sources costs roughly **0.1–0.25 GB per hour**, instead of 1.04. Loud stereo music, the worst realistic case, was measured later in a live session: 66 % of PCM, about 0.7 GB per hour for the whole journal (see "Live session" below). There was also no reference FLAC encoder on this host to compare against; LPC-based FLAC typically compresses a few percent better than fixed predictors.

**Cost.** In the journal benchmark, writing one 10 ms block of 48 kHz stereo (tone plus noise at about −20 dBFS, with different channels) takes **31.5 µs lossless versus 8.0 µs raw**. Both include the amortized 1 s fsync. That is 0.3 % of one core, on the journal thread, not the capture thread. Re-encoding the recordings ran at about 0.001× real time. The 5-minute, 60× no-loss simulation still passes with lossless journaling.

**Fix found by this change.** `RecordingSummary::health.journaled_frames` was the drain thread's last snapshot, taken before the journal thread had written its remaining queue. The slower debug-build encoder exposed this as a flaky count; it was not data loss, because the journal itself was complete. The summary now reads the counters after the journal thread has joined.

## Live vs offline replay

Offline replay of recordings used to report a few reference resyncs and timeouts that the live run did not. I first attributed this to the aligner's 8 ms tolerance at loopback idle/resume boundaries. **That was wrong.** Debug output showed every event at the very end of the recording:
- **Stop tail:** the system stream's last block ends 60–100 ms before the mic's (the two never stop at the same instant), so the last 6–10 mic frames have no reference audio.
- **Counted twice:** each of those frames was counted as a timeout, and also as a "resync", because the read pointer could not advance past the end of the reference.
- **Different reports, same behaviour:** live had the same tail, but `record`'s summary used the drain thread's last health snapshot, taken before the processing thread flushed it. The two paths behaved identically; the live report just stopped counting too early.

**Fixes:**
1. The stop summary reads the processing counters after the processing thread has joined. The journal counters got the same fix earlier.
2. Mic frames after the reference's end, at flush, go in a new `reference_ended_frames` counter, not `reference_timeouts`. A resync is counted only when it is applied.
3. At flush, a partially available final reference frame is used (missing samples read as silence) instead of being dropped.
4. The live stage now receives samples at journal precision (16-bit, via the journal's own conversion), so it sees exactly what a replay will read. Before this, live and replay outputs differed by ±1 LSB in about 10 % of samples. 16-bit input is far beyond what recognition needs.

**Results:**
- **Every recording reprocessed:** timeouts are now 0 everywhere; the stop tail is 6–10 frames. The only resyncs left are 4 in `hotplug1`, which has 7 mic discontinuities from Bluetooth device switches, so they are real events.
- **Fresh 20 s live recording with processing and AGC, then `sdctl process` on its journal:** identical counters, and identical processed mic and system WAVs, sample for sample.
- **The threaded engine test** (synthetic echo, AEC active, ERLE > 20 dB) now reprocesses its own journal and requires every output chunk to be identical. It passed 3 of 3 runs. With the live input switched back to full precision, it fails on the first chunk, so the comparison is sensitive.
- **Stop-tail unit test:** `reference_ending_first_is_not_a_timeout` checks the case where the system stream ends 7 blocks early: 0 timeouts, 0 resyncs and exactly 7 ended frames.

Exact equality requires that live processing had no input drops and no reference timeouts; the counters show when that does not hold.

**Delivery-timing independence.** The bit-exact check then failed once on a slow Linux CI runner. Locally I reproduced it by running 12 copies of the test at once (the test feeds audio at 5x real time). Three timing dependencies were found and fixed:
1. **Bursts misread as a stalled reference.** After a scheduling stall, the drain can deliver a burst of one source's blocks before the other's, and the host-time lag alone then looked like a stalled reference (the 400 ms budget is only 80 ms of wall time at 5x). The live stage now gives up only when the reference has been missing in both host time and wall-clock time. Offline replay delivers blocks in host-time order and keeps the host-time rule.
2. **Reference position computed from whatever had arrived.** It came from the newest system block starting before the mic frame's time, so it depended on which blocks had arrived. It is now computed only once a system block starting *after* that time has arrived. Blocks of one source arrive in order, so the inputs are then always the same; this costs at most one block of extra latency.
3. **Block timestamps capped at the last 256.** When one source ran more than 256 blocks ahead, needed timestamps were dropped. Timestamps are now trimmed together with their audio.

The pre-start give-up was also not counted at all; every frame processed without a reference while a system stream is configured is now counted. Result: 48 of 48 runs of all processing tests pass with 16 test processes at once, where 1 of 12 failed before. A fresh live recording still replays bit for bit.

## Live session: real double talk and loud music

Three recordings on this laptop's speakers and built-in mic array, with the user present. The far end was the TTS file `far_en.wav` (male voice, about 37 s) played through the speakers at a normal meeting volume (about 50–70 %). In the double-talk recordings the user read a known paragraph over it, then a second one alone. The recordings are kept locally (`target/dt1`, `target/dt2`, `target/music1`, not committed) and have known text, so M3 can reuse them for WER.

**A: default mic mode (driver processing on).** The Realtek driver's own echo cancellation removed the echo before capture.
- **Far end only:** the raw mic read −71 to −88 dBFS, and during one stretch the driver output exact digital zeros.
- **Echo gate:** it correctly stayed off, and our chain passed the mic through unchanged (±0.5 dB).
- **User's voice:** about −25 dBFS during double talk versus about −24 dBFS alone. There was no sign of the driver chopping near-end speech.
- **VAD:** detected the user's speech in both readings, and nothing during far-end-only periods.

**A2: raw mic (driver processing bypassed), so our AEC does the work.** Real echo is present at about −27 dBFS, over a room-noise floor of about −32 dBFS that the driver normally hides.
- **Echo only (31–51 s, 20 s):** raw echo plus noise −24 to −34 dBFS, processed −33 to −72 dBFS. **No mic speech was detected in those 20 s.**
- **Double talk:** the user's reading was detected throughout (17.1–31.3 s). Processing lowered the mic by 2–3 dB during double talk, versus about 0.5 dB when the user spoke alone. Roughly half of that difference is the removed echo, so near-end attenuation is at most about 1–2 dB. Whether the speech is *intact* (not just present) will be measured by WER in M3.
- **Echo gate onset:** the gate turned on 3.3 s after the far end started, and the first **1.3 s of far-end speech leaked through and was detected as mic speech**. This is the cost of starting with the gate off and requiring 10 consistent correlation evaluations.
- **Replay:** reprocessing the journal reproduced the live output bit for bit, with AEC active.

**B: loud music, raw mic, user silent (60 s).** A full-band stereo song played at the same volume.
- **System audio:** −16 dBFS, genuinely stereo (side signal −19 dBFS).
- **Echo:** the gate turned on at 5.8 s. The mic went from −23..−27 dBFS raw to −32..−41 dBFS processed. **No mic speech was detected in 60 s of music with vocals**; all VAD speech segments were on the system source.
- **Journal size, worst case:** the system stream compressed to **66 % of PCM** (1.0 Mbit/s). The raw mic (music echo plus room noise) also came to about 66 % (0.5 Mbit/s). The whole journal was 11.8 MB for 60 s, about **0.7 GB per hour**, versus 1.04 GB raw. Speech-dominated sessions measured 0.1–0.25 GB per hour.

## Cost (criterion, per 10 ms frame, i5-12450HX)

| Chain | Time | Share of one core |
|---|---|---|
| 48 kHz full APM (AEC + NS + AGC + HPF) | 57.5 µs | 0.58 % |
| `MicProcessor` (16 kHz AEC3 + gate + mixer + bypass), AGC on | 47.1 µs | 0.47 % |
| `MicProcessor`, AEC off (AEC keeps adapting in the background) | 46.2 µs | 0.46 % |

## Open items

- **Echo gate onset latency.** In the live double-talk run the gate engaged 3.3 s after the far end started, and the first 1.3 s of far-end speech was detected as mic speech. Options are a faster first decision, or running AEC from the start and gating only the bypass. Transcript-level deduplication (below) is the safety net either way.
- **Raw-mic noise.** With driver processing bypassed, room noise is about −32 dBFS. Whether noise suppression helps recognition there will be measured by WER in M3.
- **AEC on a linear echo path.** A USB mic with external speakers is untested; this laptop's raw path is nonlinear.
- **Transcript-level deduplication of remote speech that leaks into the mic** is still needed as a final safety net. That is M3/M4.
- **Noise suppression** is kept optional and off by default. Whether it helps or hurts recognition will be measured by WER in M3.
- **Replay with active system audio was verified only by the synthetic threaded test.** The live recordings used for the bit-exact comparison had idle loopback.
