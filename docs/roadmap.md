# Roadmap: meeting audio, robust on Windows, macOS and Linux

Date: 2026-10-13. This plan replaces the milestone table in [architecture-proposal.md](architecture-proposal.md) §7 from M3 on. It is based on three milestone reports: [M0](m0-report.md), [M1](m1-report.md), [M2](m2-report.md).

## 1. Scope

The audio side of a meeting recorder:
- separate mic and system capture;
- transcripts with timestamps;
- speaker labels, including across meetings;
- language handling;
- mic-bleed removal;
- crash-safe recording;
- device recovery;
- final audio export.

And beyond that:
- real AEC/NS/AGC;
- true streaming with partial results;
- Gujarati and code-switching;
- overlap handling;
- native Linux capture with no CLI tools.

| Capability | Status | Planned in |
|---|---|---|
| Separate mic + system capture, device recovery | **Windows done** (M1); **Linux done in CI** (M6, PulseAudio API, no CLI tools) | M7 macOS |
| Crash-safe storage | **Done**: CRC journal, lossless, bit-exact replay | n/a |
| AEC / NS / AGC | **Done** (M2): AEC3 + echo gate, AGC2, NS optional | n/a |
| VAD | **Done**: Silero, F1 0.95–0.98 | M4 adds seg-3.0 for overlap |
| Transcription | **Done** (M3): streaming partials (p50 0.5 s) and finals (p50 1.8 s after speech ends), [m3-report.md](m3-report.md) | Model tiers for CPU-only machines still open |
| Mic-bleed removal | **Done**: AEC (M2) plus transcript dedupe of finals (M3) | n/a |
| Diarization | **Done** (M4): pyannote segmentation for speech/overlap, WeSpeaker embeddings on 3 s windows, average-linkage clustering, live assignment with re-clustering and stable ids; AMI DER 15–16% vs 40.5% for a plain baseline ([m4-report.md](m4-report.md)) | Live overlap, VoxConverse, speaker profiles |
| Cross-meeting speaker identity | Not started | M4 (opt-in, local) |
| Languages | **Done in M3** (pulled forward): all 15 product languages, Auto by default, Whisper + IndicConformer router with two-stage LID (99.2% per clip) | Number tables for 7 Indic languages |
| Final audio export | WAV export | M8 (Opus/AAC via a pluggable encoder) |
| Meeting-app detection | Not planned | Optional M8 helper (detection only) |

**Out of scope:** UI, cloud upload by default, LLM notes. The SDK exposes events that an app can build those on.

## 2. Cross-platform architecture

The parts above capture are platform-independent and already verified on Windows:
- journal;
- timeline aligner and drift estimation;
- processing stage (AEC, AGC, VAD);
- everything built from M3 on.

Each OS therefore only has to implement **one capture backend contract**. This is the shape of [`sd-capture`](../crates/sd-capture/src/lib.rs) today.

**Backend contract** (identical on every OS, enforced by a shared test suite):
- **Events:** `Started{device, format, raw}`, packets, `DeviceChanged`, `DeviceLost`, `Stopped`. Each packet carries frames, the device position, a host timestamp (`HostTime`, monotonic ns) and glitch/discontinuity flags.
- **Format:** the backend delivers **48 kHz float** (mic mono, system stereo). If the OS can't convert, the backend resamples; the core never sees other rates.
- **Default device:** "default" follows the default device. An explicit id falls back to default and reports it.
- **No blocking:** the real-time callback never blocks; it only writes to the lock-free ring.

| | Windows (done) | Linux (M6) | macOS (M7) |
|---|---|---|---|
| Mic | WASAPI shared, event-driven | **PipeWire** native stream (`pipewire-rs`) | Core Audio AUHAL / `AVAudioEngine` input |
| System audio | WASAPI loopback of the default render endpoint | PipeWire: monitor of the default sink (`stream.capture.sink=true`) | **Core Audio process tap** (macOS 14.2+) on an aggregate device; **ScreenCaptureKit** audio fallback (macOS 13+) |
| Fallback | n/a | PulseAudio (`libpulse`) for distros without PipeWire | SCK when tap creation fails or the tap delivers all-zero buffers (a known beta issue) |
| Timestamps | QPC + device position | `pw_stream_get_time_n` (CLOCK_MONOTONIC) + stream ticks | `mHostTime` (mach ticks → ns) + sample time |
| Device change | IMMNotificationClient, 200 ms settle | PipeWire registry events (default node metadata) | `kAudioHardwarePropertyDefault*Device` listeners |
| Permissions | Mic privacy setting → `E_ACCESSDENIED` reported clearly | None for desktop sessions; Flatpak needs the portal (documented) | TCC: Microphone + "System Audio Recording"; checked up front and reported as a typed error |
| Driver voice processing | Default category; RAW optional (measured in M2) | None by default | Optional `kAUVoiceIOProperty` path off by default (we run our own AEC) |

Three robustness rules apply to every backend:
1. **Cap automatic silence insertion** per session: a reported event, and a backend restart once the cap is hit. Loopback idle fill (silence while nothing plays) is legitimate and stays uncapped.
2. **Long gaps become a new segment, not inserted silence.** The journal already records gaps without filling the file. The processing stage will reset AEC and VAD state across a gap of more than 10 s.
3. **Configure every client property before stream start.** A property set after the stream starts may silently not apply. Each backend has a test that reads back the effective configuration.

## 3. Robustness work shared by all platforms (M2.5, before M3)

These are small, Windows-verifiable, and needed before long ASR sessions run on top:

| Item | Detail | Acceptance |
|---|---|---|
| Silence-fill cap | Per source per session, default 300 s; event + restart | Unit test with a fake device that keeps reporting gaps |
| Stall watchdog | No packets for more than 2 s while started: restart the stream | Fault-injection test; live test with a device that is suspended |
| Disk tiers | Warn below 500 MiB; pause journal writes when the disk is full (queue absorbs ≥ 30 s); resume at ≥ 50 MiB free | Test on a small RAM disk/VHD |
| Long-session guard | Configurable max duration + event | Config test |
| Echo-gate onset | Start AEC adapting immediately; gate only the bypass mixing. Target: first far-end speech no longer reaches the mic VAD (M2 measured a 1.3 s leak) | Rerun of the M2 live double-talk script: 0 false mic speech segments |
| 8-hour soak | Real devices, processing on, periodic device switches | 0 lost frames; drift residual < 20 ms; RSS flat |

## 4. Milestones

Order rationale:
- Transcription, diarization and language work are platform-independent. Built and measured on Windows first, they then carry over to Linux and macOS unchanged.
- To keep platform problems from surfacing late, **CI builds and unit-tests every new crate on all three OSes from M3 on**, including whisper.cpp and ONNX Runtime linking.
- The Linux backend comes before macOS because CI can test it end to end with virtual PipeWire devices.

### M3: Streaming ASR
- **Engine:** whisper.cpp through Rust bindings. It runs CPU, Vulkan and Metal on all three OSes, and has proven multilingual support. Builds:
  - CPU (AVX2 with an SSE fallback; runtime dispatch);
  - Vulkan (Windows/Linux);
  - Metal (macOS).
- **Streaming policy** (architecture proposal §3):
  - VAD-bounded windows;
  - hypotheses every 1–2 s, with the stable prefix emitted as `partial`;
  - commit on N agreeing hypotheses or end of speech;
  - word timestamps mapped to the session clock;
  - `revision` events reserved for speaker relabels.
- **Model tiers** by measured capability (CPU cores, RAM ≈ 10× model, GPU): tiny/base/small/medium/large-v3-turbo. On a low-end CPU-only machine, base with int8/q5 must keep up in real time.
- **Mic-bleed safety net:** AEC first. Remaining duplicates are removed with text rules (VAD overlap, fuzzy match within ±4 s, ≥ 0.8 similarity) applied to finals only.
- **Measured:**
  - WER/CER on FLEURS en/hi/gu subsets and on our recorded double-talk sessions (known text);
  - partial latency p50/p95 and final latency;
  - real-time factor and RSS per tier on this machine.

  Also decided here, by WER: whether NS should be on, and which AEC output is used (raw/linear/full).

### M4: Diarization
- **Baseline:**
  - kaldi-native-fbank (80 mel, 25/10 ms, Hamming, dither 0);
  - WeSpeaker ResNet34 ONNX (256-d, L2-normalised);
  - average-linkage AHC on Euclidean distance, cut around 1.0 (≈ cosine 0.5), or at K when the speaker count is known.
- **Beyond the baseline:**
  - online assignment to centroids with a new-speaker threshold;
  - global re-clustering every ~60 s and at the end, with Hungarian matching for stable ids and `revision` events;
  - **overlap** from pyannote seg-3.0 powerset output, emitted as overlapping turns;
  - mic and system diarized separately.
- **Optional:** local speaker profiles (opt-in) for cross-meeting identity.
- **Measured:** DER with and without overlap on AMI / VoxConverse subsets; id stability across revisions; a threshold sweep (starting from the baseline's cut of 1.0).

### M5: Multilingual and code-switching
- Per-segment LID (Whisper's own LID, plus an evaluated lightweight LID if needed), with a language prior from neighbouring segments.
- **Router:**
  - English and other non-Indic languages → Whisper.
  - Gujarati and other Indic languages → IndicConformer when installed. It is gated, MIT licensed and has no English, so the router never sends English to it.
  - Hindi → whichever engine wins the WER benchmark.
- **Code-switching (Hindi–English, Gujarati–English):** measured on a recorded set with references; limits are reported, not assumed.

### M6: Linux capture backend

**Done (CI-validated), with one change:** a single backend on the PulseAudio API serves both PulseAudio and PipeWire (`pipewire-pulse`); see [m6-report.md](m6-report.md) §2 for why. The real-hardware run is open.
- PipeWire native backend with a PulseAudio fallback, implementing the contract and passing the shared contract suite.
- **CI end-to-end on `ubuntu-latest`:**
  - start PipeWire in the job;
  - create a null sink + virtual source;
  - play a known signal with `pw-play`;
  - capture mic + monitor through our backend;
  - verify no loss, alignment, drift and a forced default-device switch.
- **Locally:** WSL2/WSLg (PulseAudio over RDP) for a second real-audio path on this laptop.
- **Real hardware:** one physical Linux run (live USB or a Linux machine), reported like M1.

### M7: macOS capture backend

**Done (CI-validated on macOS 14 and 15 with virtual devices):** Core Audio IOProc for the mic, a process tap on a private aggregate device for system audio, permission handling; ScreenCaptureKit fallback not built. Hosted runners *can* grant the audio permissions and route virtual audio after all (BlackHole + TCC database writes), so capture is proven by CI except on real hardware. See [m7-report.md](m7-report.md).
- Core Audio tap + aggregate device for system audio, AUHAL for the mic, ScreenCaptureKit fallback, TCC permission checks with typed errors.
- **CI (`macos-latest`):** compile, unit tests, offline pipeline. Hosted runners cannot grant TCC or route real audio, so **capture is not proven by CI**.
- **Real validation needs a Mac** (see §6). Until then, macOS is shipped as "builds, untested capture", and the docs say so.

### M8: SDK and release

**Done:** see [m8-report.md](m8-report.md) and [sdk.md](sdk.md). Rust API, C library, Python and Node.js bindings; releases on GitHub (wheels, npm package, C archives, licence report, SBOM, build-provenance attestations instead of signed tags). Not done: PyPI/npm publishing (by choice), code-signing certificates.
- C ABI frozen (`sd-ffi`), Rust async stream API, Node (napi-rs) and Python (pyo3/maturin) bindings with the same event model.
- **Packaging:** per-OS artifacts bundle ONNX Runtime and whisper.cpp builds; Windows needs the VC++ runtime. Models are fetched and SHA-pinned at first use, never bundled.
- **Release workflow:** signed tags, SBOM, licence report, platform test matrix, benchmark report.
- **Docs:** integration guide, privacy notes (all local; cloud is opt-in), known limits per platform.

## 5. Test strategy (what proves what, where)

| Layer | Windows (this laptop) | Linux | macOS |
|---|---|---|---|
| Core, journal, DSP, VAD, ASR, diarization (offline) | CI + local | CI | CI |
| Capture contract suite (fake device) | CI | CI | CI |
| Capture with virtual devices | n/a | **CI** (PipeWire null sink / virtual source) | not possible on hosted runners |
| Capture with real devices | **local** (done in M1/M2, plus the M2.5 soak) | WSLg locally + one physical run | **needs a Mac** |
| Long soak (8 h) | local | CI (virtual, 1 h) + physical (if available) | Mac (if available) |
| Accuracy (WER, DER, VAD F1) | local bench, reproducible from the model manifest | CI smoke | CI smoke |

Every milestone ends with a report in `docs/` giving measured numbers, the open items and anything not verified, as M0–M2 did.

## 6. Decisions and risks

**Decided (no input needed):**
- whisper.cpp for ASR;
- WeSpeaker + AHC baseline for diarization;
- ~~PipeWire-first on Linux~~ the PulseAudio API for both Linux servers (decided in M6, [report](m6-report.md) §2);
- Core Audio tap + SCK on macOS;
- the order above.

**Needs the owner:**
- **macOS hardware.** Without a Mac (owned, borrowed or cloud-rented Apple silicon with a desktop session), macOS capture can be built but not validated.
- **A physical Linux machine** (optional): WSLg + CI virtual devices cover most of Linux, but not real drivers.

**Risks:**
- macOS taps are young (all-zero buffers reported on a beta); hence SCK fallback and health detection.
- IndicConformer size may not run in real time on low-end CPUs (int8 measured in M5).
- Code-switching quality is model-bound.
- Low-end real time for streaming Whisper may require the base/tiny tiers.
- Overlap ASR is research-grade; overlapping regions are labelled, but transcribing both talkers cleanly is not promised.
