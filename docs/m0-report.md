# M0 report: foundations

Date: 2026-10-13. Host: Windows 11 x64, Intel i5-12450HX (8C/12T), 12 GB RAM, NVMe SSD, Rust 1.94.1.

## What was built

| Component | Crate | Purpose |
|---|---|---|
| Core types | `sd-core::types` | `Source` (mic/system), `StreamFormat`, `BlockMeta`, `BlockFlags` (discontinuity, silent, gap-fill, estimated timestamp) |
| Host clock | `sd-core::clock` | Nanosecond host-monotonic time on the clock each OS reports device positions in (QPC / `CLOCK_UPTIME_RAW` / `CLOCK_MONOTONIC`), plus WASAPI `qpcPosition` conversion |
| Drift estimator | `sd-core::drift` | Sliding-window least-squares fit of device frame position vs host time → true rate, ppm, jitter; predicts host time of any frame |
| Audio ring | `sd-core::ring` | Lock-free SPSC ring for real-time callbacks: no allocation, all-or-nothing block writes, dropped blocks counted and kept on the timeline as explicit gaps |
| Audio journal | `sd-journal` | Crash-safe segmented on-disk store: per-segment index, CRC on every record and its sample data, data-before-index fsync, recovery that never returns corrupted audio, WAV export with gap filling |
| CLI | `sdctl` | `journal inspect`, `journal export`, `simulate` (accelerated end-to-end no-loss check) |
| CI | `.github/workflows/ci.yml` | fmt, clippy (-D warnings), tests, simulation and bench smoke on Windows/macOS/Linux runners |

## Verification

### Automated tests: 29 passing (`cargo test --workspace`)

- **sd-core (18):**
  - format validation and 24 h frame/ns round trip;
  - clock monotonicity, and that WASAPI `qpcPosition` lands on the same clock;
  - drift recovery for −120…+200 ppm under ±2 ms callback jitter (error < 5 ppm), restart detection, frame→host-time prediction (< 1 ms);
  - ring round trip, whole-block drop on overflow with the gap flagged, metadata-ring overflow, timeline skip, too-small buffers;
  - a 50,000-block concurrent producer/consumer stress test where every frame is either delivered bit-exact or counted as dropped.
- **sd-journal (11):**
  - header/record encode-decode with bit-flip detection, sample conversion and clipping;
  - round trip across segment rotation;
  - crash simulation that discards unflushed buffers: all synced audio recovered;
  - torn writes at hundreds of truncation offsets of data and index files, with only whole verified blocks returned;
  - a corrupted byte stopping its segment while later segments stay readable and the hole is reported as a gap;
  - missing index file;
  - gap reporting and silence fill on export;
  - invalid-use errors.

`cargo clippy --workspace --all-targets -- -D warnings`: clean.

### No-loss simulation (`sdctl simulate`, release build)

Synthetic mic (48 kHz mono) and system (48 kHz stereo) audio is produced in 10 ms blocks at N× real time, passed through the ring into the journal, and every sample is then verified.

| Run | Audio frames (per source) | Lost | Sample mismatches | Wall time | Journal size | Worst single write (incl. fsync) |
|---|---|---|---|---|---|---|
| 10 min at 60× | 28.8 M | 0 | 0 | 10.0 s | 164.8 MiB | 29.7 ms |
| 60 min at 120× | 172.8 M | 0 | 0 | 30.1 s | 988.8 MiB | 240.4 ms |

### Micro-benchmarks (criterion)

| Benchmark | Time per 10 ms 48 kHz stereo block | Share of the 10 ms real-time budget |
|---|---|---|
| ring push + pop | 0.57 µs (95 % CI 0.54–0.60) | 0.006 % |
| journal write (amortized 1 s fsync) | 18.7 µs (95 % CI 17.4–19.9) | 0.19 % |

### CI (GitHub Actions run [38379790937](https://github.com/Nikhil1602/sonicdiar/actions/runs/38379790937), commit `582246b`)

fmt, clippy (`-D warnings`), all tests and the 5-minute no-loss simulation passed on `windows-latest`, `macos-latest` and `ubuntu-latest`. This is the first time the macOS (`CLOCK_UPTIME_RAW`) and Linux (`CLOCK_MONOTONIC`) clock paths were compiled and tested.

## Findings that change the design

1. **fsync stalls must not reach the ring.** The first simulation run (ring not scaled for 60×, so ~33 ms of wall-clock headroom) dropped 0.3–0.6 % of frames because the draining thread blocked in fsync for up to 64 ms. Worst observed stall on this NVMe SSD was 240 ms. Slow disks, antivirus scanning, or network drives can stall for seconds. **M1 requirement:** the ring-drain thread hands blocks to the journal through a large bounded in-memory queue (≥ 30 s), and the journal writes and fsyncs on its own thread. Ring overflow then requires a multi-second stall *and* a full queue, and is reported as a health event.
2. **Journal size.** Uncompressed 16-bit PCM at 48 kHz (mono + stereo) is ~989 MiB per hour, too large for long sessions on low-end machines. **M1/M2 decision:** journal at the processing rate (16 kHz) and/or store FLAC-compressed segments. This is to be measured against CPU cost.

## Not yet verified

- Any real audio device: capture starts in M1.
- Power-loss durability on real hardware: tested by simulation (discarded buffers, truncation, corruption), not by cutting power.
