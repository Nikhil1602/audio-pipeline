# Soak and memory-probe scripts (Windows)

| Script | Purpose |
|---|---|
| `soak.ps1 -Seconds 14400 -Out target\soak2` | Runs `sdctl record --process --agc --restart-every 1800` for the given time. Keeps an inaudible noise file playing so the loopback stream carries real audio, and logs the recorder's working set, private bytes, CPU time, handles and threads to `<out>.resources.csv` every minute. |
| `leakprobe.ps1 -Mode none\|proc\|vad -Minutes 20` | Runs `crates/sd-engine/examples/leakprobe` (the recorder on synthetic audio at ~20x real time) and samples its private memory. Isolates which stage grows. |
| `vadmem.ps1 -Minutes 30 -Arena 1 -Pattern 1 -Opt 1 -Pre 1` | Runs `crates/sd-vad/examples/vadmem` (Silero VAD alone) with the given ONNX Runtime session options. |

Build first: `cargo build --release -p sdctl` and the examples (`--example leakprobe`, `--example vadmem`). `models fetch` must have run.

The noise file is 60 s of 48 kHz stereo uniform noise at about −70 dBFS RMS (inaudible at normal volume), saved as `target\soak_noise.wav`:

```bash
python -c "import wave,random,struct;w=wave.open('target/soak_noise.wav','wb');w.setnchannels(2);w.setsampwidth(2);w.setframerate(48000);n=48000*60;w.writeframes(struct.pack('<%dh'%(2*n),*[random.randint(-18,18) for _ in range(2*n)]))"
```

Soak recordings contain whatever the microphone hears. Delete them after analysis, and never commit them; `/target*` is ignored.
