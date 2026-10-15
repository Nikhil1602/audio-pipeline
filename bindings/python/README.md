# SonicDiar (Python)

Meeting audio capture: microphone and system audio on Windows, macOS and Linux, with echo
cancellation, voice activity detection, live transcription in 15 languages and speaker labels.

```python
import sonicdiar as sd

sd.fetch_models("models", "core")      # once: voice detection + ONNX Runtime (~50 MB)
sd.fetch_models("models", only="whisper-small")  # for transcription (~490 MB)

config = {"transcription": {"whisper_model": "ggml-small.bin"}}
with sd.Session(config) as session:
    for event in session.events(seconds=30):
        if event["type"] == "final":
            print(event["source"], event["speaker"], event["text"])
print(session.summary)
```

See the repository's `docs/sdk.md` for the configuration, every event and the platform notes.
