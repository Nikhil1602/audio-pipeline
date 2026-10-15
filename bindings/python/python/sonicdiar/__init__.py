"""SonicDiar: meeting audio capture.

Microphone and system audio (what the computer plays) on Windows, macOS and Linux, with echo
cancellation, voice activity detection, live transcription in 15 languages and speaker labels.

    import sonicdiar as sd

    sd.fetch_models("models", "core")            # once: voice detection + ONNX Runtime
    with sd.Session({"transcription": None}) as session:
        for event in session.events(seconds=10):
            print(event["type"], event)
    print(session.summary)

Events are dicts with a "type" key ("device_started", "speech", "partial", "final",
"speaker_revision", ...); see docs/sdk.md for every field.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from . import _native
from ._native import SonicDiarError

__all__ = [
    "SonicDiarError",
    "Session",
    "version",
    "list_devices",
    "list_models",
    "fetch_models",
]

__version__ = _native.version()


def version() -> str:
    """The library version."""
    return _native.version()


def list_devices(kind: str = "input") -> List[Dict[str, Any]]:
    """Audio devices of kind "input" (microphones) or "output" (system audio)."""
    return json.loads(_native.list_devices(kind))


def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """Every known model with its size, licence and whether it is present."""
    return json.loads(_native.list_models(str(models_dir)))


def fetch_models(
    models_dir: str = "models",
    group: str = "core",
    only: Optional[str] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Downloads and verifies a model group ("core", "asr", "diar", "indic", "all") or one
    model (`only`) into `models_dir`. `progress(step)` receives a dict per step."""
    cb = (lambda s: progress(json.loads(s))) if progress else None
    _native.fetch_models(str(models_dir), group, only, cb)


class Session:
    """A recording session. `config` is a dict (or JSON string); `{}` records the default
    microphone and system audio with processing on and no transcription."""

    def __init__(self, config: Union[Dict[str, Any], str, None] = None) -> None:
        if config is None:
            config = {}
        text = config if isinstance(config, str) else json.dumps(config)
        self._native = _native.NativeSession(text)
        self.summary: Optional[Dict[str, Any]] = None

    @property
    def journal_dir(self) -> str:
        """The crash-safe recording on disk."""
        return self._native.journal_dir()

    @property
    def running(self) -> bool:
        return self._native.is_running()

    def next_event(self, timeout: float = 0.2) -> Optional[Dict[str, Any]]:
        """The next event, waiting up to `timeout` seconds; None if none arrived."""
        s = self._native.next_event(timeout)
        return None if s is None else json.loads(s)

    def events(self, seconds: Optional[float] = None, timeout: float = 0.2) -> Iterator[Dict[str, Any]]:
        """Yields events, for `seconds` (then stops the session) or until the session is
        stopped from elsewhere. After a stop, the remaining events are yielded too."""
        until = None if seconds is None else time.monotonic() + seconds
        while self.running and (until is None or time.monotonic() < until):
            event = self.next_event(timeout)
            if event is not None:
                yield event
        if self.running:
            self.stop()
        while True:
            event = self.next_event(0)
            if event is None:
                return
            yield event

    def set_device(self, source: str, device: str) -> None:
        """Switches "mic" or "system" to a device id (from list_devices) or "default"."""
        self._native.set_device(source, device)

    def pause(self) -> None:
        """Pauses: audio is replaced by silence (nothing recorded or transcribed) until
        resume(); event times keep counting from the start."""
        self._native.pause()

    def resume(self) -> None:
        """Resumes after pause()."""
        self._native.resume()

    def stop(self) -> Dict[str, Any]:
        """Stops capture and finishes transcribing queued speech (whose events stay readable
        with next_event). Returns the summary."""
        self.summary = json.loads(self._native.stop())
        return self.summary

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.running:
            self.stop()
