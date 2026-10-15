"""Smoke test of the Python binding: version, errors, lists, and (with --record) a short
session without processing, printing its events.

    python bindings/python/tests/smoke.py [--record] [--models DIR]
"""

import argparse
import sys

import sonicdiar as sd

failures = 0


def check(ok: bool, what: str) -> None:
    global failures
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failures += 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--models", default="models")
    p.add_argument("--config", default='{"processing": false}')
    a = p.parse_args()

    print(f"SonicDiar {sd.version()} (python {sys.version.split()[0]})")
    check(sd.__version__ == sd.version(), "__version__")
    try:
        sd.list_devices("sideways")
        check(False, "bad device kind is rejected")
    except sd.SonicDiarError as e:
        check("input" in str(e), f"bad device kind is rejected ({e})")
    try:
        sd.Session({"sytem": "none"})
        check(False, "unknown config field is rejected")
    except sd.SonicDiarError as e:
        check("sytem" in str(e), "unknown config field is rejected")
    models = sd.list_models(a.models)
    check(isinstance(models, list) and len(models) > 0, f"{len(models)} models listed")
    try:
        outputs = sd.list_devices("output")
        print(f"     outputs: {[d['name'] for d in outputs]}")
    except sd.SonicDiarError as e:
        print(f"     (no device list here: {e})")

    if a.record:
        started = 0
        with sd.Session(a.config) as session:
            for event in session.events(seconds=a.seconds):
                print(f"     event: {str(event)[:160]}")
                started += event["type"] == "device_started"
        check(session.summary is not None, f"summary: {str(session.summary)[:200]}")
        check(started > 0, "a device started")
        try:
            session.stop()
            check(False, "stopping twice is an error")
        except sd.SonicDiarError:
            check(True, "stopping twice is an error")
    print("SMOKE FAILED" if failures else "SMOKE PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
