"""Fetch a small FLEURS subset per language for ASR / LID evaluation.

FLEURS (Google, CC-BY 4.0): https://huggingface.co/datasets/google/fleurs
Streams each language's <split>.tar.gz and stops after N utterances, so only part of each
archive is downloaded. Writes bench/data/fleurs[-<split>]/<config>/<id>.wav and manifest.tsv
(file, transcription, raw transcription). The test split goes to bench/data/fleurs (used for
reporting); dev goes to bench/data/fleurs-dev (used for tuning, never for reporting).

Usage: python tools/testdata/fetch_fleurs.py [N] [--split test|dev] [config ...]
"""
import csv
import os
import sys
import tarfile
import urllib.request

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
CONFIGS = ["en_us", "cmn_hans_cn", "es_419", "fr_fr", "bn_in", "gu_in", "hi_in", "kn_in",
           "ml_in", "mr_in", "pa_in", "ta_in", "te_in", "or_in", "ur_pk"]
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "data")


def transcripts(config: str, split: str) -> list:
    """All rows of a split's TSV: id, file name, raw transcription, normalized
    transcription, phonemes, num samples, gender."""
    with urllib.request.urlopen(f"{BASE}/{config}/{split}.tsv") as r:
        return list(csv.reader(r.read().decode("utf-8").splitlines(), delimiter="\t",
                               quoting=csv.QUOTE_NONE))


def fetch(config: str, n: int, split: str) -> None:
    out = os.path.join(DATA, "fleurs" if split == "test" else f"fleurs-{split}", config)
    os.makedirs(out, exist_ok=True)
    by_file = {row[1]: row for row in transcripts(config, split)}
    kept = []
    with urllib.request.urlopen(f"{BASE}/{config}/audio/{split}.tar.gz") as r:
        with tarfile.open(fileobj=r, mode="r|gz") as tar:
            for m in tar:
                name = os.path.basename(m.name)
                if not m.isfile() or name not in by_file:
                    continue
                data = tar.extractfile(m).read()
                with open(os.path.join(out, name), "wb") as f:
                    f.write(data)
                kept.append(by_file[name])
                if len(kept) >= n:
                    break
    with open(os.path.join(out, "manifest.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["file", "transcription", "raw_transcription"])
        for row in kept:
            w.writerow([row[1], row[3], row[2]])
    print(f"{config} {split}: {len(kept)} utterances")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args.pop(0)) if args and args[0].isdigit() else 100
    split = "test"
    if args[:1] == ["--split"]:
        split = args[1]
        args = args[2:]
    for c in args or CONFIGS:
        fetch(c, n, split)
