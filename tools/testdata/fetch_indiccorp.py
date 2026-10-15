"""Fetch a text sample per Indic language from IndicCorp v2 for language-model training.

IndicCorp v2 (AI4Bharat, CC0): https://huggingface.co/datasets/ai4bharat/IndicCorpV2
Each language is one large text file (up to 27 GB), sorted alphabetically. Rather than
downloading it all, this takes CHUNKS byte ranges of CHUNK_MB each, spread evenly through the
file (many small ranges, so the sample covers the whole alphabet rather than a few bands),
from a pinned revision, and keeps whole lines only.

Leakage guard: every line sharing an 8-word run with any FLEURS test or dev transcription of
that language is dropped, so a language model built from this text has not seen the
evaluation sentences.

Writes bench/data/indiccorp/<code>.txt (one sentence per line, NFC) and <code>.json (source,
revision, byte ranges, counts). bench/data/ is not committed.

Usage: python tools/testdata/fetch_indiccorp.py [--chunks N] [--out DIR] [code ...]
       (default: 200 ranges of 2 MB, bench/data/indiccorp, the 11 Indic languages)
"""
import json
import os
import re
import sys
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from fetch_fleurs import transcripts  # noqa: E402

REPO = "ai4bharat/IndicCorpV2"
REVISION = "2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f"
FILES = {
    "bn": "bn.txt", "gu": "gu.txt", "hi": "hi-1.txt", "kn": "kn.txt", "ml": "ml.txt",
    "mr": "mr.txt", "pa": "pa.txt", "ta": "ta.txt", "te": "te.txt", "or": "or.txt",
    "ur": "ur.txt",
}
FLEURS = {
    "bn": "bn_in", "gu": "gu_in", "hi": "hi_in", "kn": "kn_in", "ml": "ml_in", "mr": "mr_in",
    "pa": "pa_in", "ta": "ta_in", "te": "te_in", "or": "or_in", "ur": "ur_pk",
}
CHUNKS = 200
CHUNK_MB = 2
SHINGLE = 8
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "data", "indiccorp")


def words(text: str) -> list:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return text.split()


def shingles(ws: list):
    return {" ".join(ws[i:i + SHINGLE]) for i in range(len(ws) - SHINGLE + 1)}


def url(name: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{name}"


def size_of(name: str) -> int:
    req = urllib.request.Request(url(name), method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def fetch_range(name: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url(name), headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req) as r:
        if r.status != 206:
            raise RuntimeError(f"{name}: range request returned HTTP {r.status}")
        return r.read()


def fetch(code: str, chunks: int = CHUNKS, out_dir: str = OUT) -> None:
    os.makedirs(out_dir, exist_ok=True)
    blocked = set()
    for split in ("test", "dev"):
        for row in transcripts(FLEURS[code], split):
            for text in (row[2], row[3]):
                blocked |= shingles(words(text))
    name = FILES[code]
    total = size_of(name)
    chunk = CHUNK_MB << 20
    step = max(chunk, (total - chunk) // max(1, chunks - 1))
    ranges = [(s, min(total, s + chunk)) for s in range(0, max(1, total - chunk) + 1, step)]
    ranges = ranges[:chunks]
    kept = dropped = 0
    seen = set()
    # Ranges are fetched in parallel (latency, not bandwidth, limits many small requests), 16
    # at a time so memory stays bounded, and processed in file order.
    def batches():
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i in range(0, len(ranges), 16):
                group = ranges[i:i + 16]
                yield from zip(group, pool.map(lambda r: fetch_range(name, r[0], r[1]), group))

    with open(os.path.join(out_dir, f"{code}.txt"), "w", encoding="utf-8", newline="\n") as out:
        for (start, end), body in batches():
            data = body.decode("utf-8", errors="ignore")
            lines = data.split("\n")
            # The first and last lines of a range are usually cut; keep whole lines only.
            lines = lines[1:-1] if start > 0 else lines[:-1]
            for line in lines:
                line = unicodedata.normalize("NFC", line.strip())
                key = hash(line)
                if not line or key in seen:
                    continue
                seen.add(key)
                ws = words(line)
                if len(ws) >= SHINGLE and shingles(ws) & blocked:
                    dropped += 1
                    continue
                out.write(line + "\n")
                kept += 1
    meta = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "license": "CC0-1.0",
        "revision": REVISION,
        "file": name,
        "file_bytes": total,
        "ranges": ranges,
        "lines_kept": kept,
        "lines_dropped_fleurs_overlap": dropped,
    }
    with open(os.path.join(out_dir, f"{code}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"{code}: {kept} lines kept, {dropped} dropped (FLEURS overlap), "
          f"{len(ranges)} ranges of {CHUNK_MB} MB from {total / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    chunks, out_dir = CHUNKS, OUT
    while args and args[0].startswith("--"):
        opt, val = args[0], args[1]
        args = args[2:]
        if opt == "--chunks":
            chunks = int(val)
        elif opt == "--out":
            out_dir = val
        else:
            raise SystemExit(f"unknown option {opt}")
    for c in args or list(FILES):
        if not re.fullmatch(r"[a-z]{2}", c) or c not in FILES:
            raise SystemExit(f"unknown language code {c}")
        fetch(c, chunks, out_dir)
