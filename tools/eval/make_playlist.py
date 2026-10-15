"""Concatenate FLEURS test clips into one 16-bit PCM WAV for a live latency/accuracy run.

    python tools/eval/make_playlist.py [--out bench/data/live_playlist]

Takes the first N clips of each listed language (bench/data/fleurs, 16 kHz float WAV), joins
them with gaps of silence, and writes <out>.wav (16 kHz mono 16-bit, which any player handles)
and <out>.tsv (offset, duration, language, file, reference) so the live transcripts can be
scored against what was played.
"""
import argparse
import csv
import os
import struct
import wave

FLEURS = {"en": "en_us", "hi": "hi_in", "ta": "ta_in", "bn": "bn_in", "zh": "cmn_hans_cn",
          "es": "es_419", "fr": "fr_fr", "te": "te_in", "or": "or_in", "mr": "mr_in"}
RATE = 16000


def read_float_wav(path):
    """Samples of a 16 kHz mono WAV, float32 (format 3) or 16-bit PCM."""
    with open(path, "rb") as f:
        data = f.read()
    pos, fmt, samples = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    tag, channels, rate, _, _, bits = fmt
    if channels != 1 or rate != RATE:
        raise SystemExit(f"{path}: expected 16 kHz mono")
    if tag == 3 and bits == 32:
        return list(struct.unpack(f"<{len(samples) // 4}f", samples))
    if tag == 1 and bits == 16:
        return [s / 32768 for s in struct.unpack(f"<{len(samples) // 2}h", samples)]
    raise SystemExit(f"{path}: unsupported format {tag}/{bits}")


def main():
    ap = argparse.ArgumentParser()
    sd.add_argument("--langs", default="en,hi,ta,bn,zh")
    sd.add_argument("--per-lang", type=int, default=2)
    sd.add_argument("--gap", type=float, default=2.0)
    sd.add_argument("--lead", type=float, default=3.0, help="silence before the first clip")
    sd.add_argument("--out", default="bench/data/live_playlist")
    a = sd.parse_args()
    pcm, rows, t = [0.0] * int(a.lead * RATE), [], a.lead
    for i in range(a.per_lang):
        for lang in a.langs.split(","):
            d = os.path.join("bench/data/fleurs", FLEURS[lang])
            r = list(csv.DictReader(open(os.path.join(d, "manifest.tsv"), encoding="utf-8"),
                                    delimiter="\t"))[i]
            x = read_float_wav(os.path.join(d, r["file"]))
            rows.append((round(t, 3), round(len(x) / RATE, 3), lang, r["file"], r["transcription"]))
            pcm += x + [0.0] * int(a.gap * RATE)
            t += len(x) / RATE + a.gap
    peak = max(1e-9, max(abs(v) for v in pcm))
    scale = min(1.0, 0.9 / peak) * 32767
    with wave.open(a.out + ".wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(struct.pack(f"<{len(pcm)}h", *(int(v * scale) for v in pcm)))
    with open(a.out + ".tsv", "w", encoding="utf-8", newline="") as f:
        f.write("offset_s\tduration_s\tlanguage\tfile\treference\n")
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")
    print(f"{a.out}.wav: {len(rows)} clips, {t:.0f} s")


if __name__ == "__main__":
    main()
