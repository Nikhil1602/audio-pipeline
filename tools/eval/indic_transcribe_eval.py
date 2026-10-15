"""Benchmark Indic-Transcribe-core (Bodhan AI, Indic Open Model License) on the FLEURS
subsets, for comparison with IndicConformer. Evaluation only: the model is not shipped.

Runs the model's own code from its downloaded repository, offline, in a separate Python
environment (PyTorch + transformers):

    D:\\eval-venv\\Scripts\\python.exe tools/eval/indic_transcribe_eval.py \\
        --model D:\\eval-models\\indic-transcribe-core --out target/eval/indic-transcribe.tsv

Writes the same TSV columns as `sdctl asr eval` (language, file, detected, errors,
ref_len, reference, hypothesis), so `sdctl asr score` rescores both models identically.
`errors`/`ref_len` are left at 0 here; `asr score` recomputes them. Each clip is
transcribed with the true language given (`lang=`), and the model's own language
identification over the 11 Indic product languages is recorded in `detected` and
`probs_indictranscribe` from the same encoder pass.
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

INDIC = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te", "or", "ur"]
FLEURS = {
    "bn": "bn_in", "gu": "gu_in", "hi": "hi_in", "kn": "kn_in", "ml": "ml_in", "mr": "mr_in",
    "pa": "pa_in", "ta": "ta_in", "te": "te_in", "or": "or_in", "ur": "ur_pk",
}
SHORT_PASS_MAX_S = 30.0  # the model's training length; longer clips use its chunked helper


def main():
    ap = argparse.ArgumentParser()
    sd.add_argument("--model", required=True, help="downloaded repository directory")
    sd.add_argument("--data", default="bench/data/fleurs")
    sd.add_argument("--langs", default=",".join(INDIC))
    sd.add_argument("--limit", type=int, default=100)
    sd.add_argument("--mode", default="native", help="native | mixed (digits, Latin loanwords)")
    sd.add_argument("--device", default=None)
    sd.add_argument("--out", required=True)
    a = sd.parse_args()

    sys.path.insert(0, a.model)
    import soundfile as sf
    import torch
    from indic_transcribe import IndicTranscribe
    from long_form import transcribe_long

    t0 = time.time()
    asr = IndicTranscribe.from_pretrained(a.model, device=a.device)
    print(f"loaded in {time.time() - t0:.1f} s on {asr.device}; "
          f"torch {torch.__version__}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        def writerow(cells):
            f.write("\t".join(str(c) for c in cells) + "\n")

        writerow(["language", "file", "detected", "errors", "ref_len", "reference",
                    "hypothesis", "probs_indictranscribe", "seconds", "compute_s"])
        print(f"{'language':<10} {'n':>4} {'LID':>6} {'rtf':>6}", flush=True)
        total_audio = total_compute = 0.0
        for lang in a.langs.split(","):
            d = os.path.join(a.data, FLEURS[lang])
            rows = list(csv.DictReader(open(os.path.join(d, "manifest.tsv"), encoding="utf-8"),
                                       delimiter="\t"))[: a.limit]
            ok = 0
            audio_s = compute_s = 0.0
            for r in rows:
                wav, sr = sf.read(os.path.join(d, r["file"]), dtype="float32")
                secs = len(wav) / sr
                t = time.time()
                if secs <= SHORT_PASS_MAX_S:
                    # One encoder pass serves both identification and transcription.
                    text, info = asr.transcribe(wav, lang=lang, mode=a.mode, return_lid=True,
                                                allowed_langs=INDIC)
                    lid = info["topk"]
                else:
                    lid = asr.identify(wav, topk=len(INDIC), allowed_langs=INDIC)
                    text = transcribe_long(asr, wav, lang=lang, mode=a.mode)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - t
                audio_s += secs
                compute_s += dt
                detected = lid[0][0]
                ok += detected == lang
                probs = ",".join(f"{l}:{p:.4f}" for l, p in lid)
                clean = lambda s: s.replace("\t", " ").replace("\n", " ")
                writerow([lang, r["file"], detected, 0, 0, clean(r["transcription"]),
                            clean(text), probs, f"{secs:.2f}", f"{dt:.3f}"])
            f.flush()
            total_audio += audio_s
            total_compute += compute_s
            print(f"{lang:<10} {len(rows):>4} {100 * ok / len(rows):5.1f}% "
                  f"{compute_s / audio_s:6.3f}", flush=True)
        print(f"overall real-time factor {total_compute / total_audio:.3f} "
              f"({total_audio:.0f} s of audio in {total_compute:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
