"""Évaluation en mémoire par signaux reproductibles ou par WAV mono de référence."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample_poly
from isimm.audio import Frontend, Silero, Turn
from isimm.config import ROOT, Settings
from isimm.denoise import DeepFilter
from isimm.stt import Whisper
from isimm.tts import Speech


def normalize(text):
    return " ".join(
        re.findall(
            r"\w+",
            "".join(
                c
                for c in unicodedata.normalize("NFKD", text.casefold())
                if not unicodedata.combining(c)
            ),
        )
    )


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def errors(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    a, b = ref.split(), hyp.split()
    x, y = ref.replace(" ", ""), hyp.replace(" ", "")
    return round(distance(a, b) / max(1, len(a)), 3), round(
        distance(x, y) / max(1, len(x)), 3
    )


def to_48k(samples, rate):
    gcd = math.gcd(rate, 48000)
    return resample_poly(samples, 48000 // gcd, rate // gcd).astype("float32")


def generated(speech, text, lang):
    chunks = list(speech.render(text, lang))
    return np.concatenate([to_48k(samples, rate) for samples, rate in chunks])


def segment(signal, cfg, vad, enhancer):
    front = Frontend(enhancer)
    turn = Turn(cfg.vad)
    vad.reset()
    for i in range(0, len(signal) - 1535, 1536):
        frame = front.process(signal[i : i + 1536])
        if turn.push(frame, vad(frame)):
            return turn.samples()
    return turn.samples()


def scenarios(clean, background):
    # Les positions et SNR sont synthétiques et identiques pour les deux traitements.
    clean = np.pad(clean, (24000, 72000))
    n = len(clean)
    t = np.arange(n) / 48000
    rng = np.random.default_rng(42)
    fan = (0.025 * np.sin(2 * np.pi * 125 * t) + rng.normal(0, 0.014, n)).astype(
        "float32"
    )
    music = (
        0.035 * np.sin(2 * np.pi * 330 * t) + 0.025 * np.sin(2 * np.pi * 495 * t)
    ).astype("float32")
    first, second = (np.resize(voice, n) for voice in background)
    crowd = (first + np.roll(second, 18000)) / 2
    gap = len(clean) // 2
    return {
        "quiet": clean,
        "fan_music": clean + fan + music,
        "two_background": 0.22 * (first + np.roll(second, 27000)),
        "overlap": clean + 0.82 * crowd,
        "near_far": clean + 0.12 * crowd,
        "pause": np.insert(clean, gap, np.zeros(14400)),
        "noise_only": fan + music,
        "silence": np.zeros(n, np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["base", "small"], default="base")
    parser.add_argument("--ids", nargs="+", default=["fr", "en", "ar"])
    parser.add_argument(
        "--scenes", nargs="+", default=["quiet", "fan_music", "noise_only"]
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=["direct", "deepfilter"],
        default=["direct", "deepfilter"],
    )
    parser.add_argument(
        "--wav-dir", type=Path, help="WAV mono int16 ou float par identifiant"
    )
    args = parser.parse_args()
    cfg = replace(Settings.load(), stt=args.model)
    vad = Silero(cfg.models / "silero.onnx")

    def silent(*_):
        pass

    speech, stt = Speech(cfg, silent), Whisper(cfg, silent)
    import onnxruntime as ort

    ort.set_seed(42)
    bg = (
        generated(speech, "Where is the university library?", "en"),
        generated(speech, "أين توجد المكتبة؟", "ar"),
    )
    enhancer = (
        DeepFilter(cfg.models / "deepfilter.so")
        if "deepfilter" in args.filters
        else None
    )
    for row in json.loads((ROOT / "tests/corpus.json").read_text()):
        if row["id"] not in args.ids:
            continue
        if args.wav_dir:
            sr, clip = read(args.wav_dir / (row["id"] + ".wav"))
            if clip.ndim != 1 or clip.dtype not in (np.int16, np.float32, np.float64):
                raise ValueError("WAV mono int16 ou flottant obligatoire")
            clean = to_48k(
                clip.astype("float32") / (32768 if clip.dtype == np.int16 else 1), sr
            )
        else:
            clean = generated(speech, row["text"], row["lang"])
        scenes = scenarios(clean, bg)
        for name in args.scenes:
            for mode in args.filters:
                enhance = enhancer if mode == "deepfilter" else None
                try:
                    t = time.monotonic()
                    audio = segment(scenes[name], cfg, vad, enhance)
                    elapsed = time.monotonic() - t
                finally:
                    if enhance:
                        for _ in range(16):
                            enhance.process(np.zeros(1536, np.float32))
                item = stt.transcribe(audio) if audio is not None else None
                text = item.text if item else ""
                has_question = name not in ("silence", "noise_only", "two_background")
                wer, cer = errors(row["text"], text) if has_question else (None, None)
                print(
                    json.dumps(
                        dict(
                            id=row["id"],
                            scene=name,
                            filter=mode,
                            stt=stt.profile,
                            lang=item.lang if item else None,
                            transcript=text,
                            wer=wer,
                            cer=cer,
                            false_turn=not has_question
                            and bool(item and not item.uncertain),
                            vad_turn=audio is not None,
                            uncertain=item.uncertain if item else None,
                            frontend_s=round(elapsed, 3),
                            stt_s=round(item.seconds, 3) if item else None,
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if enhancer:
        enhancer.close()


if __name__ == "__main__":
    main()
