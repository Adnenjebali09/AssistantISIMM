"""Mesures locales. Les essais audio et vidéo ne conservent aucun échantillon."""

import argparse
import json
import multiprocessing as mp
import queue
import time
from threading import Event

from isimm.config import Settings


def camera(cfg, duration=12):
    from isimm.vision import camera_worker

    ctx = mp.get_context("spawn")
    frames, events = ctx.Queue(1), ctx.Queue()
    end, armed = ctx.Event(), ctx.Event()
    armed.set()
    worker = ctx.Process(
        target=camera_worker, args=(cfg, frames, events, end, armed), daemon=True
    )
    worker.start()
    received = 0
    max_hold = 0.0
    labels = {}
    triggers = []
    start = time.monotonic()
    first = None
    try:
        while time.monotonic() - start < duration:
            try:
                jpeg, label, progress = frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if jpeg:
                received += 1
                labels[label] = labels.get(label, 0) + 1
                max_hold = max(max_hold, progress * 5)
                try:
                    kind, value = events.get_nowait()
                    if kind == "gaze":
                        triggers.append(value)
                except queue.Empty:
                    pass
                if first is None:
                    first = time.monotonic() - start
        print(
            json.dumps(
                dict(
                    decoded_frames=received,
                    first_preview_seconds=first,
                    max_hold=max_hold,
                    activations_s=triggers,
                    frames_by_status=labels,
                )
            )
        )
        if not received:
            raise RuntimeError("Aucune image décodée")
    finally:
        end.set()
        worker.join(3)
        if worker.is_alive():
            worker.terminate()
        worker.join(1)


def microphone(cfg):
    import numpy as np
    import sounddevice as sd
    from isimm.audio import Frontend, Silero
    from isimm.denoise import DeepFilter

    df = DeepFilter(cfg.models / "deepfilter.so")
    try:
        vad, front = Silero(cfg.models / "silero.onnx"), Frontend(df)
        dev = int(cfg.mic) if cfg.mic.isdigit() else (cfg.mic or None)
        n = overflows = activity = 0
        energy = 0.0
        with sd.InputStream(
            channels=1, samplerate=48000, dtype="float32", device=dev, blocksize=1536
        ) as s:
            for _ in range(94):
                data, overflow = s.read(1536)
                p = vad(front.process(data[:, 0]))
                n += len(data)
                overflows += int(overflow)
                activity += int(p >= cfg.vad)
                energy += float(np.sum(data**2))
        print(
            json.dumps(
                dict(
                    seconds=n / 48000,
                    rms=(energy / n) ** 0.5,
                    vad_positive_frames=activity,
                    overflows=overflows,
                )
            )
        )
    finally:
        df.close()


def tts(cfg):
    import numpy as np
    from isimm.tts import Speech

    backend = Speech(cfg, lambda k, v: print(v, flush=True))
    for lang, voice, text in [
        ("fr", "ff_siwis", "Quels sont les départements de l’institut ?"),
        ("fr", "piper", "Quels sont les départements de l’institut ?"),
        ("ar", "piper", "أين أجد جدول الدروس؟"),
    ]:
        cfg.voice = voice
        t = time.monotonic()
        for i in range(2):
            t = time.monotonic()
            chunks = list(backend.render(text, lang))
            print(
                json.dumps(
                    dict(
                        lang=lang,
                        voice=voice,
                        run=i + 1,
                        generation_s=time.monotonic() - t,
                        audio_s=sum(len(a) / sr for a, sr in chunks),
                        peak=max(float(np.max(np.abs(a))) for a, sr in chunks),
                    )
                ),
                flush=True,
            )


def llm(cfg, model):
    from isimm.llm import Ollama

    cfg.llm = model
    engine = Ollama(cfg, lambda *_: None)
    for lang, q in [
        ("fr", "Quels sont les départements de l’ISIMM ?"),
        ("en", "What time is my exam tomorrow?"),
        ("ar", "ما هي أقسام المعهد؟"),
    ]:
        t = time.monotonic()
        first = None
        answer = ""
        for tok in engine.stream(q, lang, [], Event()):
            if first is None:
                first = time.monotonic() - t
            answer += tok
        print(
            json.dumps(
                dict(
                    model=model,
                    lang=lang,
                    ttft_s=engine.first_token_at - t,
                    spoken_text_s=first,
                    total_s=time.monotonic() - t,
                    response=answer,
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("test", choices=["camera", "microphone", "tts", "llm"])
    p.add_argument(
        "--model", choices=["qwen3.5:4b", "qwen3.5:2b"], default="qwen3.5:2b"
    )
    p.add_argument("--duration", type=int, default=12)
    args = p.parse_args()
    cfg = Settings.load()
    if args.test == "llm":
        llm(cfg, args.model)
    elif args.test == "camera":
        camera(cfg, args.duration)
    else:
        {"microphone": microphone, "tts": tts}[args.test](cfg)
