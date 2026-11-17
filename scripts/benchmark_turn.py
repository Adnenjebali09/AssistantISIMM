"""Tour complet avec question synthétique chronométrée et vraie sortie PortAudio."""

import json
import threading
import time

import numpy as np

from isimm.audio import Frontend, Silero, Turn
from isimm.config import Settings
from isimm.conversation import Session
from isimm.denoise import DeepFilter
from isimm.llm import Ollama
from isimm.resources import resources
from isimm.stt import Whisper
from isimm.tts import Speech
from scripts.evaluate import generated


class FixtureInput:
    def __init__(self, signal, cfg):
        self.signal, self.cfg = signal, cfg
        self.last_speech = None

    def listen(self, stop):
        if self.signal is None:
            return None
        df = DeepFilter(self.cfg.models / "deepfilter.so") if self.cfg.denoise else None
        front = Frontend(df)
        vad = Silero(self.cfg.models / "silero.onnx")
        turn = Turn(self.cfg.vad)
        start = time.monotonic()
        try:
            for i in range(0, len(self.signal) - 1535, 1536):
                stop.wait(max(0, start + i / 48000 - time.monotonic()))
                frame = front.process(self.signal[i : i + 1536])
                prob = vad(frame)
                if prob >= self.cfg.vad * 0.7:
                    self.last_speech = time.monotonic()
                if turn.push(frame, prob):
                    return turn.samples()
            return turn.samples()
        finally:
            self.signal = None
            if df:
                df.close()


def main():
    cfg = Settings.load()

    def emit(k, v):
        if k in ("timing", "transcript", "model", "state"):
            print(json.dumps({k: v}, ensure_ascii=False), flush=True)

    voice = Speech(cfg, emit)
    sample = np.pad(
        generated(voice, "Quels sont les départements de l’institut ?", "fr"),
        (9600, 72000),
    )
    stop = threading.Event()
    peaks = {"ram_system_gib": 0.0, "vram_system_mib": 0}

    def monitor():
        while not stop.is_set():
            r = resources()
            peaks["ram_system_gib"] = max(peaks["ram_system_gib"], r["ram"])
            peaks["vram_system_mib"] = max(peaks["vram_system_mib"], r["vram"])
            stop.wait(0.4)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    session = Session(
        FixtureInput(sample, cfg), Whisper(cfg, emit), Ollama(cfg, emit), voice, emit
    )
    try:
        session.run(stop)
    finally:
        stop.set()
        th.join(2)
        print(json.dumps(peaks))


if __name__ == "__main__":
    main()
