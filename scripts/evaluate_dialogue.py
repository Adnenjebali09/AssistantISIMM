"""Rejouer les mêmes transcriptions sans micro, TTS ni données personnelles."""

import argparse
import json
import re
import time
from threading import Event

from isimm.config import ROOT, Settings
from isimm.llm import Ollama


def evaluate(model, trace=False):
    corpus = json.loads((ROOT / "tests/dialogues.json").read_text())
    bot = Ollama(Settings(llm=model, dialogue_debug=trace), lambda *_: None)
    for index, chat in enumerate(corpus, 1):
        history = []
        for lang, question, *expected in chat:
            bot.trace.begin()
            start = time.monotonic()
            reply = "".join(bot.stream(question, lang, history, Event()))
            yield dict(
                chat=index,
                model=bot.last_model,
                lang=lang,
                question=question,
                reply=reply,
                seconds=round(time.monotonic() - start, 3),
                missing=[p for p in expected if not re.search(p, reply, re.I)],
            )
            history = (
                history
                + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reply},
                ]
            )[-6:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["qwen3.5:2b", "qwen3.5:4b", "auto"], default="qwen3.5:2b"
    )
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    for result in evaluate(args.model, args.trace):
        print(json.dumps(result, ensure_ascii=False), flush=True)
