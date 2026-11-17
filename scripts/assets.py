"""Téléchargement explicite des modèles. Aucun accès réseau depuis les moteurs locaux."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "models"
HF = "https://huggingface.co/"
GH = "https://github.com/"
KOKORO = GH + "thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
PIPER = HF + "rhasspy/piper-voices/resolve/main/"
GROUPS = {
    "audio": {
        "silero.onnx": "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx",
        "deepfilter.so": GH
        + "Rikorose/DeepFilterNet/releases/download/v0.5.6/libdeep_filter_ladspa-0.5.6-x86_64-unknown-linux-gnu.so",
    },
    "camera": {
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    },
    "kokoro": {
        name: KOKORO + name for name in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
    },
}
for size in ("small", "base"):
    GROUPS[size] = {
        f"{size}/{name}": HF + f"Systran/faster-whisper-{size}/resolve/main/{name}"
        for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
    }
for lang, voice in [("fr_FR", "siwis"), ("ar_JO", "kareem"), ("en_US", "lessac")]:
    stem = f"{lang}-{voice}-medium"
    GROUPS[lang] = {
        stem + ext: PIPER + f"{lang[:2]}/{lang}/{voice}/medium/{stem}{ext}"
        for ext in (".onnx", ".onnx.json")
    }


def download(groups):
    manifest = ROOT.parent / "scripts" / "asset-hashes.json"
    hashes = json.loads(manifest.read_text()) if manifest.exists() else {}

    def digest(path):
        with path.open("rb") as src:
            return hashlib.file_digest(src, "sha256").hexdigest()

    for group in groups:
        for name, url in GROUPS[group].items():
            path = ROOT / name
            if path.is_file() and (name not in hashes or digest(path) == hashes[name]):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".part")
            print("Téléchargement", name, flush=True)
            with urllib.request.urlopen(url, timeout=60) as src, temp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
                if src.headers.get("Content-Length") and dst.tell() != int(
                    src.headers["Content-Length"]
                ):
                    raise IOError("Téléchargement incomplet")
            if name in hashes and digest(temp) != hashes[name]:
                raise IOError(f"Empreinte SHA-256 incorrecte pour {name}")
            temp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", choices=GROUPS.keys())
    download(parser.parse_args().groups)
