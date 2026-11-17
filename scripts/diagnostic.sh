#!/usr/bin/env bash
# Les outils absents sont signalés, les autres diagnostics continuent.
cd "$(dirname "$0")/.." || exit 1
nvidia-smi
free -h
lscpu | head -22
python3 --version
ldconfig -p | grep -E 'libcuda|libcudnn|libcublas|libportaudio|libasound'
command -v pactl && pactl list short sources
command -v wpctl && wpctl status
command -v arecord && arecord -l
if [[ -x .venv/bin/python ]]; then
.venv/bin/python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
for name in ('mediapipe','onnxruntime','ctranslate2','kokoro-onnx','piper-tts','faster-whisper','pyside6-essentials','sounddevice'):
    try: print(name,version(name))
    except PackageNotFoundError: print(name,'absent')
import sounddevice as sd
print(sd.query_devices())
PY
fi
