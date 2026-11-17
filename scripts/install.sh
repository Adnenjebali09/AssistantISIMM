#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install 'uv==0.13.8'
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python" UV_CACHE_DIR="$PWD/.cache/uv"
.bootstrap/bin/uv venv --allow-existing --python 3.11 .venv
if [[ -f requirements.lock ]]; then
    .bootstrap/bin/uv pip install -r requirements.lock
else
    .bootstrap/bin/uv pip install -r requirements.txt
fi
