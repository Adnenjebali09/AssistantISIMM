#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    echo "Exécutez d’abord bash scripts/install.sh." >&2
    exit 1
fi
url=http://127.0.0.1:11434/api/version
online() { curl --noproxy '*' -fsS --max-time 1 "$url" >/dev/null 2>&1; }
if online; then
    exec .venv/bin/python -m isimm "$@"
fi
scripts/ollama-local serve &
server_pid=$!
cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in {1..50}; do
    if online; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep .1
done
.venv/bin/python -m isimm "$@"
