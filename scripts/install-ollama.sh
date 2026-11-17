#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p .runtime/ollama
archive=.runtime/ollama/package.tar.zst
curl --fail --location --retry 2 --continue-at - https://github.com/ollama/ollama/releases/download/v0.40.16/ollama-linux-amd64.tar.zst --output "$archive"
echo "87bfafb2acbc89b2e6a1be0a9043de6aa75fb450696f86bc426e65fc448f788d  $archive" | sha256sum --check
tar --zstd -xf "$archive" -C .runtime/ollama
