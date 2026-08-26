#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-build-qwen4exp}"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"

# This PR/branch was the working qwen4exp path during testing.
# If it has been merged, you may be able to use llama.cpp master instead.
git fetch origin pull/27742/head:qwen4exp-pr-27742
git checkout qwen4exp-pr-27742

cmake -B "$BUILD_DIR" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build "$BUILD_DIR" -j --target llama-server llama-cli

"$LLAMA_DIR/$BUILD_DIR/bin/llama-server" --version
