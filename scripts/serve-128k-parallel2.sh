#!/usr/bin/env bash
set -euo pipefail

API_KEY="${API_KEY:-YOUR_API_KEY}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ALIAS="${ALIAS:-ornith}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-qwen4exp/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/opt/huggingface/models/Qwen3.8-Flash-Next-GGUF-UD-IQ3_XXS/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}"

exec "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --alias "$ALIAS" \
  --api-key "$API_KEY" \
  --ctx-size 131072 \
  --parallel 2 \
  --cont-batching \
  --cache-prompt \
  --batch-size 1024 \
  --ubatch-size 256 \
  --reasoning off
