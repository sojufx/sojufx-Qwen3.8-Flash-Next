#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-Flash-Next-GGUF}"
QUANT_SUBDIR="${QUANT_SUBDIR:-UD-IQ3_XXS}"
LOCAL_DIR="${LOCAL_DIR:-/opt/huggingface/models/Qwen3.8-Flash-Next-GGUF-UD-IQ3_XXS}"

huggingface-cli download "$MODEL_ID" \
  --include "${QUANT_SUBDIR}/*" \
  --local-dir "$LOCAL_DIR"

echo "Downloaded to: $LOCAL_DIR"
echo "Serve entrypoint:"
echo "$LOCAL_DIR/$QUANT_SUBDIR/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
