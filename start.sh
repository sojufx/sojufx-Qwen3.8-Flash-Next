#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
# Parse KEY=value as data; environment takes precedence.
load_env() {
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[1]}; value=${BASH_REMATCH[2]}
    [[ -n "${!key+set}" ]] || export "$key=$value"
  done < "$1"
}
[[ ! -f "$ROOT/.env" ]] || load_env "$ROOT/.env"
load_env "$ROOT/.env.example"
: "${HF_CACHE:=${HF_HOME:-$HOME/.cache/huggingface}}"
: "${KERNEL_CACHE:=$HOME/.cache/tensorfold-recipe}"
case "$ABLIT" in 0|1) ;; *) echo 'ABLIT must be 0 or 1' >&2; exit 1;; esac
# Keep the engine's SSD table reader and production parallelism fixed.
[[ "$TF_FLASHNEXT_GATHER_READ" == pread && "$TF_FLASHNEXT_HEAD8" == 1 ]] || { echo 'This recipe requires pread and HEAD8=1.' >&2; exit 1; }
stock=b9ce18a6887e18c089fdea8509a33e63429a3300
overlay=c44f5b92ee3d307b82aab1c55f4cc4ae1d440344
model_sub="hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/$stock"
[[ "$ABLIT" == 0 ]] || model_sub="tensorfold-qwen38/ablit-${stock:0:8}-${overlay:0:8}"
# --prepare downloads only pinned snapshots and builds the optional symlink view.
# Existing HF authentication is passed by environment, never stored in an image.
if [[ "${1:-}" == --prepare ]]; then
  mkdir -p "$HF_CACHE" "$KERNEL_CACHE"
  recipe=$(mktemp -d); trap 'rm -rf "$recipe"' EXIT
  git -C "$recipe" init -q
  git -C "$recipe" remote add origin https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
  git -C "$recipe" fetch --depth 1 origin 20e15e41f2f514a0d287710f4665ee1b5009ba41
  git -C "$recipe" checkout -q --detach FETCH_HEAD
  docker run --rm --entrypoint python3 ${HF_TOKEN:+-e HF_TOKEN} -e ABLIT="$ABLIT" -v "$HF_CACHE:/root/.cache/huggingface" -v "$recipe/scripts:/recipe:ro" "$IMAGE" -c '
import os,subprocess
from huggingface_hub import snapshot_download
stock=snapshot_download("nvidia/Qwen3.8-Flash-Next-FP4-Local",revision="b9ce18a6887e18c089fdea8509a33e63429a3300")
if os.environ["ABLIT"] == "1":
 overlay=snapshot_download("drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47",revision="c44f5b92ee3d307b82aab1c55f4cc4ae1d440344",allow_patterns=["fp4-local/house-qsa-oproj-mxfp8-L3-47.safetensors"])
 subprocess.run(["python3","-I","/recipe/ablit_view.py",stock,overlay+"/fp4-local/house-qsa-oproj-mxfp8-L3-47.safetensors","/root/.cache/huggingface/tensorfold-qwen38/ablit-b9ce18a6-c44f5b92"],check=True)
'
  exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: ./start.sh [--prepare]; DRY_RUN=1 prints the serve command.' >&2; exit 1; }
cmd=(docker run -d --restart unless-stopped --name "$CONTAINER_NAME" --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864)
while IFS='=' read -r name value; do
  [[ "$name" == TF_FLASHNEXT_* || "$name" == TENSORFOLD_* || "$name" == HF_HUB_OFFLINE ]] || continue
  cmd+=(-e "$name=$value")
done < <(env | sort)
cmd+=(-v "$HF_CACHE:/root/.cache/huggingface:ro" -v "$KERNEL_CACHE:/cache" "$IMAGE" tensorfold-native serve "/root/.cache/huggingface/$model_sub" --name ornith --host "$HOST" --port "$PORT" --context 262144 --parallel 4 --max-tokens 32768 --temperature 0 --top-p 0.8 --top-k 20 --no-thinking --kv-cache fp8 --vision --vision-max-images 50 --vision-max-videos 4 --vision-image-tokens 16384 --thinking-budget 8192)
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ -f "$HF_CACHE/$model_sub/config.json" ]] || { echo 'Prepare weights first: ./start.sh --prepare' >&2; exit 1; }
mkdir -p "$KERNEL_CACHE"
"${cmd[@]}"
