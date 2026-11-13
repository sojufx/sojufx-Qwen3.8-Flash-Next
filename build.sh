#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
RECIPE_COMMIT=20e15e41f2f514a0d287710f4665ee1b5009ba41
BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
IMAGE=${IMAGE:-sojufx-tensorfold:recipe-20261113}
BUILD_DIR=${BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tensorfold-recipe.XXXXXX")}
mkdir -p "$BUILD_DIR"
BUILD_DIR=$(cd "$BUILD_DIR" && pwd)
[[ ! -e "$BUILD_DIR/upstream" ]] || { echo 'Use a fresh BUILD_DIR.' >&2; exit 1; }
started=$SECONDS
# No private images, prebuilt engine downloads, or model weights enter this build.
git init -q "$BUILD_DIR/upstream"
git -C "$BUILD_DIR/upstream" remote add origin https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
git -C "$BUILD_DIR/upstream" fetch --depth 1 origin "$RECIPE_COMMIT"
git -C "$BUILD_DIR/upstream" checkout -q --detach FETCH_HEAD
[[ $(git -C "$BUILD_DIR/upstream" rev-parse HEAD) == "$RECIPE_COMMIT" ]]
cd "$BUILD_DIR/upstream"
# Ignore caller's upstream overrides: all inputs below come from the pinned recipe.
unset TF_REF TF_REPO ZIG_VERSION ZIG_SHA256 MODEL_ID MODEL_REVISION CHECKPOINT
source scripts/config.sh
BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
python3 - "$BUILD_DIR/Upstream.Dockerfile" <<'PY'
import pathlib,sys
s=pathlib.Path('scripts/prepare.sh').read_text()
s=s.split("<<'DOCKERFILE'\n",1)[1].split('\nDOCKERFILE',1)[0]
# Restrict CPU/memory pressure while a production engine is serving.
s=s.replace('-j"$(nproc)"','-j2')
pathlib.Path(sys.argv[1]).write_text(s+'\n')
PY
args=()
for name in BASE_IMAGE TF_REPO TF_REF ZIG_VERSION ZIG_SHA256 LIBJPEG_TURBO_VERSION LIBJPEG_TURBO_SHA256 LIBWEBP_VERSION LIBWEBP_SHA256 TF_RELEASE_VERSION TF_RELEASE_AARCH64_SHA256; do
  args+=(--build-arg "$name=${!name}")
done
# Unique tags identify only images built in this invocation.
job="tensorfold-recipe-build:$(date +%s)-$$"
cleanup() { docker image rm "$job-public-builder" "$job-public-runtime" "$job-local-builder" >/dev/null 2>&1 || true; }
trap cleanup EXIT
export DOCKER_BUILDKIT=1
docker build --pull --no-cache "${args[@]}" --build-arg PATCHES_HASH="$(image_hash)" --target build -t "$job-public-builder" -f "$BUILD_DIR/Upstream.Dockerfile" .
docker build "${args[@]}" --build-arg PATCHES_HASH="$(image_hash)" -t "$job-public-runtime" -f "$BUILD_DIR/Upstream.Dockerfile" .
local_args=(--build-arg "PUBLIC_BUILDER=$job-public-builder" --build-arg "PUBLIC_RUNTIME=$job-public-runtime")
# The stages above are freshly built from the pinned public source/base, never production images.
docker build "${local_args[@]}" --target build -t "$job-local-builder" -f "$ROOT/Dockerfile" "$ROOT"
docker run --rm --network none --cap-drop DAC_OVERRIDE --cap-drop DAC_READ_SEARCH --entrypoint /bin/bash "$job-local-builder" -c 'cd /tensorfold && /opt/zig/zig build -j2 test-cuda-host --summary all' > "$BUILD_DIR/host-tests.log" 2>&1
cat "$BUILD_DIR/host-tests.log"
docker build "${local_args[@]}" -t "$IMAGE" -f "$ROOT/Dockerfile" "$ROOT"
docker image inspect "$IMAGE" --format '{{.Id}}' > "$BUILD_DIR/image-id"
printf 'Build completed in %s seconds. Image: %s. Receipts: %s\n' "$((SECONDS-started))" "$IMAGE" "$BUILD_DIR"
