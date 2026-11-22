#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Builds the image from public sources: MiaAI-Lab's recipe at a pinned commit (TensorFold v1.2.7 + MiaAI 0001-0005),
# then my patches 0006-0009 from patches/. With IMAGE_ONLY=0 it also downloads the pinned weights.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
RECIPE_COMMIT=b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9
RECIPE_DIR=${RECIPE_DIR:-$ROOT/.recipe}
# Settings come from ./.env (copy .env.example), then the environment.
env_file=$ROOT/.env; [[ -f "$env_file" ]] || env_file=$ROOT/.env.example

if [[ ! -d "$RECIPE_DIR/.git" ]]; then
  git init -q "$RECIPE_DIR"
  git -C "$RECIPE_DIR" remote add origin https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
fi
git -C "$RECIPE_DIR" fetch -q --depth 1 origin "$RECIPE_COMMIT"
git -C "$RECIPE_DIR" checkout -q --force --detach FETCH_HEAD
git -C "$RECIPE_DIR" clean -qfdx -e .env
[[ $(git -C "$RECIPE_DIR" rev-parse HEAD) == "$RECIPE_COMMIT" ]]

# recipe-owner.patch: build the image with patches/ applied after MiaAI's (and check every engine source file
# against the production manifest), two CPU jobs, an FFmpeg mirror, and an image-only mode.
git -C "$RECIPE_DIR" apply "$ROOT/patches/recipe-owner.patch"
mkdir -p "$RECIPE_DIR/owner-patches"
cp "$ROOT"/patches/000[6789]-*.patch "$ROOT/patches/production-zig-sources.sha256" "$RECIPE_DIR/owner-patches/"
cp "$env_file" "$RECIPE_DIR/.env"

cd "$RECIPE_DIR"
export IMAGE_ONLY=${IMAGE_ONLY:-0}
bash scripts/prepare.sh
