#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Serves through MiaAI-Lab's start.sh in the checkout build.sh made, with this repo's .env (or .env.example).
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]; DRY_RUN=1 ./start.sh prints the docker command.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
RECIPE_DIR=${RECIPE_DIR:-$ROOT/.recipe}
[[ -x "$RECIPE_DIR/start.sh" && -d "$RECIPE_DIR/owner-patches" ]] || { echo 'Run ./build.sh first.' >&2; exit 1; }
env_file=$ROOT/.env; [[ -f "$env_file" ]] || env_file=$ROOT/.env.example
cp "$env_file" "$RECIPE_DIR/.env"
# Production rules: four users, the n-gram table read from the SSD, the head8 output layer on.
grep -qx 'PARALLEL=4' "$RECIPE_DIR/.env" && grep -qx 'TF_FLASHNEXT_PLE_IO=pread' "$RECIPE_DIR/.env" && grep -qx 'TF_FLASHNEXT_HEAD8=1' "$RECIPE_DIR/.env" ||
  echo 'Note: .env differs from the measured setup (PARALLEL=4, TF_FLASHNEXT_PLE_IO=pread, TF_FLASHNEXT_HEAD8=1).' >&2
exec "$RECIPE_DIR/start.sh" "$@"
