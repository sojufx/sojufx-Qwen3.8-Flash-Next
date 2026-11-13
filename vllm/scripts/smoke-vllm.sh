#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${MODEL:-qwen3.8-flash-next}"

curl --fail --silent --show-error "$BASE_URL/v1/models"
printf '\n'
curl --fail --silent --show-error "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply exactly: OK\"}],\"temperature\":0,\"max_tokens\":32,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
printf '\n'
