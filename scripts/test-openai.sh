#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-YOUR_API_KEY}"
MODEL="${MODEL:-ornith}"

curl -sS \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  "$BASE_URL/chat/completions" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say OK and one short sentence.\"}],
    \"max_tokens\": 64,
    \"temperature\": 0.2,
    \"reasoning_effort\": \"none\"
  }" | python3 -m json.tool
