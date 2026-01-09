#!/usr/bin/env bash
set -euo pipefail

PRIMARY_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
EXTRA_MODELS="${OLLAMA_EXTRA_MODELS:-}"

ollama serve &
SERVER_PID=$!

until ollama list >/dev/null 2>&1; do
  sleep 1
done

for MODEL in ${PRIMARY_MODEL} ${EXTRA_MODELS}; do
  [ -z "$MODEL" ] && continue
  echo "Pulling $MODEL ..."
  ollama pull "$MODEL"
done

kill -SIGINT "$SERVER_PID" && wait "$SERVER_PID"
