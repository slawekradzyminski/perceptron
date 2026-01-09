#!/usr/bin/env sh
set -e

ollama serve &
server_pid=$!

# Wait briefly for the server to accept requests.
tries=0
while [ $tries -lt 20 ]; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  tries=$((tries + 1))
  sleep 0.5
done

if [ -n "${OLLAMA_MODEL:-}" ]; then
  attempts=0
  while [ $attempts -lt 5 ]; do
    if ollama pull "$OLLAMA_MODEL"; then
      break
    fi
    attempts=$((attempts + 1))
    sleep 2
  done
fi

wait "$server_pid"
