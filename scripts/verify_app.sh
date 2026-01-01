#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
API_PID_FILE="/tmp/perceptron_api.pid"
FRONTEND_PID_FILE="/tmp/perceptron_frontend.pid"
API_LOG="/tmp/perceptron_api.log"
FE_LOG="/tmp/perceptron_frontend.log"
API_URL="http://127.0.0.1:8000/state"
FE_URL="http://127.0.0.1:5173/"

stop_pid() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 0.1
      done
    fi
    rm -f "$pid_file"
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    return 1
  fi
}

start_api() {
  stop_pid "$API_PID_FILE"
  if ! command -v poetry >/dev/null 2>&1; then
    echo "Poetry is required to run the backend. Install it and retry." >&2
    exit 1
  fi
  (cd "$ROOT_DIR" && poetry run perceptron-api >"$API_LOG" 2>&1 & echo $! >"$API_PID_FILE")
}

start_frontend() {
  stop_pid "$FRONTEND_PID_FILE"
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required to run the frontend." >&2
    exit 1
  fi
  (cd "$FRONTEND_DIR" && npm run dev -- --host 127.0.0.1 --port 5173 >"$FE_LOG" 2>&1 & echo $! >"$FRONTEND_PID_FILE")
}

wait_for_url() {
  local url="$1"
  local retries=50
  for _ in $(seq 1 "$retries"); do
    if curl -s "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

main() {
  start_api
  start_frontend

  if ! wait_for_url "$API_URL"; then
    echo "API failed to start. See $API_LOG" >&2
    exit 1
  fi

  if ! wait_for_url "$FE_URL"; then
    echo "Frontend failed to start. See $FE_LOG" >&2
    exit 1
  fi

  api_json="$(curl -s "$API_URL")"
  fe_html="$(curl -s "$FE_URL")"

  if ! echo "$api_json" | grep -q '"dataset"'; then
    echo "API response missing dataset field" >&2
    exit 1
  fi

  if ! echo "$fe_html" | grep -q "Perceptron Visual Lab"; then
    echo "Frontend HTML missing title" >&2
    exit 1
  fi

  echo "OK: API and frontend are running"
  echo "API PID: $(cat "$API_PID_FILE")"
  echo "Frontend PID: $(cat "$FRONTEND_PID_FILE")"
}

main "$@"
