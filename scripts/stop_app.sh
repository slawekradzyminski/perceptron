#!/usr/bin/env bash
set -euo pipefail

API_PID_FILE="/tmp/perceptron_api.pid"
FRONTEND_PID_FILE="/tmp/perceptron_frontend.pid"

stop_pid() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

stop_pid "$API_PID_FILE"
stop_pid "$FRONTEND_PID_FILE"

echo "Stopped app processes (if any)."
