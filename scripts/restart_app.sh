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
OLLAMA_CONTAINER="ollama-llama"
OLLAMA_IMAGE="ollama/ollama:0.13.5"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
USE_DOCKER=false

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

stop_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    set +e
    local pids
    pids=$(lsof -ti tcp:"$port")
    set -e
    if [[ -n "$pids" ]]; then
      echo "$pids" | xargs -r kill
    fi
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
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

require_cmd poetry
require_cmd npm
require_cmd curl

for arg in "$@"; do
  case "$arg" in
    --docker)
      USE_DOCKER=true
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--docker]" >&2
      exit 1
      ;;
  esac
done

start_ollama() {
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker before running this script." >&2
    exit 1
  fi

  if ! docker image inspect "$OLLAMA_IMAGE" >/dev/null 2>&1; then
    docker pull "$OLLAMA_IMAGE"
  fi

  if docker ps -a --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
    local current_image
    current_image="$(docker inspect -f '{{.Config.Image}}' "$OLLAMA_CONTAINER")"
    if [[ "$current_image" != "$OLLAMA_IMAGE" ]]; then
      docker rm -f "$OLLAMA_CONTAINER" >/dev/null
    fi
  fi

  if ! docker ps -a --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
    docker run -d --name "$OLLAMA_CONTAINER" \
      -p 11434:11434 \
      -v ollama-llama:/root/.ollama \
      "$OLLAMA_IMAGE"
  elif ! docker ps --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
    docker start "$OLLAMA_CONTAINER" >/dev/null
  fi

  docker exec "$OLLAMA_CONTAINER" ollama pull "$OLLAMA_MODEL" >/dev/null
}

stop_pid "$API_PID_FILE"
stop_pid "$FRONTEND_PID_FILE"
stop_port 8000
stop_port 5173

if $USE_DOCKER; then
  require_cmd docker
  start_ollama
else
  GD_TOKEN_SOURCE=${GD_TOKEN_SOURCE:-none}
fi

(
  cd "$ROOT_DIR"
  GD_TOKEN_SOURCE=${GD_TOKEN_SOURCE:-none} \
  OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
  OLLAMA_MODEL="$OLLAMA_MODEL" \
  OLLAMA_TIMEOUT_S="${OLLAMA_TIMEOUT_S:-120}" \
  poetry run perceptron-api >"$API_LOG" 2>&1 & echo $! >"$API_PID_FILE"
)
(cd "$FRONTEND_DIR" && npm run dev -- --host 127.0.0.1 --port 5173 >"$FE_LOG" 2>&1 & echo $! >"$FRONTEND_PID_FILE")

if ! wait_for_url "$API_URL"; then
  echo "API failed to start. See $API_LOG" >&2
  exit 1
fi

if ! wait_for_url "$FE_URL"; then
  echo "Frontend failed to start. See $FE_LOG" >&2
  exit 1
fi

echo "Restarted backend and frontend (health checks passed)."
echo "API PID: $(cat "$API_PID_FILE")"
echo "Frontend PID: $(cat "$FRONTEND_PID_FILE")"
