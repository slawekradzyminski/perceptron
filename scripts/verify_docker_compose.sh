#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"
API_URL="http://127.0.0.1:8000/state"
FE_URL="http://127.0.0.1:5173/"

wait_for_url() {
  local url="$1"
  local retries=60
  for _ in $(seq 1 "$retries"); do
    if curl -s "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

cleanup() {
  docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
}
trap cleanup EXIT

docker compose -f "$COMPOSE_FILE" up -d --build

echo "Waiting for API..."
if ! wait_for_url "$API_URL"; then
  docker compose -f "$COMPOSE_FILE" logs
  echo "API failed to start" >&2
  exit 1
fi

echo "Waiting for frontend..."
if ! wait_for_url "$FE_URL"; then
  docker compose -f "$COMPOSE_FILE" logs
  echo "Frontend failed to start" >&2
  exit 1
fi

api_json="$(curl -s "$API_URL")"
fe_html="$(curl -s "$FE_URL")"

if ! echo "$api_json" | grep -q '"dataset"'; then
  docker compose -f "$COMPOSE_FILE" logs
  echo "API response missing dataset field" >&2
  exit 1
fi

if ! echo "$fe_html" | grep -q "Perceptron Visual Lab"; then
  docker compose -f "$COMPOSE_FILE" logs
  echo "Frontend HTML missing title" >&2
  exit 1
fi

echo "OK: Dockerized API and frontend are running"
