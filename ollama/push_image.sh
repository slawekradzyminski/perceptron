#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-slawekradzyminski/llama3.2:1b}"

echo "Building image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" -f "./Dockerfile" .

echo "Pushing image: $IMAGE_NAME"
docker push "$IMAGE_NAME"

echo "Done: $IMAGE_NAME pushed."
