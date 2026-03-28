#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SAMPLE_IMAGE="${1:-}"
USERS="${2:-25}"
SPAWN_RATE="${3:-5}"
DURATION="${4:-2m}"
HOST="${5:-http://127.0.0.1:5000}"

if [[ -z "$SAMPLE_IMAGE" ]]; then
  for candidate in \
    "$ROOT_DIR/data/test/NORMAL" \
    "$ROOT_DIR/data/test/PNEUMONIA" \
    "$ROOT_DIR/data/val/NORMAL" \
    "$ROOT_DIR/data/val/PNEUMONIA"; do
    if [[ -d "$candidate" ]]; then
      first_file="$(find "$candidate" -maxdepth 1 -type f | head -n 1 || true)"
      if [[ -n "$first_file" ]]; then
        SAMPLE_IMAGE="$first_file"
        break
      fi
    fi
  done
fi

if [[ -z "$SAMPLE_IMAGE" || ! -f "$SAMPLE_IMAGE" ]]; then
  echo "Usage: $0 [/absolute/path/to/sample_image] [users] [spawn_rate] [duration] [host]"
  echo "No sample image found. Provide one explicitly or place images in data/test or data/val."
  exit 1
fi

export LOCUST_SAMPLE_IMAGE="$SAMPLE_IMAGE"

echo "Using sample image: $SAMPLE_IMAGE"
echo "Users: $USERS | Spawn rate: $SPAWN_RATE | Duration: $DURATION | Host: $HOST"

LOCUST_CMD="locust"
if [[ -x "$ROOT_DIR/venv/bin/locust" ]]; then
  LOCUST_CMD="$ROOT_DIR/venv/bin/locust"
fi

"$LOCUST_CMD" -f locustfile.py \
  --host "$HOST" \
  --headless \
  --users "$USERS" \
  --spawn-rate "$SPAWN_RATE" \
  --run-time "$DURATION"
