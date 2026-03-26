#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/sample_image [users] [spawn_rate] [duration] [host]"
  exit 1
fi

SAMPLE_IMAGE="$1"
USERS="${2:-25}"
SPAWN_RATE="${3:-5}"
DURATION="${4:-2m}"
HOST="${5:-http://127.0.0.1:5000}"

if [[ ! -f "$SAMPLE_IMAGE" ]]; then
  echo "Sample image not found: $SAMPLE_IMAGE"
  exit 1
fi

export LOCUST_SAMPLE_IMAGE="$SAMPLE_IMAGE"

locust -f locustfile.py \
  --host "$HOST" \
  --headless \
  --users "$USERS" \
  --spawn-rate "$SPAWN_RATE" \
  --run-time "$DURATION"
