#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:5000}"

echo "[1/4] Health"
curl -s "${BASE_URL}/health" | tee /tmp/cv_health.json > /dev/null

echo "[2/4] Metrics"
curl -s "${BASE_URL}/metrics" | tee /tmp/cv_metrics.json > /dev/null

echo "[3/4] Visualization"
curl -s "${BASE_URL}/visualization-data" | tee /tmp/cv_visualization.json > /dev/null

echo "[4/4] Data status"
curl -s "${BASE_URL}/data-status" | tee /tmp/cv_data_status.json > /dev/null

echo "Smoke test completed."
echo "Saved responses:"
echo "- /tmp/cv_health.json"
echo "- /tmp/cv_metrics.json"
echo "- /tmp/cv_visualization.json"
echo "- /tmp/cv_data_status.json"
