#!/bin/sh
set -euo pipefail

# Mora biti jedan proces zbog in-memory queue-a
export WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"
export WEB_TIMEOUT="${WEB_TIMEOUT:-120}"

WEB_CMD="gunicorn -k uvicorn.workers.UvicornWorker \
  -w ${WEB_CONCURRENCY} \
  --timeout ${WEB_TIMEOUT} \
  -b 0.0.0.0:${PORT:-8000} app.main:app"

echo "[start.sh] Starting web (single worker for in-memory queue)"
exec $WEB_CMD