#!/bin/sh
set -e

echo "[start.sh] Python version:"
python -V

echo "[start.sh] Sys.path and /app listing:"
python - <<'PY'
import sys, os
print(sys.path)
print(os.listdir("/app"))
PY

echo "[start.sh] Trying to import app.main:app..."
python - <<'PY'
import importlib
m = importlib.import_module("app.main")
print("Imported:", m)
print("Has 'app' attribute:", hasattr(m, "app"))
PY

echo "[start.sh] Starting uvicorn directly"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
