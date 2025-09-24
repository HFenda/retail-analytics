#!/bin/sh
set -e

echo "[start.sh] Python version:"
python -V

echo "[start.sh] sys.path:"
python - <<'PY'
import sys; [print(" -", p) for p in sys.path]
PY

echo "[start.sh] /app listing:"
python - <<'PY'
import os; print(os.listdir("/app"))
PY

echo "[start.sh] Import test: app.main:app"
python - <<'PY'
import importlib
m = importlib.import_module("app.main")
print("Imported:", m)
print("Has 'app':", hasattr(m, "app"))
PY

echo "[start.sh] Starting uvicorn"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
