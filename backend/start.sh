set -e

WEB_CMD="gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:${PORT:-8000} app.main:app"

if [ "${RUN_EMBEDDED_WORKER:-1}" = "1" ]; then
  echo "[start.sh] Starting web + embedded worker"

  $WEB_CMD &
  WEB_PID=$!

  sleep 2

  export BACKEND_URL="http://127.0.0.1:${PORT:-8000}"

  python /worker/run_worker.py &
  WORKER_PID=$!

  trap 'kill -TERM $WEB_PID $WORKER_PID 2>/dev/null' TERM INT
  wait -n $WEB_PID $WORKER_PID
else
  echo "[start.sh] Starting web only"
  exec $WEB_CMD
fi
