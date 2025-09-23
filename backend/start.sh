set -eu

for d in "$UPLOADS_DIR" "$RESULTS_DIR" "$PROCESSED_DIR" "$FAILED_DIR"; do
  [ -n "${d:-}" ] && mkdir -p "$d"
done

export PYTHONPATH="/app:/ml:${PYTHONPATH:-}"

python /worker/run_worker.py &

exec gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:$PORT app.main:app
