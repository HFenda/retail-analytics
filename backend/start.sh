set -e

export BACKEND_URL="http://127.0.0.1:${PORT}"

export UPLOADS_DIR="/data/uploads"
export RESULTS_DIR="/data/results"
export PROCESSED_DIR="/data/processed"
export FAILED_DIR="/data/failed"

python /worker/run_worker.py &

exec gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:${PORT} app.main:app