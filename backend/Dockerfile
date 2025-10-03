# -------- base --------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements iz backend/ (jer je build kontekst repo root)
COPY backend/requirements.in /app/requirements.in
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=20 --prefer-binary -r requirements.in

# app kod (samo backend/app)
COPY backend/app /app/app

# >>> KLJUČNO: kopiraj lokalni ml/ u image, da import "ml" postoji
#   (pobrini se da u root/ml postoji __init__.py, može biti prazan)
COPY ml /app/ml

ENV PYTHONPATH=/app
EXPOSE 8000

CMD exec gunicorn -k uvicorn.workers.UvicornWorker app.main:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 2 --threads 2 --timeout 3600
