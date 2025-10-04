# worker/Dockerfile
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

# 1) deps iz worker/ (prilagodi naziv fajla ako koristiš requirements.txt umjesto .in)
COPY worker/requirements.in /app/
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=20 --prefer-binary -r requirements.in

# 2) Kod workera
COPY worker/ /app/

# 3) UBACI backendov paket `app/` (da import app.services radi)
COPY backend/app /app/app

# 4) Shared ML modul
COPY ml /app/ml

ENV PYTHONPATH=/app

CMD ["python","run_worker.py"]
