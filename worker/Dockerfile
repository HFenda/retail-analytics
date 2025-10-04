FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# PyTorch CPU wheels
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements
COPY worker/requirements.in /tmp/requirements.in
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=20 --prefer-binary -r /tmp/requirements.in

# worker kod
COPY worker/ /app

# da bi import "backend.app.services.processor" i "ml" radili:
COPY backend/app /app/app
COPY ml /app/ml

ENV PYTHONPATH=/app

# pokretanje
CMD ["python","run_worker.py"]
