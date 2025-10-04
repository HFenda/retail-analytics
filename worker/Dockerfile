FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Omogući CPU wheelove za PyTorch
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps
COPY worker/requirements.in /tmp/requirements.in
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=20 --prefer-binary -r /tmp/requirements.in

# kod workera
COPY worker/ /app

# da bi importi radili (run_pipeline i ml)
COPY backend/app /app/app
COPY ml /app/ml

ENV PYTHONPATH=/app

CMD ["python","run_worker.py"]
