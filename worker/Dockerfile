FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CPU wheels for torch
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⬇️ copy requirements from the *worker folder* explicitly
COPY worker/requirements.in /app/requirements.in
# (If you only have requirements.txt, change both lines to ...requirements.txt)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=20 --prefer-binary -r /app/requirements.in

# ⬇️ copy worker code
COPY worker/ /app/

# ⬇️ copy shared ML package (visible because build context will be repo root)
COPY ml/ /app/ml

ENV PYTHONPATH=/app

CMD ["python", "run_worker.py"]
