FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180

WORKDIR /app

# 1) Kopiraj backend kod i requirements u /app
COPY backend/ /app/

# 2) (Ako import-aš nešto iz ml/ ili worker/ u runtime-u, kopiraj i to)
COPY ml /ml
COPY worker /worker

# 3) Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 4) Start skripta
COPY backend/start.sh /start.sh
RUN chmod +x /start.sh

ENV PYTHONPATH=/app:/ml:/worker

CMD ["/bin/sh","/start.sh"]