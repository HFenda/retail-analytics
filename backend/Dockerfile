FROM python:3.11-slim

# Sistemske zavisnosti
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180

WORKDIR /app

# Install deps (backend reqs)
COPY backend/requirements.txt /tmp/req-backend.txt
RUN pip install --no-cache-dir -r /tmp/req-backend.txt

# Kod iz root konteksta
COPY backend /app
COPY worker /worker
COPY ml /ml
COPY backend/start.sh /start.sh
RUN chmod +x /start.sh

# Nek Python vidi i worker i ml
ENV PYTHONPATH=/app:/ml:/worker

CMD ["/bin/sh","/start.sh"]
