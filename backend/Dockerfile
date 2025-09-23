FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Worker deps (ako postoji poseban requirements; u suprotnom ostaje prazan)
COPY worker/requirements.txt /tmp/worker-req.txt
RUN pip install --no-cache-dir -r /tmp/worker-req.txt || true

# Kod
COPY backend /app
COPY ml /ml
COPY worker /worker

# Start skripta
COPY backend/start.sh /start.sh
RUN chmod +x /start.sh

ENV PYTHONPATH="/app:/ml:${PYTHONPATH}"

CMD ["/bin/sh","/start.sh"]