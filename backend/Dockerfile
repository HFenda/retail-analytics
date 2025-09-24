FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /tmp/req-backend.txt
COPY worker/requirements.txt /tmp/req-worker.txt

RUN pip install --no-cache-dir -r /tmp/req-backend.txt \
 && pip install --no-cache-dir -r /tmp/req-worker.txt

COPY backend /app
COPY ml /ml
COPY worker /worker
COPY backend/start.sh /start.sh
RUN chmod +x /start.sh

ENV PYTHONPATH=/app

CMD ["/bin/sh","/start.sh"]
