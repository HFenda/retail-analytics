FROM python:3.11-slim

# Sistemske zavisnosti
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Pip i Python env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180

WORKDIR /app

# Requirements (posebni layeri radi cache-a)
COPY backend/requirements.txt /tmp/req-backend.txt
COPY worker/requirements.txt /tmp/req-worker.txt

# Instalacija – prvo backend pa worker
RUN pip install --no-cache-dir -r /tmp/req-backend.txt \
 && pip install --no-cache-dir -r /tmp/req-worker.txt

# Kod
COPY backend /app
COPY ml /ml
COPY worker /worker
COPY backend/start.sh /start.sh
RUN chmod +x /start.sh

# Jednostavnije postavi PYTHONPATH (bez onog var ekspanziona što ti je davalo warning)
ENV PYTHONPATH=/app:/ml

CMD ["/bin/sh","/start.sh"]
