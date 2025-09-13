FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# instalacije ostaju kako jesu
COPY backend/requirements.txt /app/
RUN pip install -r /app/requirements.txt

# kopiraj backend kod
COPY backend /app

# KOPIRAJ I ML PAKET
COPY ml /app/ml

# osiguraj da je /app na sys.path
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","app.main:app","--bind","0.0.0.0:8000","--workers","2","--timeout","120"]