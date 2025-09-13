# backend/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kontekst je REPO ROOT, pa putanje imaju prefix backend/
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Kopiraj backend kod i ML modul
COPY backend /app
COPY ml /ml

# Omogući import "ml" i "app"
ENV PYTHONPATH="/app:/ml:${PYTHONPATH}"

# Pokreni kao i do sada
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","-w","2","-b","0.0.0.0:8000","app.main:app"]
