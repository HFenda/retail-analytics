FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /worker
COPY requirements.in /worker/
RUN pip install --no-cache-dir -r requirements.in
COPY . /worker
CMD ["python","run_worker.py"]
