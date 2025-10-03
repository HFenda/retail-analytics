# -------- base --------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (Opcionalno) Ako ti trebaju PyTorch CPU wheelovi u workeru, dodaj ove dvije linije.
# Ako worker NEMA torch/ultralytics, slobodno ih izbriši.
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# >>> VAŽNO: context je root repo-a, pa koristi "worker/requirements.in"
COPY worker/requirements.in /tmp/requirements.in
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=20 --prefer-binary -r /tmp/requirements.in

# Aplikacijski kod workera (iz podmape worker/)
COPY worker/ /app

# Treba nam i shared ML paket iz root-a (ako ga worker koristi)
# Ako worker ne import-a ništa iz ml/, ovu liniju možeš izbaciti.
COPY ml /app/ml

ENV PYTHONPATH=/app

# JSON forma CMD-a da OS signali rade ispravno
CMD ["python", "run_worker.py"]