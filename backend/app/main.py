# backend/app/main.py
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.jobs import router as jobs_router

app = FastAPI(title="Retail Analytics")

# CORS — za lokalni dev može '*' ili suzi na tačne origin-e
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ili ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared data root u kontejneru (bind-mount: ./data -> /data)
DATA_ROOT = Path("/data")
(DATA_ROOT / "uploads").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "results").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "processed").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "failed").mkdir(parents=True, exist_ok=True)

# Exponuj sve iz /data kao statiku pod /files, npr. /files/results/<job_id>/...
app.mount("/files", StaticFiles(directory=str(DATA_ROOT)), name="files")

# API rute
app.include_router(jobs_router)

@app.get("/health")
def health():
    return {"ok": True}