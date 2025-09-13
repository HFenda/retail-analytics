from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.jobs import router as jobs_router

# Repo root (npr. .../retail-analytics)
ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Retail Analytics")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE = ROOT / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(STORAGE)), name="files")

app.include_router(jobs_router)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/healthz")
def healthz():
    return {"ok": True}
