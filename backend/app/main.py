from pathlib import Path
import os, sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="Retail Analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))
STORAGE.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(STORAGE)), name="files")

from app.api.v1.jobs import router as jobs_router
app.include_router(jobs_router)

@app.get("/health")
def health():
    return {"ok": True}
