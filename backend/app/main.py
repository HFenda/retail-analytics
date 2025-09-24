from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute

from app.api.v1.jobs import router as jobs_router
from app.background_queue import start_consumer  # pokreće in-memory worker

DATA_ROOT = Path("/data")
(DATA_ROOT / "uploads").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "results").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "processed").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "failed").mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # start worker
    start_consumer()
    # debug: ispiši rute nakon što je app kompletno sastavljen
    paths = [r.path for r in app.routes if isinstance(r, APIRoute)]
    print("[routes]", paths, flush=True)
    yield
    # ovdje bi stavio cleanup ako ti ikad zatreba (gašenje thread-ova itd.)

app = FastAPI(title="Retail Analytics", lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True}

@app.head("/health")
def health_head():
    return Response(status_code=200)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(DATA_ROOT)), name="files")
app.include_router(jobs_router)
