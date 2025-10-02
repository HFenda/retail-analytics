from pathlib import Path
from fastapi import UploadFile
import shutil, uuid, os

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))

def ensure_dirs():
    for sub in ("uploads", "results", "status", "processed", "failed"):
        (BASE / sub).mkdir(parents=True, exist_ok=True)

def save_upload(file: UploadFile) -> str:
    ensure_dirs()
    suffix = Path(file.filename or "").suffix or ".mp4"
    vid = f"{uuid.uuid4().hex}{suffix}"
    dest = BASE / "uploads" / vid
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(dest)

def job_dir(job_id: str) -> str:
    ensure_dirs()
    d = BASE / "results" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

def status_path(job_id: str) -> Path:
    ensure_dirs()
    return BASE / "status" / f"{job_id}.json"
