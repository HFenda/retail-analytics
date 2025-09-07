# backend/app/services/storage.py
from pathlib import Path
from fastapi import UploadFile
import shutil, uuid

# project root: .../retail-analytics
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "storage"

def ensure_dirs():
    (BASE / "uploads").mkdir(parents=True, exist_ok=True)
    (BASE / "results").mkdir(parents=True, exist_ok=True)

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
