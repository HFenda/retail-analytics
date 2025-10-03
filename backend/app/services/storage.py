from __future__ import annotations
from pathlib import Path
from fastapi import UploadFile
import shutil, uuid, os, mimetypes

# R2/S3
import boto3
from botocore.client import Config

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))

def _mime_for(path: str | Path) -> str | None:
    m, _ = mimetypes.guess_type(str(path))
    return m

def ensure_dirs():
    for sub in ("uploads", "results", "status", "processed", "failed"):
        (BASE / sub).mkdir(parents=True, exist_ok=True)

# ----------------------------
# R2 / S3 kompatibilni klijent
# ----------------------------
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
IS_R2 = STORAGE_BACKEND in ("r2", "s3")

def _make_s3like_client():
    access_key = os.getenv("R2_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_ACCESS_KEY")
    endpoint   = os.getenv("R2_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    region     = os.getenv("R2_REGION") or os.getenv("S3_REGION") or "auto"
    if not (access_key and secret_key and endpoint):
        raise RuntimeError("S3/R2 kredencijali nisu kompletni (ACCESS_KEY, SECRET, ENDPOINT).")
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        region_name=None if region == "auto" else region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )

if IS_R2:
    _S3 = _make_s3like_client()
    _BUCKET = os.getenv("R2_BUCKET") or os.getenv("S3_BUCKET")
    _PUBLIC_BASE = os.getenv("R2_PUBLIC_BASE_URL") or os.getenv("S3_PUBLIC_BASE_URL")  # ako je bucket public

def _r2_put(local_path: Path, key: str, content_type: str | None = None) -> str:
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    with open(local_path, "rb") as f:
        _S3.put_object(Bucket=_BUCKET, Key=key, Body=f.read(), **extra)
    if _PUBLIC_BASE:  # sklopi public URL
        return f"{_PUBLIC_BASE.rstrip('/')}/{key.lstrip('/')}"
    # fallback: presigned za čitanje (1h)
    return _S3.generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET, "Key": key},
        ExpiresIn=3600,
    )

# ----------------------------
# API koji koristi ostatak backenda
# ----------------------------
def save_upload(file: UploadFile) -> str:
    """
    Spremi upload na lokalni FS (pipeline radi nad lokalnim pathom).
    Ako je R2 aktivan, kasnije pozivamo upload_path_and_url() da dižemo izlaze.
    """
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

def local_to_url(p: str) -> str:
    """
    Za lokalni FS vrati '/files/...' URL koji frontend zna otvoriti.
    """
    abs_p = Path(p).resolve()
    rel = abs_p.relative_to(BASE)   # pukne ako nije u storage/
    return "/files/" + rel.as_posix()

def upload_path_and_url(local_path: str, key: str | None = None, content_type: str | None = None) -> str:
    """
    Ako je R2 uključen, upload u bucket i vrati (public/presigned) URL.
    Ako nije, vrati '/files/...' URL prema lokalnom pathu.
    """
    if not IS_R2:
        return local_to_url(local_path)

    lp = Path(local_path)
    if key is None:
        # default ključ = relativni path ispod BASE (npr results/abc/heatmap/preview.jpg)
        try:
            rel = lp.resolve().relative_to(BASE)
            key = rel.as_posix()
        except Exception:
            key = lp.name

    ct = content_type or _mime_for(lp) or "application/octet-stream"
    return _r2_put(lp, key, content_type=ct)
