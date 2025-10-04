from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import os, json, uuid, shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))
IS_R2 = os.getenv("STORAGE_BACKEND", "local").lower() in ("r2", "s3")

_s3 = None
_bucket = None
_public_base: Optional[str] = None

if IS_R2:
    import boto3
    from botocore.client import Config
    _access = os.getenv("R2_ACCESS_KEY_ID")
    _secret = os.getenv("R2_SECRET_ACCESS_KEY")
    _endpoint = os.getenv("R2_ENDPOINT") or os.getenv("R2_ENDPOINT_URL")
    _bucket  = os.getenv("R2_BUCKET", "retail-analytics")
    _public_base = (
        os.getenv("R2_PUBLIC_BASE_URL")
        or os.getenv("R2_PUBLIC_URL")
    )
    if not (_access and _secret and _endpoint and _bucket):
        raise RuntimeError("R2 credentials/env nisu kompletni.")
    _s3 = boto3.client(
        "s3",
        aws_access_key_id=_access,
        aws_secret_access_key=_secret,
        endpoint_url=_endpoint,
        region_name=None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )

def ensure_dirs():
    for sub in ("uploads", "results", "status", "processed", "failed"):
        (BASE / sub).mkdir(parents=True, exist_ok=True)

def _local_save(src_file, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(src_file, f)

def _local_url(rel_path: str) -> str:
    return "/files/" + rel_path.replace("\\", "/")

_DEF_CT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".csv": "text/csv",
    ".mp4": "video/mp4",
    ".json": "application/json",
}
def _guess_ct(path: Path, fallback: Optional[str]=None) -> Optional[str]:
    return _DEF_CT.get(path.suffix.lower(), fallback)


def _r2_put_file(key: str, local_path: str, content_type: Optional[str]=None) -> str:
    assert _s3 is not None and _bucket is not None
    ct = content_type or _guess_ct(Path(local_path))
    extra = {"ContentType": ct} if ct else {}
    with open(local_path, "rb") as f:
        _s3.upload_fileobj(f, _bucket, key, ExtraArgs=extra)
    if _public_base:
        return f"{_public_base.rstrip('/')}/{key.lstrip('/')}"
    return _s3.generate_presigned_url("get_object", Params={"Bucket": _bucket, "Key": key}, ExpiresIn=60*60)

def _r2_put_bytes(key: str, data: bytes, content_type: Optional[str]=None) -> str:
    assert _s3 is not None and _bucket is not None
    ct = content_type
    extra = {"ContentType": ct} if ct else {}
    _s3.put_object(Bucket=_bucket, Key=key, Body=data, **extra)
    if _public_base:
        return f"{_public_base.rstrip('/')}/{key.lstrip('/')}"
    return _s3.generate_presigned_url("get_object", Params={"Bucket": _bucket, "Key": key}, ExpiresIn=60*60)

def _r2_get_text(key: str) -> str:
    assert _s3 is not None and _bucket is not None
    obj = _s3.get_object(Bucket=_bucket, Key=key)
    return obj["Body"].read().decode("utf-8")

def _r2_list(prefix: str):
    assert _s3 is not None and _bucket is not None
    token = None
    while True:
        kw = {"Bucket": _bucket, "Prefix": prefix}
        if token: kw["ContinuationToken"] = token
        resp = _s3.list_objects_v2(**kw)
        for it in resp.get("Contents", []):
            yield it["Key"]
        token = resp.get("NextContinuationToken")
        if not token:
            break

# ---------- PUBLIC API ----------
def save_upload(file) -> str:
    """
    Vrati lokalni path upload-a (dev) ili lokalnu kopiju koju smo upravo snimili
    i (ako je R2) dodatno uploadali u R2 pod uploads/<name>.
    """
    ensure_dirs()
    suffix = Path(file.filename or "").suffix or ".mp4"
    vid = f"{uuid.uuid4().hex}{suffix}"
    dest = BASE / "uploads" / vid
    _local_save(file.file, dest)

    if IS_R2:
        _r2_put_file(f"uploads/{vid}", str(dest), content_type="video/mp4")

    return str(dest)  # lokalni path

def job_dir(job_id: str) -> str:
    ensure_dirs()
    d = BASE / "results" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

def status_path(job_id: str) -> Path:
    return BASE / "status" / f"{job_id}.json"

def write_status(job_id: str, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    if IS_R2:
        _r2_put_bytes(f"status/{job_id}.json", data, content_type="application/json")
    else:
        sp = status_path(job_id)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_bytes(data)

def read_status(job_id: str) -> Dict[str, Any]:
    if IS_R2:
        txt = _r2_get_text(f"status/{job_id}.json")
        return json.loads(txt)
    sp = status_path(job_id)
    if not sp.exists():
        raise FileNotFoundError("status not found")
    return json.loads(sp.read_text(encoding="utf-8"))

def list_pending_jobs() -> list[Tuple[str, Dict[str, Any]]]:
    out = []
    if IS_R2:
        for key in _r2_list("status/"):
            if not key.endswith(".json"): continue
            jid = Path(key).stem
            try:
                st = json.loads(_r2_get_text(key))
                if st.get("status") == "pending":
                    out.append((jid, st))
            except Exception:
                pass
        return out
    # local
    sp_dir = BASE / "status"
    if not sp_dir.exists(): return []
    for p in sp_dir.glob("*.json"):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
            if st.get("status") == "pending":
                out.append((p.stem, st))
        except Exception:
            pass
    return out

def upload_path_and_url(local_path: str, key: Optional[str]=None, *, content_type: Optional[str]=None) -> str:
    """
    Ako je R2 -> upload i vrati public/presigned URL (po defaultu zadrži relativni layout
    unutar STORAGE_DIR, npr. results/<job_id>/heatmap/preview.jpg).
    Inače -> vrati /files/<rel>.
    """
    lp = Path(local_path).resolve()
    if IS_R2:
        if not key:
            try:
                rel = lp.relative_to(BASE)
            except ValueError:
                rel = Path("results") / lp.name
            key = str(rel).replace("\\", "/")
        return _r2_put_file(key, str(lp), content_type=content_type or _guess_ct(lp))
    # local
    rel = lp.relative_to(BASE)
    return _local_url(str(rel))

def write_result_json(job_id: str, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    if IS_R2:
        _r2_put_bytes(f"results/{job_id}/response.json", data, content_type="application/json")
    else:
        rp = BASE / "results" / job_id / "response.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_bytes(data)

def read_result_json(job_id: str) -> Dict[str, Any]:
    if IS_R2:
        txt = _r2_get_text(f"results/{job_id}/response.json")
        return json.loads(txt)
    rp = BASE / "results" / job_id / "response.json"
    if not rp.exists():
        raise FileNotFoundError("result not found")
    return json.loads(rp.read_text(encoding="utf-8"))
