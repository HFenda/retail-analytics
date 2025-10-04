from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, json, time, uuid

from app.services.storage import (
    save_upload, job_dir,
    write_status, read_status, read_result_json,
    upload_path_and_url,
)

router = APIRouter(prefix="/api/v1", tags=["jobs"])

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()

R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "").rstrip("/")
IS_R2 = STORAGE_BACKEND in ("r2", "s3")

def _to_r2(url_or_path: str) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    if url_or_path.startswith("/files/") and R2_PUBLIC_URL:
        rel = url_or_path[len("/files/"):]
        return f"{R2_PUBLIC_URL}/{rel}"
    return url_or_path

def _rewrite_result_payload_for_r2(payload: dict) -> dict:
    out = dict(payload)

    for k in ["counts_csv", "per_sec_csv", "by_min_csv", "peaks_csv", "snapshot", "video"]:
        if k in out and isinstance(out[k], str):
            out[k] = _to_r2(out[k])

    if isinstance(out.get("heatmap"), dict):
        hm = dict(out["heatmap"])
        for hk in ["overlay", "colored", "gray", "transparent_png", "preview"]:
            if hk in hm and isinstance(hm[hk], str):
                hm[hk] = _to_r2(hm[hk])
        out["heatmap"] = hm

    return out


@router.post("/analyze")
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    try:
        local_path = save_upload(file)
        fname = os.path.basename(local_path)
        job_id = uuid.uuid4().hex

        _ = job_dir(job_id)

        status_payload = {
            "status": "pending",
            "ts": int(time.time()),
            "file": fname,
            "vid_stride": vid_stride,
        }
        if STORAGE_BACKEND in ("r2", "s3"):
            status_payload["r2_key"] = f"uploads/{fname}"

        write_status(job_id, status_payload)

        return JSONResponse(
            {
                "job_id": job_id,
                "message": "Accepted. Worker će obraditi u pozadini.",
                "status_url": f"/api/v1/status/{job_id}",
                "result_url": f"/api/v1/result/{job_id}",
            },
            status_code=202,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status/{job_id}")
def status(job_id: str):
    try:
        return read_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not Found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{job_id}")
def result(job_id: str):
    try:
        data = read_result_json(job_id)
        if IS_R2:
            data = _rewrite_result_payload_for_r2(data)
        return JSONResponse(data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not Found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
