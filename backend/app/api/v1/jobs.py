from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, json, time, uuid

from app.services.storage import (
    save_upload, job_dir,
    write_status, read_status, read_result_json,
    upload_path_and_url, # koristi se indirektno kroz worker
)

router = APIRouter(prefix="/api/v1", tags=["jobs"])
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()

@router.post("/analyze")
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    try:
        local_path = save_upload(file)         # snimi (i ako je R2, uploadan je pod uploads/<vid>)
        fname = os.path.basename(local_path)
        job_id = uuid.uuid4().hex

        # pripremi prazan results dir (lokalno, radi developera)
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
        return read_result_json(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not Found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
