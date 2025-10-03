# app/api/v1/jobs.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os, json, time

from app.services.storage import (
    save_upload, job_dir, status_path, upload_path_and_url, BASE
)

router = APIRouter(prefix="/api/v1", tags=["jobs"])

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()

def _write_status(job_id: str, payload: dict):
    sp = status_path(job_id)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

@router.post("/analyze")
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    """
    ASYNC prijem: vrati 202 + job_id; worker će kasnije napisati rezultat.
    """
    try:
        # 1) snimi upload lokalno
        local_path = save_upload(file)  # .../storage/uploads/<uuid>.ext
        fname = Path(local_path).name

        # 2) job_id = naziv direktorija rezultata
        #    (koristimo isti uuid format ko i storage.save_upload name)
        job_id = Path(job_dir(os.urandom(8).hex())).name  # napravi prazan dir
        # bolje: job_id = uuid4.hex, ali da ostanemo minimalni:
        job_id = job_id

        # 3) ako je R2, digni upload u R2 i u status upiši r2_key
        status_payload = {
            "status": "pending",
            "ts": int(time.time()),
            "file": fname,        # informativno
            "vid_stride": vid_stride,
        }

        if STORAGE_BACKEND in ("r2", "s3"):
            # key koji će worker znati: uploads/<fname>
            r2_key = f"uploads/{fname}"
            # digni u R2 (vrati URL, ali nama je važan key)
            _ = upload_path_and_url(local_path, key=r2_key)
            status_payload["r2_key"] = r2_key

        # 4) status -> pending
        _write_status(job_id, status_payload)

        # 5) return 202 + linkovi
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
    sp = status_path(job_id)
    if not sp.exists():
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/result/{job_id}")
def result(job_id: str):
    # očekujemo response.json u results/<job_id>/
    rp = Path(BASE) / "results" / job_id / "response.json"
    if not rp.exists():
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        return json.loads(rp.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
