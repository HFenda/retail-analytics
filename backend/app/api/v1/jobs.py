# backend/app/api/v1/jobs.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os

from app.services.storage import save_upload, job_dir
from app.services.processor import run_pipeline, new_job_id

router = APIRouter(prefix="/api/v1", tags=["jobs"])

# root storage (isti kao u main.py / STORAGE_DIR)
ROOT = Path(__file__).resolve().parents[4]
STORAGE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))

def to_url(p: str) -> str:
    abs_p = Path(p).resolve()
    rel = abs_p.relative_to(STORAGE)   # pukne ako nije u storage/
    return "/files/" + rel.as_posix()

@router.post("/analyze")
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    """
    SINKRONI endpoint — izvrši cijeli pipeline i vrati isti 'flattened' payload
    kao stari backend, da frontend ne treba mijenjati.
    """
    try:
        # 1) spremi upload
        video_path = save_upload(file)

        # 2) pripremi radni dir (job)
        jid = new_job_id()
        workdir = job_dir(jid)

        # 3) pokreni pipeline (blokira do kraja)
        res = run_pipeline(video_path, workdir, vid_stride=vid_stride)

        # 4) mapiraj u "stari" response oblik
        out = {
            "job_id": res["job_id"],
            "video": to_url(res["video"]),
            "counts_csv": to_url(res["counts"]["csv"]),
            "unique_total": res["counts"]["unique_total"],
            "peak": res["counts"]["peak"],
            "per_sec_csv": to_url(res["analysis"]["files"]["per_sec"]),
            "by_min_csv":  to_url(res["analysis"]["files"]["by_minute"]),
            "peaks_csv":   to_url(res["analysis"]["files"]["peaks"]),
            "heatmap": {
                "overlay":         to_url(res["heatmap"]["artifacts"]["overlay"]),
                "colored":         to_url(res["heatmap"]["artifacts"]["colored"]),
                "gray":            to_url(res["heatmap"]["artifacts"]["gray"]),
                "transparent_png": to_url(res["heatmap"]["artifacts"]["transparent_png"]),
                "preview":         to_url(res["heatmap"]["artifacts"]["preview"]),
            },
            "snapshot": to_url(res["snapshot"]["out"]),
        }
        return JSONResponse(out)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
