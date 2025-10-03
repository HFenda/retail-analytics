from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os

from app.services.storage import save_upload, job_dir, upload_path_and_url

router = APIRouter(prefix="/api/v1", tags=["jobs"])

@router.post("/analyze")
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    """
    Sinkroni endpoint — izvrši cijeli pipeline i vrati isti 'flattened' payload
    koji frontend očekuje (bez promjena na frontendu).
    """
    try:
        # 1) spremi upload
        video_path = save_upload(file)

        # 2) pripremi radni dir (job)
        from app.services.processor import run_pipeline, new_job_id
        jid = new_job_id()
        workdir = job_dir(jid)

        # 3) pokreni pipeline (blokira do kraja)
        res = run_pipeline(video_path, workdir, vid_stride=vid_stride)

        # 4) mapiraj u “stari” response oblik i usput (ako je R2) digni fajlove
        out = {
            "job_id": res["job_id"],
            "video": upload_path_and_url(res["video"]),  # video/mp4
            "counts_csv": upload_path_and_url(res["counts"]["csv"]),  # text/csv
            "unique_total": res["counts"]["unique_total"],
            "peak": res["counts"]["peak"],
            "per_sec_csv": upload_path_and_url(res["analysis"]["files"]["per_sec"]),   # text/csv
            "by_min_csv":  upload_path_and_url(res["analysis"]["files"]["by_minute"]), # text/csv
            "peaks_csv":   upload_path_and_url(res["analysis"]["files"]["peaks"]),     # text/csv
            "heatmap": {
                "overlay":         upload_path_and_url(res["heatmap"]["artifacts"]["overlay"]),         # image/png
                "colored":         upload_path_and_url(res["heatmap"]["artifacts"]["colored"]),         # image/png
                "gray":            upload_path_and_url(res["heatmap"]["artifacts"]["gray"]),            # image/png
                "transparent_png": upload_path_and_url(res["heatmap"]["artifacts"]["transparent_png"]), # image/png
                "preview":         upload_path_and_url(res["heatmap"]["artifacts"]["preview"]),         # image/jpeg
            },
            "snapshot": upload_path_and_url(res["snapshot"]["out"]), # image/jpeg
        }
        return JSONResponse(out)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
