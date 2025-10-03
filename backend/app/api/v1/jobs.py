from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services import r2
from pathlib import Path
import os, uuid

router = APIRouter(prefix="/api/v1", tags=["jobs"])

@router.get("/result/{job_id}")
def get_result(job_id: str):
    # Worker treba da je upisao "results/<job_id>/response.json" (relativne putanje)
    try:
        meta = r2.get_json(r2.k_results(job_id, "response.json"))
    except Exception:
        raise HTTPException(404, "Result not ready")

    # presign sve putanje koje su relativne (bez "http")
    def sig(u: str) -> str:
        return u if u.startswith("http") else r2.presign_get(u)

    # očekujemo meta sa ključevima kao ranije ali sa relativnim s3 keyevima
    out = {
        "job_id": job_id,
        "counts_csv": sig(meta["counts_csv"]),
        "per_sec_csv": sig(meta["per_sec_csv"]),
        "by_min_csv":  sig(meta["by_min_csv"]),
        "peaks_csv":   sig(meta["peaks_csv"]),
        "heatmap": {
            "overlay":         sig(meta["heatmap"]["overlay"]),
            "colored":         sig(meta["heatmap"]["colored"]),
            "gray":            sig(meta["heatmap"]["gray"]),
            "transparent_png": sig(meta["heatmap"]["transparent_png"]),
            "preview":         sig(meta["heatmap"]["preview"]),
        },
        "snapshot": sig(meta["snapshot"]),
        "unique_total": meta["unique_total"],
        "peak": meta["peak"],
    }
    return JSONResponse(out)
