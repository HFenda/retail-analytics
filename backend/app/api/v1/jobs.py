from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os

from app.services.storage import (
    save_upload, job_dir, upload_path_and_url, local_to_url, status_path
)
from app.services.processor import run_pipeline, new_job_id

router = APIRouter(prefix="/api/v1", tags=["jobs"])

ROOT = Path(__file__).resolve().parents[4]
STORAGE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))

@router.post("/analyze")
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    """
    Sinkroni pipeline: generiši sve artefakte i vrati response istog oblika
    kao ranije. Ako je STORAGE_BACKEND=r2, linkovi će biti R2 URL-ovi
    (public ili presigned). Inače će biti /files/... lokalni URL-ovi.
    """
    try:
        # 1) upload -> lokalni fajl (za obradu)
        video_path = save_upload(file)

        # 2) pripremi radni dir (job)
        jid = new_job_id()
        workdir = job_dir(jid)

        # 3) pipeline
        res = run_pipeline(video_path, workdir, vid_stride=vid_stride)

        # 4) upload u storage (R2 ili lokalni /files) i mapiranje na stari oblik
        #    ključ u bucketu: jobs/<jid>/...
        def up(local_path: str, rel_key: str) -> str:
            key = f"jobs/{jid}/{rel_key}".replace("\\", "/")
            return upload_path_and_url(local_path, key=key)

        # video
        video_url = up(res["video"], "uploads/source" + Path(res["video"]).suffix)

        # counts
        counts_csv_url = up(res["counts"]["csv"], "counts.csv")

        # analysis
        per_sec_url   = up(res["analysis"]["files"]["per_sec"],   "analysis/per_sec.csv")
        by_min_url    = up(res["analysis"]["files"]["by_minute"], "analysis/by_minute.csv")
        peaks_url     = up(res["analysis"]["files"]["peaks"],     "analysis/peaks.csv")

        # heatmap
        h_art = res["heatmap"]["artifacts"]
        heat_urls = {
            "overlay":         up(h_art["overlay"],         "heatmap/overlay.mp4"),
            "colored":         up(h_art["colored"],         "heatmap/colored.png"),
            "gray":            up(h_art["gray"],            "heatmap/gray.png"),
            "transparent_png": up(h_art["transparent_png"], "heatmap/transparent.png"),
            "preview":         up(h_art["preview"],         "heatmap/preview.jpg"),
        }

        # snapshot
        snapshot_url = up(res["snapshot"]["out"], "snapshot.jpg")

        out = {
            "job_id": jid,
            "video": video_url,
            "counts_csv": counts_csv_url,
            "unique_total": res["counts"]["unique_total"],
            "peak": res["counts"]["peak"],
            "per_sec_csv": per_sec_url,
            "by_min_csv":  by_min_url,
            "peaks_csv":   peaks_url,
            "heatmap": heat_urls,
            "snapshot": snapshot_url,
        }
        return JSONResponse(out)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
