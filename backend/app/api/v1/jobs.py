# backend/app/api/v1/jobs.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os, json

from app.services.storage import save_upload, job_dir, status_path
from app.services.processor import new_job_id

router = APIRouter(prefix="/api/v1", tags=["jobs"])

ROOT = Path(__file__).resolve().parents[4]
STORAGE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))

def to_url(p: str) -> str:
    abs_p = Path(p).resolve()
    rel = abs_p.relative_to(STORAGE)
    return "/files/" + rel.as_posix()

@router.post("/analyze")
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    """
    ASYNC: primi upload, vrati 202 + job_id + status_url.
    Worker će pokupiti iz uploads/ i generirati rezultate u results/<job_id>/...
    """
    try:
        # 1) spremi upload (u STORAGE/uploads/..)
        video_path = save_upload(file)

        # 2) napravi job_id i pripremi radni dir (da postoji results/<job_id>)
        jid = new_job_id()
        _ = job_dir(jid)

        # 3) zapiši početni status (nije nužno, ali korisno)
        sp = status_path(jid)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"status": "processing"}), encoding="utf-8")

        # 4) vrati 202 Accepted + gdje da se polla
        return JSONResponse(
            {"job_id": jid, "status": "processing", "status_url": f"/api/v1/jobs/{jid}"},
            status_code=202
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    """
    Vrati status ili kompletne rezultate u ISTOM obliku kao i stari sinkroni endpoint.
    """
    # status.json piše worker u STORAGE/status/<job_id>.json
    sp = status_path(job_id)
    if not sp.exists():
        # još ništa (ili je worker nije krenuo)
        return {"status": "processing"}

    data = json.loads(sp.read_text(encoding="utf-8"))
    st = data.get("status", "processing")
    if st != "done":
        # može biti "processing" ili "failed"
        if st == "failed":
            raise HTTPException(status_code=500, detail=data.get("error", "failed"))
        return {"status": "processing"}

    # Ako je gotovo, složi isti output kao prije:
    # Pretpostavka: worker je generirao iste datoteke/strukturu u results/<job_id>/...
    workdir = STORAGE / "results" / job_id

    # Ovdje čitaš ono što je worker napravio – nazivi datoteka kao i ranije:
    # counts.csv, analysis per_sec.csv, by_minute.csv, peaks.csv, heatmap/... , snapshot.jpg
    counts_csv = workdir / "counts.csv"
    per_sec = workdir / "analysis" / "per_sec.csv"
    by_min  = workdir / "analysis" / "by_minute.csv"
    peaks   = workdir / "analysis" / "peaks.csv"
    heat    = workdir / "heatmap"
    snap    = workdir / "snapshot.jpg"

    # unique_total/peak – ako ih worker već računa i negdje sprema (npr. u response.json),
    # pročitaj iz njega; ili ako imaš CSV iz kojeg se čita – minimalno:
    uniq_total = data.get("unique_total")
    peak = data.get("peak")

    out = {
        "job_id": job_id,
        "video": None,  # ili worker negdje sprema relativni path
        "counts_csv": to_url(str(counts_csv)),
        "unique_total": uniq_total,
        "peak": peak,
        "per_sec_csv": to_url(str(per_sec)),
        "by_min_csv":  to_url(str(by_min)),
        "peaks_csv":   to_url(str(peaks)),
        "heatmap": {
            "overlay":         to_url(str(heat / "heatmap_overlay.png")),
            "colored":         to_url(str(heat / "heatmap_colored.png")),
            "gray":            to_url(str(heat / "heatmap_gray.png")),
            "transparent_png": to_url(str(heat / "heatmap_transparent.png")),
            "preview":         to_url(str(heat / "preview.jpg")),
        },
        "snapshot": to_url(str(snap)),
    }
    return JSONResponse(out)
