# backend/app/api/v1/jobs.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os, json, uuid, time

from app.services.storage import save_upload, job_dir, status_path
# processor NE koristimo više u /analyze (obradu radi worker)
# from app.services.processor import run_pipeline, new_job_id

router = APIRouter(prefix="/api/v1", tags=["jobs"])

ROOT = Path(__file__).resolve().parents[4]
STORAGE = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage")))

def to_url(p: str) -> str:
    abs_p = Path(p).resolve()
    rel = abs_p.relative_to(STORAGE)   # pukne ako nije u storage/
    return "/files/" + rel.as_posix()

@router.post("/analyze", status_code=202)
async def analyze(file: UploadFile = File(...), vid_stride: int = 6):
    """
    ASINHRONO:
    1) Čuvamo upload u storage (lokalni ili R2 preko storage.py).
    2) Kreiramo job_id + status pending.
    3) Vraćamo 202 i job_id odmah (obradu radi worker iz uploads/).
    """
    try:
        # 1) spremi upload
        video_path = save_upload(file)

        # 2) kreiraj job_id i inicijalni status
        job_id = uuid.uuid4().hex
        jd = job_dir(job_id)
        st = status_path(job_id)
        st.parent.mkdir(parents=True, exist_ok=True)
        st.write_text(json.dumps({
            "status": "pending",
            "ts": int(time.time()),
            "file": Path(video_path).name,
            "vid_stride": vid_stride
        }, indent=2), encoding="utf-8")

        # 3) vrati 202 sa linkovima koje frontend može da koristi za poll
        return JSONResponse({
            "job_id": job_id,
            "status_url": f"/api/v1/status/{job_id}",
            "result_url": f"/api/v1/result/{job_id}",
            "message": "Accepted. Worker će obraditi u pozadini."
        }, status_code=202)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status/{job_id}")
def job_status(job_id: str):
    """
    Vrati status.json za dati job_id.
    """
    st = status_path(job_id)
    if not st.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    data = json.loads(st.read_text(encoding="utf-8"))
    return data

@router.get("/result/{job_id}")
def job_result(job_id: str):
    """
    Kad je gotovo, isporuči rezultat u ISTOM obliku kao i ranije
    (tj. ono što je sinhroni analyze vraćao).
    Pretpostavka: worker upisuje canonical izlaz u results/{job_id}/response.json
    sa istim poljima, a fajlove snima u storage i zato ih možemo mapirati u /files URLs.
    """
    jd = Path(job_dir(job_id))
    resp = jd / "response.json"
    if not resp.exists():
        # Ako nije gotovo, vrati status
        st = status_path(job_id)
        if st.exists():
            return JSONResponse({"status": "processing"}, status_code=202)
        raise HTTPException(status_code=404, detail="Job not found")

    payload = json.loads(resp.read_text(encoding="utf-8"))

    # Mapiraj relativne putanje na /files/ URL (ako već nisu apsolutni linkovi)
    def absurl(u: str | None) -> str | None:
        if not u:
            return None
        if u.startswith("http://") or u.startswith("https://") or u.startswith("/files/"):
            return u
        return to_url(u)

    # Očekujemo polja kao ranije
    out = {
        "job_id": payload.get("job_id", job_id),
        "video": absurl(payload.get("video")),
        "counts_csv": absurl(payload["counts"]["csv"]) if payload.get("counts") else None,
        "unique_total": payload["counts"]["unique_total"] if payload.get("counts") else None,
        "peak": payload["counts"]["peak"] if payload.get("counts") else None,
        "per_sec_csv": absurl(payload["analysis"]["files"]["per_sec"]) if payload.get("analysis") else None,
        "by_min_csv":  absurl(payload["analysis"]["files"]["by_minute"]) if payload.get("analysis") else None,
        "peaks_csv":   absurl(payload["analysis"]["files"]["peaks"]) if payload.get("analysis") else None,
        "heatmap": {
            "overlay":         absurl(payload["heatmap"]["artifacts"]["overlay"]) if payload.get("heatmap") else None,
            "colored":         absurl(payload["heatmap"]["artifacts"]["colored"]) if payload.get("heatmap") else None,
            "gray":            absurl(payload["heatmap"]["artifacts"]["gray"]) if payload.get("heatmap") else None,
            "transparent_png": absurl(payload["heatmap"]["artifacts"]["transparent_png"]) if payload.get("heatmap") else None,
            "preview":         absurl(payload["heatmap"]["artifacts"]["preview"]) if payload.get("heatmap") else None,
        } if payload.get("heatmap") else None,
        "snapshot": absurl(payload.get("snapshot", {}).get("out") if isinstance(payload.get("snapshot"), dict) else payload.get("snapshot")),
    }
    return JSONResponse(out)
