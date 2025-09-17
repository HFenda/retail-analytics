from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
from pathlib import Path
import os, shutil, json

router = APIRouter(prefix="/api/v1", tags=["jobs"])

DATA_ROOT = Path("/data")
UPLOADS_DIR = DATA_ROOT / "uploads"
RESULTS_DIR = DATA_ROOT / "results"

def to_url(abs_path: str) -> str:
    p = Path(abs_path).resolve()
    rel = p.relative_to(DATA_ROOT) 
    return "/files/" + rel.as_posix()

def enqueue_job(video_path: str, vid_stride: int) -> str:
    """
    Najjednostavniji 'queue':
    napravi job dir i status.json koje worker može da pokupi.
    Pravi queue (Redis, RQ, Celery) je bolje.
    """
    job_id = str(uuid4())
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "request.json").write_text(json.dumps({
        "job_id": job_id,
        "video_path": video_path,
        "vid_stride": vid_stride
    }, ensure_ascii=False, indent=2))
    (job_dir / "status.json").write_text(json.dumps({"status": "queued"}))
    return job_id

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    vid_stride: int = Form(6)    
):
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = f"{uuid4()}_{file.filename}"
        dst_path = UPLOADS_DIR / safe_name

        with open(dst_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        job_id = enqueue_job(str(dst_path), vid_stride)
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "video": to_url(str(dst_path)),
                "status_url": f"/api/v1/jobs/{job_id}",
                "results_url": f"/files/results/{job_id}/"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    status_file = RESULTS_DIR / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(status_file.read_text())
