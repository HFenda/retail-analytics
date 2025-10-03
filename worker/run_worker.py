import os
import time
import json
import uuid
import shutil
import logging
from pathlib import Path
from typing import Dict, Any

# === ML pipeline direktno (NE diramo tvoj ml) ===
# Ovo je ista logika kao u backend/app/services/processor.py,
# samo lokalno u workeru da ne ovisimo o backendu.
from ml import (
    run_unique_count,
    run_analysis,
    run_people_heatmap,
    run_snapshot_peak,
)

def run_pipeline(video_path: str, workdir: str, *, vid_stride: int = 6) -> Dict[str, Any]:
    Path(workdir).mkdir(parents=True, exist_ok=True)

    counts_csv = str(Path(workdir) / "counts.csv")
    uniq = run_unique_count(video=video_path, csv_path=counts_csv, vid_stride=vid_stride)

    ana = run_analysis(video=video_path, counts_csv=counts_csv, vid_stride=vid_stride)

    heat_dir = str(Path(workdir) / "heatmap")
    heat = run_people_heatmap(video=video_path, outdir=heat_dir)

    snap_path = str(Path(workdir) / "snapshot.jpg")
    snap = run_snapshot_peak(video=video_path, counts_csv=counts_csv, vid_stride=vid_stride, out=snap_path)

    return {
        "job_id": Path(workdir).name,
        "video": video_path,
        "counts": uniq,
        "analysis": ana,
        "heatmap": heat,
        "snapshot": {"out": snap_path} if isinstance(snap, str) else snap,
    }

# ============ Config iz ENV-a ============
# Backend URL nam više ne treba za obradu, sve radimo lokalno (R2/lokalni storage).
UPLOADS_DIR   = Path(os.getenv("UPLOADS_DIR", "/storage/uploads"))
RESULTS_DIR   = Path(os.getenv("RESULTS_DIR", "/storage/results"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "/storage/processed"))
FAILED_DIR    = Path(os.getenv("FAILED_DIR", "/storage/failed"))

# Backendov /api/v1/analyze sada pri uploadu kreira status fajl u STORAGE_DIR/status/<job_id>.json.
# Mi ga čitamo. Ako STORAGE_DIR nije zadan, pretpostavimo parent od RESULTS_DIR.
BASE_STORAGE  = Path(os.getenv("STORAGE_DIR", str(RESULTS_DIR.parent)))
STATUS_DIR    = BASE_STORAGE / "status"

POLL_SECONDS  = int(os.getenv("POLL_SECONDS", "2"))
RETRY_MAX     = int(os.getenv("RETRY_MAX", "1"))  # mi sami radimo retry loop unutar obrade
YOLO_MODEL_REF = os.getenv("YOLO_MODEL_REF", "yolov8n.pt")  # samo log/info, ml već zna što će

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".mpeg", ".mpg", ".webm", ".m4v", ".ts"}

# ============ Logging ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s: %(message)s"
)
log = logging.getLogger("worker")

# ============ Helpers ============
def ensure_dirs():
    for d in (UPLOADS_DIR, RESULTS_DIR, PROCESSED_DIR, FAILED_DIR, STATUS_DIR):
        d.mkdir(parents=True, exist_ok=True)

def list_pending_jobs():
    """Nađi sve status/*.json sa statusom 'pending'."""
    ensure_dirs()
    jobs = []
    for p in STATUS_DIR.glob("*.json"):
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
            if js.get("status") == "pending":
                jobs.append((p.stem, js))  # (job_id, payload)
        except Exception:
            continue
    return jobs

def write_status(job_id: str, status: str, extra: dict | None = None):
    ensure_dirs()
    sp = STATUS_DIR / f"{job_id}.json"
    data = {"status": status, "ts": int(time.time())}
    if extra:
        data.update(extra)
    sp.write_text(json.dumps(data, indent=2), encoding="utf-8")

def save_result(job_id: str, payload: dict):
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_dir

def move_safe(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        dst = dst_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    shutil.move(str(src), str(dst))
    return dst

def process_job(job_id: str, meta: dict):
    """
    Obradi jedan job:
      - meta["file"] = ime fajla u UPLOADS_DIR (backend ga je stavio pri uploadu)
      - meta["vid_stride"] opcionalno (default 6)
    """
    vid_name = meta.get("file")
    if not vid_name:
        write_status(job_id, "failed", {"error": "No file in status meta"})
        return

    video_path = UPLOADS_DIR / vid_name
    if not video_path.exists():
        write_status(job_id, "failed", {"error": f"Upload not found: {vid_name}"})
        return

    vid_stride = int(meta.get("vid_stride", 6))
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_status(job_id, "processing", {"file": vid_name})
    log.info("Obrada start: file=%s job_id=%s stride=%s", vid_name, job_id, vid_stride)

    try:
        res = run_pipeline(str(video_path), str(out_dir), vid_stride=vid_stride)
        save_result(job_id, res)
        write_status(job_id, "done", {"attempts": 1})
        move_safe(video_path, PROCESSED_DIR)
        log.info("Obrada OK: %s (job_id=%s)", vid_name, job_id)
    except Exception as e:
        write_status(job_id, "failed", {"error": str(e)})
        move_safe(video_path, FAILED_DIR)
        log.error("Obrada FAIL: %s (job_id=%s) err=%s", vid_name, job_id, e)

# ============ Main loop ============
def main():
    log.info("Worker startuje. Uploads=%s Results=%s Status=%s", UPLOADS_DIR, RESULTS_DIR, STATUS_DIR)
    ensure_dirs()

    while True:
        try:
            pending = list_pending_jobs()
            if not pending:
                time.sleep(POLL_SECONDS)
                continue

            for job_id, meta in pending:
                process_job(job_id, meta)

        except Exception as loop_err:
            log.error("Global error u loopu: %s", loop_err)
            time.sleep(2)

if __name__ == "__main__":
    main()
