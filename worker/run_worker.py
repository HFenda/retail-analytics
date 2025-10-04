import os
import time
import json
import uuid
import shutil
import logging
from pathlib import Path

import requests

# ============ Config iz ENV-a ============
BACKEND_URL   = os.getenv("BACKEND_URL", "http://backend:8000")
UPLOADS_DIR   = Path(os.getenv("UPLOADS_DIR", "/data/uploads"))
RESULTS_DIR   = Path(os.getenv("RESULTS_DIR", "/data/results"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "/data/processed"))
FAILED_DIR    = Path(os.getenv("FAILED_DIR", "/data/failed"))
POLL_SECONDS  = int(os.getenv("POLL_SECONDS", "3"))
RETRY_MAX     = int(os.getenv("RETRY_MAX", "3"))

# Ako backend treba model ref, šalje se kao form field
YOLO_MODEL_REF = os.getenv("YOLO_MODEL_REF", "yolov8n.pt")

# Ekstenzije koje pratimo
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".mpeg", ".mpg", ".webm", ".m4v", ".ts"}

# ============ Logging ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s: %(message)s"
)
log = logging.getLogger("worker")

# ============ Helpers ============
def ensure_dirs():
    for d in (UPLOADS_DIR, RESULTS_DIR, PROCESSED_DIR, FAILED_DIR):
        d.mkdir(parents=True, exist_ok=True)

def claim_file(p: Path) -> Path:
    """Stavi .lock pored fajla da drugi krug ne pokupi isti fajl."""
    lock = p.with_suffix(p.suffix + ".lock")
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        pass
    return lock

def release_file(p: Path):
    lock = p.with_suffix(p.suffix + ".lock")
    if lock.exists():
        lock.unlink(missing_ok=True)

def list_new_videos():
    return [
        p for p in UPLOADS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and not (p.with_suffix(p.suffix + ".lock")).exists()
    ]

def post_analyze(video_path: Path) -> dict:
    """
    Pošalji video backendu. Ovdje pretpostavljam da postoji POST /api/v1/analyze
    koji prima multipart field "video". Ako si endpoint nazvao drugačije, promijeni ispod.
    """
    url = f"{BACKEND_URL}/api/v1/analyze"
    files = {"file": (video_path.name, open(video_path, "rb"), "video/mp4")}
    data  = {"model_ref": YOLO_MODEL_REF}  # ako backend to koristi; ako ne, ukloni

    try:
        r = requests.post(url, files=files, data=data, timeout=300)
        r.raise_for_status()
        return r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw": r.text}
    finally:
        files["video"][1].close()

def save_result(job_id: str, payload: dict):
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_dir

def mark_status(out_dir: Path, status: str, extra: dict | None = None):
    status_path = out_dir / "status.json"
    data = {"status": status, "ts": int(time.time())}
    if extra:
        data.update(extra)
    status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def move_safe(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    # ako već postoji isto ime, dodaj sufiks
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        dst = dst_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    shutil.move(str(src), str(dst))
    return dst

# ============ Main loop ============
def main():
    log.info("Worker startuje. Uploads=%s  Results=%s  Backend=%s", UPLOADS_DIR, RESULTS_DIR, BACKEND_URL)
    ensure_dirs()

    while True:
        try:
            candidates = list_new_videos()
            if not candidates:
                time.sleep(POLL_SECONDS)
                continue

            for video in candidates:
                lock = claim_file(video)
                job_id = uuid.uuid4().hex[:12]
                out_dir = RESULTS_DIR / job_id
                out_dir.mkdir(parents=True, exist_ok=True)
                log.info("Obrada start: %s  (job_id=%s)", video.name, job_id)
                mark_status(out_dir, "processing", {"file": video.name})

                attempt = 0
                ok = False
                last_err = None
                while attempt < RETRY_MAX and not ok:
                    attempt += 1
                    try:
                        resp = post_analyze(video)
                        save_result(job_id, resp)
                        mark_status(out_dir, "done", {"attempts": attempt})
                        move_safe(video, PROCESSED_DIR)
                        ok = True
                        log.info("Obrada OK: %s  (job_id=%s, attempt=%d)", video.name, job_id, attempt)
                    except Exception as e:
                        last_err = str(e)
                        log.warning("Pokusaj %d/ %d failed za %s: %s", attempt, RETRY_MAX, video.name, last_err)
                        time.sleep(2)

                if not ok:
                    mark_status(out_dir, "failed", {"error": last_err})
                    move_safe(video, FAILED_DIR)
                    log.error("Obrada FAIL: %s  (job_id=%s)  err=%s", video.name, job_id, last_err)

                release_file(video)

        except Exception as loop_err:
            # globalni guard da petlja ne umre
            log.error("Global error u loopu: %s", loop_err)
            time.sleep(2)

if __name__ == "__main__":
    main()

