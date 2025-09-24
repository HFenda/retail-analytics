# backend/app/background_queue.py
import os
import time
import json
import queue
import shutil
import traceback
import threading
from pathlib import Path
from typing import Any, Dict

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
RESULTS_DIR = DATA_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_TASK_Q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_STARTED = False
_LOCK = threading.Lock()

TASK_FQN = os.getenv("TASK_FQN", "")  # npr. "worker.tasks.process_video"
# Ako TASK_FQN nije postavljen, koristimo dummy task za smoke-test.

def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)

def _write_status(job_id: str, **fields):
    status_path = RESULTS_DIR / job_id / "status.json"
    existing: Dict[str, Any] = {}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
        except Exception:
            existing = {}
    existing.update(fields)
    _atomic_write_json(status_path, existing)

def _resolve_task():
    if not TASK_FQN:
        return _default_task
    mod_name, func_name = TASK_FQN.rsplit(".", 1)
    mod = __import__(mod_name, fromlist=[func_name])
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"TASK_FQN '{TASK_FQN}' nije callable")
    return fn

def _default_task(video_path: str, vid_stride: int, out_dir: Path) -> Dict[str, Any]:
    """
    Minimalni smoke-test task:
    - malo 'spava'
    - kopira ulazni fajl u results/
    - napiše result.json
    Zamijeni ovo svojom pravom obradom kad poželiš.
    """
    time.sleep(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(video_path)
    if src.exists():
        shutil.copy2(src, out_dir / src.name)
    result = {"message": "processed (dummy)", "copied": src.name if src.exists() else None}
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result

def enqueue_for_processing(job_id: str, video_path: str, vid_stride: int) -> None:
    _TASK_Q.put({
        "job_id": job_id,
        "video_path": video_path,
        "vid_stride": int(vid_stride),
    })

def _consumer():
    task = _resolve_task()
    while True:
        item = _TASK_Q.get()
        if item is None:
            break
        job_id = item["job_id"]
        video_path = item["video_path"]
        vid_stride = item["vid_stride"]
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            _write_status(job_id, status="started", started_at=time.time())
            # Pozovi stvarni task
            result = task(video_path=video_path, vid_stride=vid_stride, out_dir=job_dir) \
                     if "out_dir" in task.__code__.co_varnames else \
                     task(video_path, vid_stride, job_dir)

            # upiši result.json ako ga task nije napisao
            res_path = job_dir / "result.json"
            if not res_path.exists():
                res_path.write_text(json.dumps(result if isinstance(result, dict) else {"result": str(result)}, ensure_ascii=False, indent=2))

            _write_status(job_id,
                          status="finished",
                          ended_at=time.time(),
                          results_url=f"/files/results/{job_id}/")
        except Exception as e:
            _write_status(job_id,
                          status="failed",
                          ended_at=time.time(),
                          error=str(e),
                          traceback=traceback.format_exc())
        finally:
            _TASK_Q.task_done()

def start_consumer():
    global _STARTED
    if _STARTED:
        return
    t = threading.Thread(target=_consumer, name="inmem-consumer", daemon=True)
    t.start()
    _STARTED = True
