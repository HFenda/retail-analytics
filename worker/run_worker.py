import os, time, json, uuid, shutil, logging, tempfile
from pathlib import Path

from backend.app.services.storage import (
    BASE, status_path, upload_path_and_url, job_dir, IS_R2
)
from backend.app.services.processor import run_pipeline
from backend.app.services.storage import ensure_dirs  # koristi /storage strukturu

# R2/S3 download (minimalno)
import boto3
from botocore.client import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(levelname)s: %(message)s")
log = logging.getLogger("worker")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

# R2/S3 client ako je uključen
_S3 = None
_BUCKET = None
if IS_R2:
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint   = os.getenv("R2_ENDPOINT") or os.getenv("R2_ENDPOINT_URL")
    region     = os.getenv("R2_REGION") or "auto"
    _BUCKET    = os.getenv("R2_BUCKET", "retail-analytics")
    if not (access_key and secret_key and endpoint and _BUCKET):
        raise RuntimeError("R2 env varijable nisu kompletne.")
    _S3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        region_name=None if region == "auto" else region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )

def _r2_download(key: str, dest_path: Path):
    assert _S3 is not None
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        _S3.download_fileobj(_BUCKET, key, f)
    return str(dest_path)

def _list_pending_jobs():
    sp_dir = Path(BASE) / "status"
    if not sp_dir.exists():
        return []
    out = []
    for p in sp_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("status") == "pending":
                jid = p.stem
                out.append((jid, data))
        except Exception:
            continue
    return out

def _write_status(job_id: str, payload: dict):
    sp = status_path(job_id)
    sp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _flatten_response(job_id: str, pipeline_res: dict) -> dict:
    """
    Konstruiraj izlaz IDENTIČNOG oblika kao sinkroni backend ranije,
    ali umjesto lokalnih /files/... linkova koristimo upload_path_and_url
    da dobijemo R2 (ili /files u local modu).
    """
    # Svi pathovi u pipeline_res su lokalni; uploadaj ih (ili mapiraj na /files)
    video_url     = upload_path_and_url(pipeline_res["video"], key=f"results/{job_id}/video.mp4", content_type="video/mp4")
    counts_csv    = upload_path_and_url(pipeline_res["counts"]["csv"], key=f"results/{job_id}/counts.csv", content_type="text/csv")
    per_sec_csv   = upload_path_and_url(pipeline_res["analysis"]["files"]["per_sec"],    key=f"results/{job_id}/occupancy_per_sec.csv", content_type="text/csv")
    by_min_csv    = upload_path_and_url(pipeline_res["analysis"]["files"]["by_minute"],  key=f"results/{job_id}/by_minute.csv", content_type="text/csv")
    peaks_csv     = upload_path_and_url(pipeline_res["analysis"]["files"]["peaks"],      key=f"results/{job_id}/peaks.csv", content_type="text/csv")

    heat_overlay  = upload_path_and_url(pipeline_res["heatmap"]["artifacts"]["overlay"],         key=f"results/{job_id}/heatmap/overlay.png",        content_type="image/png")
    heat_colored  = upload_path_and_url(pipeline_res["heatmap"]["artifacts"]["colored"],         key=f"results/{job_id}/heatmap/colored.png",        content_type="image/png")
    heat_gray     = upload_path_and_url(pipeline_res["heatmap"]["artifacts"]["gray"],            key=f"results/{job_id}/heatmap/gray.png",           content_type="image/png")
    heat_trans    = upload_path_and_url(pipeline_res["heatmap"]["artifacts"]["transparent_png"], key=f"results/{job_id}/heatmap/transparent.png",    content_type="image/png")
    heat_preview  = upload_path_and_url(pipeline_res["heatmap"]["artifacts"]["preview"],         key=f"results/{job_id}/heatmap/preview.jpg",        content_type="image/jpeg")

    snapshot_url  = upload_path_and_url(pipeline_res["snapshot"]["out"], key=f"results/{job_id}/snapshot.jpg", content_type="image/jpeg")

    return {
        "job_id": job_id,
        "video": video_url,
        "counts_csv": counts_csv,
        "unique_total": pipeline_res["counts"]["unique_total"],
        "peak": pipeline_res["counts"]["peak"],
        "per_sec_csv": per_sec_csv,
        "by_min_csv": by_min_csv,
        "peaks_csv": peaks_csv,
        "heatmap": {
            "overlay": heat_overlay,
            "colored": heat_colored,
            "gray": heat_gray,
            "transparent_png": heat_trans,
            "preview": heat_preview,
        },
        "snapshot": snapshot_url,
    }

def main():
    log.info("Worker startuje. BASE=%s  R2=%s", BASE, "ON" if IS_R2 else "OFF")
    ensure_dirs()

    while True:
        try:
            pend = _list_pending_jobs()
            if not pend:
                time.sleep(POLL_SECONDS)
                continue

            for job_id, st in pend:
                try:
                    log.info("Job %s: start", job_id)
                    _write_status(job_id, {**st, "status": "processing", "ts": int(time.time())})

                    # gdje je video?
                    if IS_R2 and st.get("r2_key"):
                        # skini u tmp fajl
                        with tempfile.TemporaryDirectory() as td:
                            tmp_path = Path(td) / Path(st["r2_key"]).name
                            _r2_download(st["r2_key"], tmp_path)
                            video_path = str(tmp_path)
                            workdir = job_dir(job_id)
                            res = run_pipeline(video_path, workdir, vid_stride=int(st.get("vid_stride", 6)))
                    else:
                        # local shared disk (dev docker-compose)
                        video_path = Path(BASE) / "uploads" / st["file"]
                        if not video_path.exists():
                            raise RuntimeError(f"Upload ne postoji: {video_path}")
                        workdir = job_dir(job_id)
                        res = run_pipeline(str(video_path), workdir, vid_stride=int(st.get("vid_stride", 6)))

                    flat = _flatten_response(job_id, res)

                    # snimi response.json (da backend /result/{id} može pročitati)
                    (Path(workdir) / "response.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")

                    _write_status(job_id, {**st, "status": "done", "ts": int(time.time())})
                    log.info("Job %s: done", job_id)

                except Exception as e:
                    log.exception("Job %s: fail: %s", job_id, e)
                    _write_status(job_id, {**st, "status": "failed", "error": str(e), "ts": int(time.time())})

        except Exception as loop_err:
            log.error("Global loop error: %s", loop_err)
            time.sleep(2)

if __name__ == "__main__":
    main()
