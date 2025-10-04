import os, time, json, tempfile, logging
from pathlib import Path

from backend.app.services.storage import (
    IS_R2, list_pending_jobs, write_status,
    job_dir, upload_path_and_url, write_result_json
)
from backend.app.services.processor import run_pipeline

# R2 download
_s3 = None; _bucket = None
if IS_R2:
    import boto3
    from botocore.client import Config
    _access = os.getenv("R2_ACCESS_KEY_ID")
    _secret = os.getenv("R2_SECRET_ACCESS_KEY")
    _endpoint = os.getenv("R2_ENDPOINT") or os.getenv("R2_ENDPOINT_URL")
    _bucket  = os.getenv("R2_BUCKET", "retail-analytics")
    _s3 = boto3.client(
        "s3",
        aws_access_key_id=_access,
        aws_secret_access_key=_secret,
        endpoint_url=_endpoint,
        region_name=None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )

def r2_download(key: str, dest: Path):
    assert _s3 is not None and _bucket is not None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        _s3.download_fileobj(_bucket, key, f)
    return str(dest)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(levelname)s: %(message)s")
log = logging.getLogger("worker")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

def flatten_result(job_id: str, res: dict) -> dict:
    # upload svih artefakata i dobij URL-ove
    video_url    = upload_path_and_url(res["video"], key=f"results/{job_id}/video.mp4", content_type="video/mp4")
    counts_csv   = upload_path_and_url(res["counts"]["csv"], key=f"results/{job_id}/counts.csv", content_type="text/csv")
    per_sec_csv  = upload_path_and_url(res["analysis"]["files"]["per_sec"],   key=f"results/{job_id}/occupancy_per_sec.csv", content_type="text/csv")
    by_min_csv   = upload_path_and_url(res["analysis"]["files"]["by_minute"], key=f"results/{job_id}/by_minute.csv", content_type="text/csv")
    peaks_csv    = upload_path_and_url(res["analysis"]["files"]["peaks"],     key=f"results/{job_id}/peaks.csv", content_type="text/csv")

    heat = res["heatmap"]["artifacts"]
    heat_overlay = upload_path_and_url(heat["overlay"],         key=f"results/{job_id}/heatmap/overlay.png",      content_type="image/png")
    heat_colored = upload_path_and_url(heat["colored"],         key=f"results/{job_id}/heatmap/colored.png",      content_type="image/png")
    heat_gray    = upload_path_and_url(heat["gray"],            key=f"results/{job_id}/heatmap/gray.png",         content_type="image/png")
    heat_trans   = upload_path_and_url(heat["transparent_png"], key=f"results/{job_id}/heatmap/transparent.png",  content_type="image/png")
    heat_prev    = upload_path_and_url(heat["preview"],         key=f"results/{job_id}/heatmap/preview.jpg",      content_type="image/jpeg")
    snapshot_url = upload_path_and_url(res["snapshot"]["out"],  key=f"results/{job_id}/snapshot.jpg",             content_type="image/jpeg")

    return {
        "job_id": job_id,
        "video": video_url,
        "counts_csv": counts_csv,
        "unique_total": res["counts"]["unique_total"],
        "peak": res["counts"]["peak"],
        "per_sec_csv": per_sec_csv,
        "by_min_csv": by_min_csv,
        "peaks_csv": peaks_csv,
        "heatmap": {
            "overlay": heat_overlay,
            "colored": heat_colored,
            "gray": heat_gray,
            "transparent_png": heat_trans,
            "preview": heat_prev,
        },
        "snapshot": snapshot_url,
    }

def main():
    log.info("Worker up. R2=%s", "ON" if IS_R2 else "OFF")
    while True:
        try:
            pend = list_pending_jobs()
            if not pend:
                time.sleep(POLL_SECONDS)
                continue

            for job_id, st in pend:
                try:
                    log.info("Job %s: processing", job_id)
                    write_status(job_id, {**st, "status":"processing", "ts": int(time.time())})

                    # nabavi video path
                    if IS_R2 and st.get("r2_key"):
                        with tempfile.TemporaryDirectory() as td:
                            tmp = Path(td) / Path(st["r2_key"]).name
                            r2_download(st["r2_key"], tmp)
                            workdir = job_dir(job_id)
                            res = run_pipeline(str(tmp), workdir, vid_stride=int(st.get("vid_stride", 6)))
                    else:
                        # dev: backend i worker dijele lokalni storage
                        from backend.app.services.storage import BASE
                        v = Path(BASE) / "uploads" / st["file"]
                        if not v.exists():
                            raise RuntimeError(f"Upload not found: {v}")
                        workdir = job_dir(job_id)
                        res = run_pipeline(str(v), workdir, vid_stride=int(st.get("vid_stride", 6)))

                    flat = flatten_result(job_id, res)
                    write_result_json(job_id, flat)           # <- response.json u R2 (ili lokalno)
                    write_status(job_id, {**st, "status":"done", "ts": int(time.time())})
                    log.info("Job %s: done", job_id)

                except Exception as e:
                    log.exception("Job %s failed: %s", job_id, e)
                    write_status(job_id, {**st, "status":"failed", "error": str(e), "ts": int(time.time())})

        except Exception as loop_err:
            log.error("Loop error: %s", loop_err)
            time.sleep(2)

if __name__ == "__main__":
    main()
