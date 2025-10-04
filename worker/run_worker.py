# worker/run_worker.py
import os, time, json, uuid, shutil, logging, tempfile
from pathlib import Path
from typing import Dict, Any

import boto3
from botocore.client import Config

# koristimo isti processor kao backend
from backend.app.services.processor import run_pipeline

# ---------- ENV ----------
BUCKET      = os.getenv("R2_BUCKET", "")
ENDPOINT    = os.getenv("R2_ENDPOINT_URL", "")
ACCESS_KEY  = os.getenv("R2_ACCESS_KEY_ID", "")
SECRET_KEY  = os.getenv("R2_SECRET_ACCESS_KEY", "")
REGION      = os.getenv("R2_REGION", "auto")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))
YOLO_MODEL_REF = os.getenv("YOLO_MODEL_REF", "yolov8n.pt")

# layout (isti kao backend/storage.py)
PREFIX_UPLOADS = "uploads/"
PREFIX_STATUS  = "status/"
PREFIX_RESULTS = "results/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(levelname)s: %(message)s")
log = logging.getLogger("worker")

# ---------- R2 helpers ----------
def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        config=Config(signature_version="s3v4"),
    )

S3 = r2_client()

def get_json(key: str) -> Dict[str, Any] | None:
    try:
        obj = S3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except S3.exceptions.NoSuchKey:
        return None

def put_json(key: str, data: Dict[str, Any]):
    S3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(data).encode("utf-8"), ContentType="application/json")

def upload_file(local: Path, key: str, content_type: str | None = None):
    extra = {"ContentType": content_type} if content_type else {}
    S3.upload_file(str(local), BUCKET, key, ExtraArgs=extra)

def download_file(key: str, local: Path):
    local.parent.mkdir(parents=True, exist_ok=True)
    S3.download_file(BUCKET, key, str(local))

def list_status_pending() -> list[str]:
    """Vrati listu status ključeva koji su pending: status/<job_id>.json"""
    keys = []
    token = None
    while True:
        resp = S3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX_STATUS, ContinuationToken=token) if token else \
               S3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX_STATUS)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if not k.endswith(".json"):
                continue
            js = get_json(k)
            if js and js.get("status") in ("pending", "processing"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

# ---------- format response (isti kao backend) ----------
def to_response_payload(job_id: str, work: Dict[str, Any]) -> Dict[str, Any]:
    # work je rezultat run_pipeline(...)
    return {
        "job_id": work["job_id"],
        "video":       f"/files/uploads/{Path(work['video']).name}",  # url backend-a mapira /files/ na storage
        "counts_csv":  f"/files/results/{job_id}/counts.csv",
        "unique_total": work["counts"]["unique_total"],
        "peak":         work["counts"]["peak"],
        "per_sec_csv":  f"/files/results/{job_id}/analysis/per_sec.csv",
        "by_min_csv":   f"/files/results/{job_id}/analysis/by_minute.csv",
        "peaks_csv":    f"/files/results/{job_id}/analysis/peaks.csv",
        "heatmap": {
            "overlay":         f"/files/results/{job_id}/heatmap/heatmap_overlay.png",
            "colored":         f"/files/results/{job_id}/heatmap/heatmap_colored.png",
            "gray":            f"/files/results/{job_id}/heatmap/heatmap_gray.png",
            "transparent_png": f"/files/results/{job_id}/heatmap/heatmap_transparent.png",
            "preview":         f"/files/results/{job_id}/heatmap/preview.jpg",
        },
        "snapshot":     f"/files/results/{job_id}/snapshot.jpg",
    }

def process_job(status_key: str):
    """
    status json izgleda npr:
    {
      "status": "pending",
      "ts": 123456,
      "file": "124c...746c.mp4",
      "vid_stride": 6
    }
    """
    st = get_json(status_key)
    if not st: 
        return

    job_id = Path(status_key).stem  # status/<job_id>.json
    upload_name = st.get("file")
    vid_stride  = int(st.get("vid_stride", 6))
    if not upload_name:
        log.warning("Status %s nema 'file'", status_key)
        return

    # mark processing
    put_json(status_key, {**st, "status": "processing"})

    src_key = f"{PREFIX_UPLOADS}{upload_name}"
    log.info("Krecem job=%s  src=%s", job_id, src_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        local_video = tmpdir / upload_name
        download_file(src_key, local_video)

        # workdir localno
        workdir = tmpdir / job_id
        workdir.mkdir(parents=True, exist_ok=True)

        # pokreni pipeline (koristi lokalne fajlove)
        res = run_pipeline(str(local_video), str(workdir), vid_stride=vid_stride)

        # upload svih artefakata u results/<job_id>/...
        # counts
        upload_file(Path(res["counts"]["csv"]),   f"{PREFIX_RESULTS}{job_id}/counts.csv", "text/csv")

        # analysis csv
        ana = res["analysis"]["files"]
        upload_file(Path(ana["per_sec"]),  f"{PREFIX_RESULTS}{job_id}/analysis/per_sec.csv", "text/csv")
        upload_file(Path(ana["by_minute"]),f"{PREFIX_RESULTS}{job_id}/analysis/by_minute.csv", "text/csv")
        upload_file(Path(ana["peaks"]),    f"{PREFIX_RESULTS}{job_id}/analysis/peaks.csv", "text/csv")

        # heatmap imgs
        hm = res["heatmap"]["artifacts"]
        upload_file(Path(hm["overlay"]),         f"{PREFIX_RESULTS}{job_id}/heatmap/heatmap_overlay.png", "image/png")
        upload_file(Path(hm["colored"]),         f"{PREFIX_RESULTS}{job_id}/heatmap/heatmap_colored.png", "image/png")
        upload_file(Path(hm["gray"]),            f"{PREFIX_RESULTS}{job_id}/heatmap/heatmap_gray.png", "image/png")
        upload_file(Path(hm["transparent_png"]), f"{PREFIX_RESULTS}{job_id}/heatmap/heatmap_transparent.png", "image/png")
        upload_file(Path(hm["preview"]),         f"{PREFIX_RESULTS}{job_id}/heatmap/preview.jpg", "image/jpeg")

        # snapshot
        upload_file(Path(res["snapshot"]["out"]), f"{PREFIX_RESULTS}{job_id}/snapshot.jpg", "image/jpeg")

        # response.json (isti format kao backend sync)
        payload = to_response_payload(job_id, res)
        put_json(f"{PREFIX_RESULTS}{job_id}/response.json", payload)

        # status done
        put_json(status_key, {**st, "status": "done", "job_id": job_id})

    log.info("Job %s DONE", job_id)

def main():
    # sanity na env
    for k, v in {
        "R2_BUCKET": BUCKET, "R2_ENDPOINT_URL": ENDPOINT, "R2_ACCESS_KEY_ID": ACCESS_KEY,
        "R2_SECRET_ACCESS_KEY": SECRET_KEY
    }.items():
        if not v:
            log.error("ENV %s nije postavljen!", k)
    log.info("Worker startuje. Bucket=%s  Endpoint=%s", BUCKET, ENDPOINT)

    while True:
        try:
            pend = list_status_pending()
            if not pend:
                time.sleep(POLL_SECONDS)
                continue
            for sk in pend:
                # svaki status file je po jedan job (pending ili processing)
                try:
                    process_job(sk)
                except Exception as e:
                    log.exception("Greska u poslu %s: %s", sk, e)
        except Exception as e:
            log.exception("Global error: %s", e)
            time.sleep(3)

if __name__ == "__main__":
    main()
