import os, time, json, uuid, tempfile, logging
from pathlib import Path
import boto3
from botocore.client import Config
import mimetypes

# === R2 env ===
R2_ACCOUNT_ID    = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY    = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET        = os.getenv("R2_BUCKET", "retail-analytics")
R2_ENDPOINT      = os.getenv("R2_ENDPOINT")  # https://<account>.r2.cloudflarestorage.com

# === Worker tunables ===
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))
VID_STRIDE   = int(os.getenv("VID_STRIDE", "6"))
YOLO_MODEL_REF = os.getenv("YOLO_MODEL_REF", "yolov8n.pt")

# === ML imports (NE diraj svoj ml/) ===
from ml import (
    run_unique_count,
    run_analysis,
    run_people_heatmap,
    run_snapshot_peak,
)

# === R2 client ===
_sess = boto3.session.Session()
s3 = _sess.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

def k_uploads(name:str) -> str: return f"uploads/{name}"
def k_results(job_id:str, rel:str) -> str: return f"results/{job_id}/{rel.lstrip('/')}"
def k_status(job_id:str) -> str: return f"status/{job_id}.json"

def put_json(key:str, payload:dict):
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=json.dumps(payload).encode("utf-8"),
                  ContentType="application/json")

def put_file(key:str, path:str, content_type:str|None=None):
    if not content_type:
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=f, ContentType=content_type)

def get_json(key:str) -> dict:
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def obj_exists(key:str) -> bool:
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=key); return True
    except Exception:
        return False

def list_pending_jobs():
    # status/<job>.json == {"status":"pending", "file":"<upload_name>", "vid_stride":6, ...}
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix="status/")
    out = []
    for it in resp.get("Contents", []):
        key = it["Key"]
        try:
            st = get_json(key)
            if st.get("status") == "pending":
                job_id = Path(key).stem
                out.append((job_id, st))
        except Exception:
            pass
    return out

def download_upload(upload_name:str) -> str:
    key = k_uploads(upload_name)
    fd, tmp = tempfile.mkstemp(suffix=Path(upload_name).suffix or ".mp4")
    os.close(fd)
    with open(tmp, "wb") as f:
        s3.download_fileobj(R2_BUCKET, key, f)
    return tmp

def process(job_id:str, upload_name:str, vid_stride:int):
    tmp_video = download_upload(upload_name)
    work = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
    # 1) counts
    counts_csv = str(work / "counts.csv")
    uniq = run_unique_count(video=tmp_video, csv_path=counts_csv, vid_stride=vid_stride)
    # 2) analysis
    ana = run_analysis(video=tmp_video, counts_csv=counts_csv, vid_stride=vid_stride)
    # 3) heatmap
    heat_dir = str(work / "heatmap")
    heat = run_people_heatmap(video=tmp_video, outdir=heat_dir)
    # 4) snapshot
    snap = str(work / "snapshot.jpg")
    run_snapshot_peak(video=tmp_video, counts_csv=counts_csv, vid_stride=vid_stride, out=snap)

    # Upload artefakte u results/<job>/…
    put_file(k_results(job_id, "counts.csv"), counts_csv, "text/csv")
    put_file(k_results(job_id, "occupancy_per_sec.csv"), ana["files"]["per_sec"], "text/csv")
    put_file(k_results(job_id, "by_minute.csv"), ana["files"]["by_minute"], "text/csv")
    put_file(k_results(job_id, "peaks.csv"), ana["files"]["peaks"], "text/csv")
    put_file(k_results(job_id, "heatmap_overlay.png"), heat["artifacts"]["overlay"], "image/png")
    put_file(k_results(job_id, "heatmap_colored.png"), heat["artifacts"]["colored"], "image/png")
    put_file(k_results(job_id, "heatmap_gray.png"),    heat["artifacts"]["gray"],    "image/png")
    put_file(k_results(job_id, "heatmap_transparent.png"), heat["artifacts"]["transparent_png"], "image/png")
    put_file(k_results(job_id, "preview.jpg"), heat["artifacts"]["preview"], "image/jpeg")
    put_file(k_results(job_id, "snapshot.jpg"), snap, "image/jpeg")

    # response.json (RELATIVNI ključevi – backend će ih presignati)
    response = {
        "counts_csv":        k_results(job_id, "counts.csv"),
        "per_sec_csv":       k_results(job_id, "occupancy_per_sec.csv"),
        "by_min_csv":        k_results(job_id, "by_minute.csv"),
        "peaks_csv":         k_results(job_id, "peaks.csv"),
        "heatmap": {
            "overlay":         k_results(job_id, "heatmap_overlay.png"),
            "colored":         k_results(job_id, "heatmap_colored.png"),
            "gray":            k_results(job_id, "heatmap_gray.png"),
            "transparent_png": k_results(job_id, "heatmap_transparent.png"),
            "preview":         k_results(job_id, "preview.jpg"),
        },
        "snapshot":          k_results(job_id, "snapshot.jpg"),
        "unique_total":      uniq["unique_total"],
        "peak":              uniq["peak"],
    }
    put_json(k_results(job_id, "response.json"), response)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(levelname)s: %(message)s")
    log = logging.getLogger("worker")
    log.info("Worker up. Poll=%ss  R2=%s", POLL_SECONDS, R2_BUCKET)

    while True:
        try:
            pend = list_pending_jobs()
            if not pend:
                time.sleep(POLL_SECONDS); continue

            for job_id, st in pend:
                try:
                    log.info("Job %s start (file=%s, stride=%s)", job_id, st["file"], st.get("vid_stride", VID_STRIDE))
                    put_json(k_status(job_id), {**st, "status":"processing", "ts": int(time.time())})
                    process(job_id, st["file"], int(st.get("vid_stride", VID_STRIDE)))
                    put_json(k_status(job_id), {**st, "status":"done", "ts": int(time.time())})
                    log.info("Job %s done", job_id)
                except Exception as e:
                    log.exception("Job %s failed: %s", job_id, e)
                    put_json(k_status(job_id), {**st, "status":"failed", "error": str(e), "ts": int(time.time())})
        except Exception as loop_err:
            logging.exception("Loop error: %s", loop_err)
            time.sleep(2)

if __name__ == "__main__":
    main()
