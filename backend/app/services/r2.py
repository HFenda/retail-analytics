import os, json, mimetypes
from datetime import timedelta
import boto3
from botocore.client import Config

R2_ACCOUNT_ID    = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY    = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET        = os.getenv("R2_BUCKET", "retail-analytics")
R2_ENDPOINT      = os.getenv("R2_ENDPOINT")

_sess = boto3.session.Session()
s3 = _sess.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

def k_uploads(fname: str) -> str: return f"uploads/{fname}"
def k_results(job_id: str, rel: str) -> str: return f"results/{job_id}/{rel.lstrip('/')}"
def k_status(job_id: str) -> str: return f"status/{job_id}.json"

def put_file(key: str, path: str, content_type: str | None = None):
    if not content_type:
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=f, ContentType=content_type)

def put_json(key: str, payload: dict):
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=json.dumps(payload).encode("utf-8"),
                  ContentType="application/json")

def get_json(key: str) -> dict:
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def presign_get(key: str, ttl: int = 3600) -> str:
    return s3.generate_presigned_url("get_object",
        Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=ttl)
