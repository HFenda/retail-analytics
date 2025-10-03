from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import uuid

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
        "created_at": datetime.utcnow().isoformat()+"Z",
        "video": video_path,
        "counts": uniq,
        "analysis": ana,
        "heatmap": heat,
        "snapshot": snap,
    }

def new_job_id() -> str:
    return uuid.uuid4().hex
