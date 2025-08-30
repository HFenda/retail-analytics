from typing import Dict, Any, Optional
from ultralytics import YOLO
import csv, os

def run_count(
    video_path: str,
    out_csv: str,
    *,
    model_path: str = "ml/models/yolov8n.pt",
    conf: float = 0.40,
    tracker: str = "bytetrack.yaml",
    imgsz: int = 640,
    vid_stride: int = 2,
    device: str = "cpu",
    max_frames: int = 0,  # 0 = all
) -> Dict[str, Any]:
    """
    Pokreće YOLO tracking na video i snima CSV:
      frame_idx,count_now,unique_total
    Vraća statistiku za backend/worker.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    model = YOLO(model_path)

    unique_ids = set()
    peak = 0
    frame_idx = 0
    wrote_header = False

    gen = model.track(
        source=video_path,
        stream=True,
        classes=[0],
        conf=conf,
        tracker=tracker,
        verbose=False,
        imgsz=imgsz,
        vid_stride=max(1, vid_stride),
        device=device,
    )

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        for res in gen:
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            ids_now = []
            if res.boxes is not None and res.boxes.id is not None:
                ids_now = list(map(int, res.boxes.id.cpu().tolist()))
                unique_ids.update(ids_now)

            peak = max(peak, len(ids_now))

            if not wrote_header:
                w.writerow(["frame_idx", "count_now", "unique_total"])
                wrote_header = True

            w.writerow([frame_idx, len(ids_now), len(unique_ids)])

    return {
        "csv_path": out_csv,
        "frames_processed": frame_idx,
        "unique_total": len(unique_ids),
        "peak_simultaneous": peak,
        "params": {
            "conf": conf, "tracker": tracker, "imgsz": imgsz,
            "vid_stride": vid_stride, "device": device
        }
    }