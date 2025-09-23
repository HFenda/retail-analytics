from ultralytics import YOLO
import argparse, csv, sys
from typing import Dict, Any, Optional


def run_unique_count(
    video: str,
    model: str = "yolov8n.pt",
    conf: float = 0.45,
    tracker: str = "bytetrack.yaml",
    imgsz: int = 640,
    vid_stride: int = 2,
    device: str = "cpu",
    csv_path: str = "people_counts.csv",
    max_frames: int = 0
) -> Dict[str, Any]:
    """
    Pokrene YOLO tracking (samo klasa 'person') i upiše CSV sa kolonama:
        frame_idx, count_now, unique_total

    Returns:
        {
          "csv": "<put/do/people_counts.csv>",
          "unique_total": int,
          "peak": int,
          "frames_processed": int,
          "params": {...}
        }
    """
    mdl = YOLO(model)

    unique_ids = set()
    peak = 0
    frame_idx = 0
    wrote_header = False

    gen = mdl.track(
        source=video, stream=True,
        classes=[0], conf=conf, tracker=tracker, verbose=False,
        imgsz=imgsz, vid_stride=max(1, vid_stride), device=device
    )

    with open(csv_path, "w", newline="") as f:
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

            if frame_idx % 50 == 0:
                print(f"frame {frame_idx:,}  now={len(ids_now)}  uniq={len(unique_ids)}  peak={peak}", end="\r")

    print("\n")  # da se progress red fino prelomi

    return {
        "csv": csv_path,
        "unique_total": len(unique_ids),
        "peak": peak,
        "frames_processed": frame_idx,
        "params": {
            "video": video, "model": model, "conf": conf, "tracker": tracker,
            "imgsz": imgsz, "vid_stride": vid_stride, "device": device,
            "max_frames": max_frames
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--tracker", default="bytetrack.yaml")   # or botsort.yaml
    ap.add_argument("--imgsz", type=int, default=640)        # YOLO input size
    ap.add_argument("--vid_stride", type=int, default=2)     # skip frames at decode
    ap.add_argument("--device", default="cpu")               # "cpu" or "0" for GPU
    ap.add_argument("--csv", default="people_counts.csv")    # output CSV
    ap.add_argument("--max_frames", type=int, default=0)     # 0 = all
    args = ap.parse_args()

    res = run_unique_count(
        video=args.video,
        model=args.model,
        conf=args.conf,
        tracker=args.tracker,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        device=args.device,
        csv_path=args.csv,
        max_frames=args.max_frames
    )

    print("Summary:")
    print(f"  Unique people total: {res['unique_total']}")
    print(f"  Peak simultaneous:   {res['peak']}")
    print(f"  CSV saved to:        {res['csv']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
