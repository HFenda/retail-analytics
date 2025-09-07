# snapshot_peak.py
import argparse, csv, os
from pathlib import Path
import cv2
from ultralytics import YOLO
from typing import Dict, Any, List

def read_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Ne mogu otvoriti video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)

def find_peak_from_counts(counts_csv: str):
    # očekuje kolone: frame_idx,count_now,unique_total
    peak_frame, peak_count = None, -1
    with open(counts_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            cnt = int(row.get("count_now", 0))
            fi  = int(row.get("frame_idx", -1))
            if cnt > peak_count:
                peak_count, peak_frame = cnt, fi
    if peak_frame is None:
        raise RuntimeError("Nisam našao peak u counts.csv")
    return peak_frame, peak_count

def frame_to_seconds(frame_idx: int, fps: float, stride: int) -> float:
    # frame_idx je broj "procesiranog" frame-a (poslije stride-a)
    return (frame_idx * stride) / fps

def grab_frame_at_time(video_path: str, t_seconds: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Ne mogu otvoriti video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target = int(round(t_seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Nisam uspio pročitati frame na {t_seconds:.2f}s (#{target})")
    return frame

def detect_and_draw(model_path: str, frame, conf: float = 0.4, imgsz: int = 640):
    model = YOLO(model_path)
    # jednokadarska detekcija samo za 'person' klasu
    res = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    # ako model ima names, nađi ID za 'person'
    names = model.names if hasattr(model, "names") else {}
    person_ids = [k for k, v in names.items() if str(v).lower() == "person"]
    person_ids = set(person_ids) if person_ids else None

    img = frame.copy()
    if res.boxes is not None:
        for b in res.boxes:
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            if (person_ids is None) or (cls_id in person_ids):
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                confv = float(b.conf[0]) if b.conf is not None else 0.0
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person {confv:.2f}"
                cv2.putText(img, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return img, res

# ---------------- public runner ----------------

def run_snapshot_peak(
    video: str,
    counts_csv: str,
    vid_stride: int,
    out: str = "snapshot_peak.jpg",
    model: str = "yolov8n.pt",
    conf: float = 0.4,
    imgsz: int = 640,
    window: float = 0.0
) -> Dict[str, Any]:
    """
    Nađe najprometniji trenutak iz counts CSV-a i snimi snapshot s bboxovima.
    Ako je window > 0, skenira ±window sekundi oko peaka i bira kadar s najviše detekcija.

    Returns:
        {
          "out": "path/to/snapshot.jpg",
          "peak": {"frame": int, "count": int, "t_sec": float},
          "searched_times": [float, ...],
          "best_count": int
        }
    """
    fps = read_fps(video)
    peak_frame, peak_count = find_peak_from_counts(counts_csv)
    t_peak = frame_to_seconds(peak_frame, fps, int(vid_stride))

    # kandidati za fino traženje oko peaka
    candidates: List[float] = [t_peak]
    if window and window > 0:
        step = max(1.0 / fps, 0.2)  # ~5 fps ili finije
        t = float(window)
        sweep = [t_peak + dt for dt in [x * step for x in range(int(-t/step), int(t/step) + 1)]]
        candidates = sorted(set([round(x, 2) for x in sweep]))

    best_img, best_n = None, -1
    for t in candidates:
        frame = grab_frame_at_time(video, t)
        img, res = detect_and_draw(model, frame, conf=conf, imgsz=imgsz)
        persons = len(res.boxes) if (res.boxes is not None) else 0
        if persons > best_n:
            best_n = persons
            best_img = img.copy()

    if best_img is None:
        # fallback: makar snimi „sirovi” frame na t_peak
        best_img = grab_frame_at_time(video, t_peak)

    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, best_img)

    return {
        "out": out,
        "peak": {"frame": int(peak_frame), "count": int(peak_count), "t_sec": float(t_peak)},
        "searched_times": candidates,
        "best_count": int(best_n)
    }

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Snimi snapshot najprometnijeg trenutka sa boxovima.")
    ap.add_argument("--video", required=True, help="Put do video fajla")
    ap.add_argument("--counts", required=True, help="counts.csv (frame_idx,count_now,unique_total)")
    ap.add_argument("--vid_stride", type=int, required=True, help="Stride korišten u brojanju (npr. 2, 6)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    ap.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Input size")
    ap.add_argument("--window", type=float, default=0.0,
                    help="(Opcionalno) pretraži ±window sekundi oko peaka i uzmi kadar sa najviše detekcija")
    ap.add_argument("--out", default="snapshot_peak.jpg", help="Output slika")
    args = ap.parse_args()

    res = run_snapshot_peak(
        video=args.video,
        counts_csv=args.counts,
        vid_stride=args.vid_stride,
        out=args.out,
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        window=args.window
    )

    print(f"[OK] Sačuvano: {res['out']}")
    print(f"  Peak iz counts.csv: frame={res['peak']['frame']}, count_now={res['peak']['count']}, t≈{res['peak']['t_sec']:.2f}s")
    print(f"  Pronađeno osoba na najboljem kadru: {res['best_count']}")

if __name__ == "__main__":
    main()
