# ml/peak.py
from typing import Dict, Any, Optional
import os, csv, cv2
from ultralytics import YOLO

def _read_fps(video: str) -> float:
    cap = cv2.VideoCapture(video); cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release(); return float(fps)

def _find_peak_from_counts(counts_csv: str):
    peak_frame, peak_count = None, -1
    with open(counts_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            c = int(row.get("count_now", 0)); fi = int(row.get("frame_idx", -1))
            if c > peak_count: peak_count, peak_frame = c, fi
    if peak_frame is None:
        raise RuntimeError("Nisam našao peak u counts.csv")
    return peak_frame, peak_count

def _grab_frame_at_time(video: str, t: float):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read(); cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Nisam uspio frame @ {t:.2f}s")
    return frame

def _detect_and_draw(model_path: str, frame, *, conf=0.4, imgsz=640):
    mdl = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")
    res = mdl.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    names = getattr(mdl, "names", {})
    person_ids = {k for k, v in names.items() if str(v).lower() == "person"} or None

    img = frame.copy(); persons = 0
    if res.boxes is not None:
        for b in res.boxes:
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            if (person_ids is None) or (cls_id in person_ids):
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                confv = float(b.conf[0]) if b.conf is not None else 0.0
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f"person {confv:.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        persons = len(res.boxes)
    return img, persons

def run_peak(
    video_path: str,
    counts_csv: str,
    out_path: str,
    *,
    vid_stride: int,
    model_path: str = "ml/models/yolov8n.pt",
    conf: float = 0.4,
    imgsz: int = 640,
    window: float = 0.0,  # pretraga ±window s oko peaka
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fps = _read_fps(video_path)
    peak_frame, peak_count = _find_peak_from_counts(counts_csv)
    t_peak = (peak_frame * vid_stride) / fps

    candidates = [t_peak]
    if window and window > 0:
        step = max(1.0 / fps, 0.2)
        rng = range(int(-window/step), int(window/step)+1)
        candidates = sorted({round(t_peak + k*step, 2) for k in rng})

    best_img, best_n = None, -1
    for t in candidates:
        frame = _grab_frame_at_time(video_path, t)
        img, persons = _detect_and_draw(model_path, frame, conf=conf, imgsz=imgsz)
        if persons > best_n: best_n, best_img = persons, img.copy()

    if best_img is None:  # fallback
        best_img = _grab_frame_at_time(video_path, t_peak)

    cv2.imwrite(out_path, best_img)

    return {
        "peak_image_path": out_path,
        "peak_from_counts": {"peak_frame": int(peak_frame), "peak_count_now": int(peak_count)},
        "peak_time_sec": float(t_peak),
        "detected_on_best_frame": int(max(best_n, 0)),
        "params": {
            "vid_stride": int(vid_stride), "model_path": model_path,
            "conf": float(conf), "imgsz": int(imgsz), "window": float(window),
        },
    }