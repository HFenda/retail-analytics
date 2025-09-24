# worker/tasks.py
from pathlib import Path
import json
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple
from ultralytics import YOLO

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_csv(path: Path, rows: list, header: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

def _overlay_heatmap_on_frame(frame: np.ndarray, heatmap_gray: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    heat_norm = cv2.normalize(heatmap_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0, heat_color, alpha, 0)

def _transparent_from_heatmap(heatmap_gray: np.ndarray) -> np.ndarray:
    heat_norm = cv2.normalize(heatmap_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    b, g, r = cv2.split(heat_color)
    a = heat_norm
    return cv2.merge([b, g, r, a])

def process_video(video_path: str, vid_stride: int, out_dir: Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Ne mogu otvoriti video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    model = YOLO("yolov8n.pt")  # CPU
    person_class_id = 0

    counts_rows = []  # (frame_idx, sec, count)
    counts_per_sec = defaultdict(int)   # sec -> max count
    counts_per_min = defaultdict(int)   # min -> max count
    peak_count = -1
    peak_frame = None
    peak_sec = 0

    heat_accum = np.zeros((height, width), dtype=np.float32)
    first_frame_for_overlay = None

    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % max(1, vid_stride) != 0:
            frame_idx += 1
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break
        if first_frame_for_overlay is None:
            first_frame_for_overlay = frame.copy()

        res = model.predict(frame, verbose=False, device="cpu")[0]
        count = 0
        if res.boxes is not None and len(res.boxes) > 0:
            cls = res.boxes.cls.cpu().numpy().astype(int).tolist()
            conf = res.boxes.conf.cpu().numpy().tolist()
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            for i, c in enumerate(cls):
                if c == person_class_id and conf[i] >= 0.25:
                    count += 1
                    x1, y1, x2, y2 = xyxy[i]
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    heat_accum[y1:y2, x1:x2] += 1.0

        sec = int(round(frame_idx / max(1.0, fps)))
        minute = sec // 60

        counts_rows.append((frame_idx, sec, count))
        counts_per_sec[sec] = max(counts_per_sec[sec], count)
        counts_per_min[minute] = max(counts_per_min[minute], count)

        if count > peak_count:
            peak_count = count
            peak_frame = frame.copy()
            peak_sec = sec

        frame_idx += 1

    cap.release()

    # CSV
    _write_csv(out_dir / "counts.csv", counts_rows, "frame,second,count")
    sec_rows = sorted((s, c) for s, c in counts_per_sec.items())
    _write_csv(out_dir / "occupancy_per_sec.csv", [(s, c) for s, c in sec_rows], "second,count")
    min_rows = sorted((m, c) for m, c in counts_per_min.items())
    _write_csv(out_dir / "by_minute.csv", [(m, c) for m, c in min_rows], "minute,count")
    _write_csv(out_dir / "peaks.csv", [(peak_sec, peak_count)], "second,peak_count")

    # Snapshot
    snapshot_path = out_dir / "snapshot.jpg"
    if peak_frame is not None:
        cv2.imwrite(str(snapshot_path), peak_frame)
    else:
        cap2 = cv2.VideoCapture(video_path)
        ok, frm = cap2.read()
        cap2.release()
        if ok:
            cv2.imwrite(str(snapshot_path), frm)

    # Heatmapi
    heat_gray = heat_accum.copy()
    if heat_gray.max() > 0:
        heat_gray = heat_gray / heat_gray.max()
    heat_gray_u8 = (heat_gray * 255.0).astype(np.uint8)

    heat_gray_path = out_dir / "heatmap_gray.png"
    cv2.imwrite(str(heat_gray_path), heat_gray_u8)

    heat_colored = cv2.applyColorMap(heat_gray_u8, cv2.COLORMAP_JET)
    heat_colored_path = out_dir / "heatmap_colored.png"
    cv2.imwrite(str(heat_colored_path), heat_colored)

    heat_transparent = _transparent_from_heatmap(heat_gray_u8)
    heat_transp_path = out_dir / "heatmap_transparent.png"
    cv2.imwrite(str(heat_transp_path), heat_transparent)   # ← ispravno

    overlay_path = out_dir / "heatmap_overlay.png"
    if first_frame_for_overlay is None:
        cap3 = cv2.VideoCapture(video_path)
        ok, frm = cap3.read()
        cap3.release()
        first_frame_for_overlay = frm if ok else np.zeros((max(1, int(height)), max(1, int(width)), 3), dtype=np.uint8)
    overlay_img = _overlay_heatmap_on_frame(first_frame_for_overlay, heat_gray_u8, alpha=0.6)
    cv2.imwrite(str(overlay_path), overlay_img)

    unique_total = int(peak_count if peak_count >= 0 else 0)

    def to_url(p: Path) -> str:
        data_root = Path("/data").resolve()
        return "/files/" + str(p.resolve().relative_to(data_root)).replace("\\", "/")

    result = {
        "message": "processed (yolo)",
        "unique_total": unique_total,
        "peak": int(peak_count if peak_count >= 0 else 0),
        "snapshot": to_url(snapshot_path) if snapshot_path.exists() else None,
        "counts_csv": to_url(out_dir / "counts.csv"),
        "per_sec_csv": to_url(out_dir / "occupancy_per_sec.csv"),
        "by_min_csv": to_url(out_dir / "by_minute.csv"),
        "peaks_csv": to_url(out_dir / "peaks.csv"),
        "heatmap": {
            "overlay": to_url(overlay_path),
            "colored": to_url(heat_colored_path),
            "gray": to_url(heat_gray_path),
            "transparent_png": to_url(heat_transp_path),
        },
    }

    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
