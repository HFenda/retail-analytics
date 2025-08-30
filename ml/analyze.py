from typing import Dict, Any, List
import csv, os, math
import numpy as np
import cv2

def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    c = np.cumsum(np.insert(arr, 0, 0))
    out = c[window:] - c[:-window]
    pad = np.full(window-1, np.nan)
    return np.concatenate([pad, out])

def run_analysis(
    video_path: str,
    counts_csv: str,
    *,
    vid_stride: int,
    win_seconds: List[int] = [60, 300, 600],
    out_dir: str | None = None,
) -> Dict[str, Any]:
    """Parsiraj counts.csv, izračunaj per-sec / per-minute serije i peak prozore.
       Snimi CSV-ove (occupancy_per_sec.csv, by_minute.csv, peaks.csv) i vrati strukturiran rezime.
    """
    # 1) FPS/duration iz videa
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_frames / fps) if fps > 0 else 0.0
    cap.release()

    # 2) Učitaj counts.csv
    frames, counts_now, uniq_tot = [], [], []
    with open(counts_csv, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header and header[0] != "frame_idx":
            f.seek(0)
            r = csv.reader(f)
        for row in r:
            if len(row) < 3: continue
            try:
                frames.append(int(row[0]))
                counts_now.append(int(row[1]))
                uniq_tot.append(int(row[2]))
            except:  # ignore bad lines
                pass

    if not frames:
        return {"error": "empty_counts_csv"}

    frames = np.array(frames, dtype=int)
    counts_now = np.array(counts_now, dtype=int)
    uniq_tot = np.array(uniq_tot, dtype=int)

    # 3) Sekundne oznake
    t_sec = (frames * vid_stride) / float(fps)
    T = int(math.ceil(max(t_sec.max(), duration_sec))) if duration_sec > 0 else int(math.ceil(t_sec.max()))
    per_sec = np.zeros(T+1, dtype=float); per_sec[:] = np.nan
    sec_idx = np.floor(t_sec).astype(int)
    last_in_sec = {}
    for s, c in zip(sec_idx, counts_now):
        last_in_sec[s] = c
    for s, c in last_in_sec.items():
        if 0 <= s <= T:
            per_sec[s] = c
    cur = 0.0
    for s in range(T+1):
        if not np.isnan(per_sec[s]): cur = per_sec[s]
        per_sec[s] = cur

    # 4) KPI
    avg_occ = float(np.mean(per_sec)) if len(per_sec) else 0.0
    p50 = float(np.percentile(per_sec, 50)) if len(per_sec) else 0.0
    p90 = float(np.percentile(per_sec, 90)) if len(per_sec) else 0.0
    p95 = float(np.percentile(per_sec, 95)) if len(per_sec) else 0.0
    peak_now = int(np.max(per_sec)) if len(per_sec) else 0
    peak_now_t = int(np.argmax(per_sec)) if len(per_sec) else 0
    uniq_final = int(np.max(uniq_tot))

    # 5) Peak prozori
    peaks = []
    for w in win_seconds:
        if w <= 0 or len(per_sec) < w: continue
        sums = _rolling_sum(per_sec, w)
        idx = int(np.nanargmax(sums))
        best_sum = float(sums[idx])
        t_end = idx
        t_start = max(0, t_end - (w - 1))
        avg_in_window = best_sum / float(w)
        peaks.append({
            "window_sec": int(w),
            "t_start_sec": int(t_start),
            "t_end_sec": int(t_end),
            "sum_people": int(best_sum),
            "avg_people": round(avg_in_window, 3),
        })

    # 6) Po-minuti
    minutes = (np.arange(T+1) // 60).astype(int)
    max_min = int(minutes.max()) if len(minutes) else 0
    by_minute = []
    for m in range(max_min + 1):
        sel = per_sec[minutes == m]
        if sel.size:
            by_minute.append({"minute": int(m), "avg_count": round(float(np.mean(sel)), 3), "peak_count": int(np.max(sel))})
        else:
            by_minute.append({"minute": int(m), "avg_count": 0.0, "peak_count": 0})

    # 7) Snimi CSV-ove (ako traženo)
    out_dir = out_dir or (os.path.dirname(os.path.abspath(counts_csv)) or ".")
    occ_path = os.path.join(out_dir, "occupancy_per_sec.csv")
    bym_path = os.path.join(out_dir, "by_minute.csv")
    peaks_path = os.path.join(out_dir, "peaks.csv")

    with open(occ_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["second", "count"])
        for s, c in enumerate(per_sec): w.writerow([s, int(c)])

    with open(bym_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["minute", "avg_count", "peak_count"])
        for row in by_minute: w.writerow([row["minute"], row["avg_count"], row["peak_count"]])

    with open(peaks_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["window_sec", "t_start_sec", "t_end_sec", "sum_people", "avg_people"])
        for row in peaks: w.writerow([row["window_sec"], row["t_start_sec"], row["t_end_sec"], row["sum_people"], row["avg_people"]])

    # 8) Response za backend/frontend
    per_sec_series = [{"t": int(i), "count": int(v)} for i, v in enumerate(per_sec)]
    return {
        "kpi": {
            "unique_total": uniq_final,
            "peak_simultaneous": peak_now,
            "peak_time_sec": peak_now_t,
            "avg_occ": round(avg_occ, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "fps": float(fps),
            "duration_sec": int(duration_sec),
        },
        "series": {
            "per_sec": per_sec_series,
            "per_minute": by_minute,
            "peaks": peaks,
        },
        "files": {
            "counts_csv": os.path.abspath(counts_csv),
            "occupancy_per_sec_csv": occ_path,
            "by_minute_csv": bym_path,
            "peaks_csv": peaks_path,
        }
    }