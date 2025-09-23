import argparse, csv, math, os
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

def rolling_sum(arr, window):
    if window <= 1:
        return arr.copy()
    c = np.cumsum(np.insert(arr, 0, 0))
    out = c[window:] - c[:-window]
    pad = np.full(window-1, np.nan)
    return np.concatenate([pad, out])

def _fmt_time(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def run_analysis(
    video: str,
    counts_csv: str = "counts.csv",
    vid_stride: int = 2,
    win_seconds: List[int] = [60, 300, 600],
) -> Dict[str, Any]:
    """
    Izračuna per-second okupiranost, po-minutne agregacije i 'peak prozore'.
    Piše 3 CSV fajla pored `counts_csv` i vraća rezime + putanje izlaza.

    Returns:
        {
          "fps": float,
          "duration_sec": int,
          "unique_total": int,
          "avg_per_sec": float,
          "p50": float, "p90": float, "p95": float,
          "peak_now": int, "peak_now_t": int,
          "windows": List[Tuple[window_sec, t_start, t_end, sum_people, avg_people]],
          "files": {"per_sec": ".../occupancy_per_sec.csv",
                    "by_minute": ".../by_minute.csv",
                    "peaks": ".../peaks.csv"}
        }
    """
    # 1) FPS iz videa
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = int((total_frames / fps)) if fps > 0 else 0
    cap.release()

    # 2) Učitavanje counts.csv (frame_idx, count_now, unique_total)
    frames, counts_now, uniq_tot = [], [], []
    with open(counts_csv, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header and header[0] != "frame_idx":
            f.seek(0)
            r = csv.reader(f)
        for row in r:
            if not row or len(row) < 3:
                continue
            try:
                frames.append(int(row[0]))
                counts_now.append(int(row[1]))
                uniq_tot.append(int(row[2]))
            except:
                pass

    if not frames:
        raise RuntimeError("Nema podataka u counts.csv. Provjeri format.")

    frames = np.array(frames, dtype=int)
    counts_now = np.array(counts_now, dtype=int)
    uniq_tot = np.array(uniq_tot, dtype=int)

    # 3) Vremenske oznake za svaku izmjeru (sekunde)
    t_sec = (frames * int(vid_stride)) / float(fps)

    # 4) Agregacija po sekundi (forward-fill zadnje poznate vrijednosti)
    T = int(math.ceil(max(t_sec.max(), float(duration_sec)))) if duration_sec > 0 else int(math.ceil(t_sec.max()))
    per_sec = np.zeros(T+1, dtype=float)
    per_sec[:] = np.nan

    sec_idx = np.floor(t_sec).astype(int)
    last_in_sec = {}
    for s, c in zip(sec_idx, counts_now):
        last_in_sec[s] = c
    for s, c in last_in_sec.items():
        if 0 <= s <= T:
            per_sec[s] = c
    cur = 0.0
    for s in range(T+1):
        if not np.isnan(per_sec[s]):
            cur = per_sec[s]
        per_sec[s] = cur

    # 5) Osnovne metrike
    avg_occ = float(np.mean(per_sec)) if len(per_sec) else 0.0
    p50 = float(np.percentile(per_sec, 50)) if len(per_sec) else 0.0
    p90 = float(np.percentile(per_sec, 90)) if len(per_sec) else 0.0
    p95 = float(np.percentile(per_sec, 95)) if len(per_sec) else 0.0
    peak_now = int(np.max(per_sec)) if len(per_sec) else 0
    peak_now_t = int(np.argmax(per_sec)) if len(per_sec) else 0
    uniq_final = int(np.max(uniq_tot))

    # 6) Peak prozori
    windows: List[Tuple[int,int,int,float,float]] = []
    for w in win_seconds:
        if w <= 0 or len(per_sec) < w:
            continue
        sums = rolling_sum(per_sec, w)
        idx = int(np.nanargmax(sums))
        best_sum = float(sums[idx])
        t_end = idx
        t_start = max(0, t_end - (w - 1))
        avg_in_window = best_sum / float(w)
        windows.append((int(w), int(t_start), int(t_end), best_sum, avg_in_window))

    # 7) Po-minutna agregacija
    minutes = (np.arange(T+1) // 60).astype(int)
    max_min = int(minutes.max()) if len(minutes) else 0
    by_min = []
    for m in range(max_min + 1):
        sel = per_sec[minutes == m]
        if sel.size:
            by_min.append((m, float(np.mean(sel)), float(np.max(sel))))
        else:
            by_min.append((m, 0.0, 0.0))

    out_dir = os.path.dirname(os.path.abspath(counts_csv)) or "."
    per_sec_path   = os.path.join(out_dir, "occupancy_per_sec.csv")
    by_minute_path = os.path.join(out_dir, "by_minute.csv")
    peaks_path     = os.path.join(out_dir, "peaks.csv")

    # 8) Snimi CSV-ove
    with open(per_sec_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["second", "count"])
        for s, c in enumerate(per_sec):
            w.writerow([s, int(c)])

    with open(by_minute_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minute", "avg_count", "peak_count"])
        for m, avgc, pk in by_min:
            w.writerow([m, round(avgc, 3), int(pk)])

    with open(peaks_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_sec", "t_start_sec", "t_end_sec", "sum_people", "avg_people"])
        for wsec, ts, te, ssum, aavg in windows:
            w.writerow([wsec, ts, te, int(ssum), round(aavg, 3)])

    # Rezime za povrat
    return {
        "fps": float(fps),
        "duration_sec": int(duration_sec),
        "unique_total": int(uniq_final),
        "avg_per_sec": float(avg_occ),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "peak_now": int(peak_now),
        "peak_now_t": int(peak_now_t),           
        "peak_now_t_hms": _fmt_time(int(peak_now_t)),
        "windows": windows,                     
        "files": {
            "per_sec": per_sec_path,
            "by_minute": by_minute_path,
            "peaks": peaks_path,
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Originalni video (radi FPS-a)")
    ap.add_argument("--counts", default="counts.csv", help="CSV iz count_people_unique.py")
    ap.add_argument("--vid_stride", type=int, required=True, help="Isti vid_stride kao pri brojanju")
    ap.add_argument("--win_seconds", type=int, nargs="+", default=[60, 300, 600],
                    help="Prozori za peak (sekunde): 60=1min, 300=5min, 600=10min")
    args = ap.parse_args()

    summary = run_analysis(
        video=args.video,
        counts_csv=args.counts,
        vid_stride=args.vid_stride,
        win_seconds=args.win_seconds,
    )

    print("\n===== Rezime popunjenosti =====")
    print(f"Video FPS:           {summary['fps']:.2f}")
    print(f"Dužina (s):          {summary['duration_sec']}  (~{_fmt_time(int(summary['duration_sec']))})")
    print(f"Jedinstveni ljudi:   {summary['unique_total']}")
    print(f"Prosjek (po sekundi): {summary['avg_per_sec']:.2f}")
    print(f"Medijan P50:         {summary['p50']:.2f}")
    print(f"P90 / P95:           {summary['p90']:.2f} / {summary['p95']:.2f}")
    print(f"Peak trenutni:       {summary['peak_now']} @ {summary['peak_now_t_hms']}")

    for wsec, ts, te, ssum, aavg in summary["windows"]:
        print(f"Najprometniji prozor {wsec}s: {_fmt_time(ts)}–{_fmt_time(te)}  | suma={int(ssum)}  avg={aavg:.2f}")

    print("\nSnimiо:")
    print(" - occupancy_per_sec.csv")
    print(" - by_minute.csv")
    print(" - peaks.csv")

if __name__ == "__main__":
    main()
