import argparse, os, cv2, numpy as np, time
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from scipy.ndimage import gaussian_filter
    SCIPY = True
except Exception:
    SCIPY = False

from ultralytics import YOLO


# ---------------- helpers ----------------

def read_metadata(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return w, h, n, fps

def _fmt_time(sec):
    if sec is None or sec == float("inf"):
        return "--:--"
    sec = int(max(0, sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _progress(prefix, cur, total, start_time, every_sec=0.5):
    if total <= 0:
        total = cur if cur > 0 else 1
    now = time.time()
    if not hasattr(_progress, "_last"):
        _progress._last = 0.0
    if now - _progress._last < every_sec and cur < total:
        return
    _progress._last = now

    pct = min(100.0, 100.0 * cur / total)
    elapsed = now - start_time
    speed = cur / elapsed if elapsed > 0 else 0.0
    remain = (total - cur) / speed if speed > 0 else float("inf")
    bar_len = 28
    filled = int(bar_len * pct / 100.0)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"{prefix} [{bar}] {pct:5.1f}%  {cur}/{total}  ETA { _fmt_time(remain) }", end="\r", flush=True)

def _progress_done():
    print()


# -------------- background ----------------

def compute_background(video_path, sample_step=30, resize_w=None, mode="median", show_progress=True):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Ne mogu otvoriti video za pozadinu.")

    if mode == "first":
        if show_progress:
            print("Background: mode=first ...", flush=True)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Ne mogu pročitati prvi frame za pozadinu.")
        if resize_w and frame.shape[1] > resize_w:
            scale = resize_w / float(frame.shape[1])
            frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        if show_progress:
            print("Background: done.")
        return frame

    w, h, n, _ = read_metadata(cap)
    if resize_w is not None and resize_w < w:
        scale = resize_w / float(w)
        size = (int(w*scale), int(h*scale))
    else:
        size = (w, h)

    if show_progress:
        print("Background: mode=median (sampling frames)")

    samples, idx = [], 0
    start = time.time()
    target_samples = min(30, max(1, n // max(1, sample_step)))
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % sample_step == 0:
            _, frame = cap.retrieve()
            if frame is None:
                break
            if size != (w, h):
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            samples.append(frame)
            if show_progress:
                _progress("Background", len(samples), target_samples, start)
        idx += 1
        if len(samples) >= 30:
            break
    cap.release()
    if show_progress:
        _progress("Background", target_samples, target_samples, start); _progress_done()

    if not samples:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Ne mogu pročitati ni prvi frame.")
        if size != (w, h):
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return frame

    stack = np.stack(samples, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg


# -------------- accumulation ----------------

def accumulate_heat(video_path, model, stride=2, conf=0.4, resize_w=960, radius=10,
                    mode="balanced", detect_every=5, show_progress=True,
                    mog_history=400, mog_var=32.0, min_area=300):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Ne mogu otvoriti video.")
    w, h, n, fps = read_metadata(cap)

    if resize_w is not None and resize_w < w:
        scale = resize_w / float(w)
        size = (int(w*scale), int(h*scale))
    else:
        size = (w, h)

    heat = np.zeros((size[1], size[0]), dtype=np.float32)
    frame_id = 0

    mog = cv2.createBackgroundSubtractorMOG2(history=int(mog_history), varThreshold=float(mog_var), detectShadows=False)
    last_centers = []

    if mode == "fast":
        detect_every = 10**9
    elif mode == "accurate":
        detect_every = 1
    else:
        detect_every = max(1, int(detect_every))

    start = time.time()
    if show_progress:
        print("Processing video ...")

    processed_frames = 0
    detections_count = 0
    weight_sum = 0.0

    while True:
        ok = cap.grab()
        if not ok:
            break

        if show_progress:
            _progress("Video", frame_id, max(1, n), start)

        if frame_id % stride != 0:
            frame_id += 1
            continue

        _, frame = cap.retrieve()
        if frame is None:
            break
        processed_frames += 1

        if size != (w, h):
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        need_yolo = (frame_id % detect_every == 0)

        if need_yolo:
            res = model.predict(frame, classes=[0], conf=conf, verbose=False)[0]
            last_centers = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                detections_count += len(xyxy)
                for x1, y1, x2, y2 in xyxy:
                    cx = int((x1 + x2) * 0.5)
                    cy = int((y1 + y2) * 0.5)
                    if 0 <= cx < size[0] and 0 <= cy < size[1]:
                        last_centers.append((cx, cy))
                        cv2.circle(heat, (cx, cy), radius, 1.0, -1)
                        weight_sum += 1.0
        else:
            for (cx, cy) in last_centers:
                cv2.circle(heat, (cx, cy), radius, 0.8, -1)
                weight_sum += 0.8

            if mode in ("balanced", "fast"):
                # Stabilniji foreground: stroži threshold + morfologija + veći min_area
                fg = mog.apply(frame, learningRate=0.003)
                _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
                fg = cv2.dilate(fg, k, iterations=1)

                cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) < int(min_area):
                        continue
                    x, y, ww, hh = cv2.boundingRect(c)
                    cx = x + ww // 2
                    cy = y + hh // 2
                    if 0 <= cx < size[0] and 0 <= cy < size[1]:
                        cv2.circle(heat, (cx, cy), radius, 0.5, -1)
                        weight_sum += 0.5

        frame_id += 1

    cap.release()
    if show_progress:
        _progress("Video", max(1, n), max(1, n), start); _progress_done()

    stats = {
        "total_frames": n,
        "fps": float(fps),
        "duration_sec": (float(n) / float(fps)) if fps > 0 else 0.0,
        "processed_frames": processed_frames,
        "detections": detections_count,
        "weight_sum": weight_sum,
        "stride": stride,
        "detect_every": detect_every,
        "mode": mode,
    }

    # Zaglađivanje (na sirovom "heat")
    if SCIPY:
        heat = gaussian_filter(heat, sigma=radius*0.8)
    else:
        k = max(3, int(radius*2) | 1)
        heat = cv2.GaussianBlur(heat, (k, k), radius*0.6)

    return heat, stats


# -------------- overlay & export ----------------

def overlay_heat_on_bg(bg_bgr, heat_norm, alpha=0.65, thr=0.05,
                       colormap=cv2.COLORMAP_TURBO, color_gain=0.7):
    heat_u8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, colormap)
    heat_color = (heat_color.astype(np.float32) * float(color_gain)).clip(0, 255).astype(np.uint8)

    mask = (heat_norm > thr).astype(np.float32)
    mask_3 = np.dstack([mask, mask, mask])

    blend = (bg_bgr*(1.0 - mask_3*alpha) + heat_color*(mask_3*alpha)).astype(np.uint8)
    return blend, heat_color, (heat_u8, mask)

def save_transparent_png(heat_u8, bg_shape, thr=10, out_path="outputs/heatmap_transparent.png"):
    h, w = heat_u8.shape[:2]
    if len(bg_shape) == 3:
        H, W, _ = bg_shape
    else:
        H, W = bg_shape
    if (H, W) != (h, w):
        heat_u8 = cv2.resize(heat_u8, (W, H), interpolation=cv2.INTER_LINEAR)

    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    rgba = np.dstack([color, np.where(heat_u8 > thr, heat_u8, 0).astype(np.uint8)])
    cv2.imwrite(out_path, rgba)

def grab_frame_hq(video_path, t_sec=0.0):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if t_sec and t_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec)*1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Ne mogu uzeti frame na {t_sec:.2f}s iz: {video_path}")
    return frame  # BGR, full rez

def composite_png_on_image(base_bgr, overlay_rgba_path, out_path, alpha_boost=1.0):
    ov = cv2.imread(overlay_rgba_path, cv2.IMREAD_UNCHANGED)
    if ov is None or ov.ndim != 3 or ov.shape[2] != 4:
        raise RuntimeError(f"Ne mogu pročitati RGBA overlay: {overlay_rgba_path}")

    H, W = base_bgr.shape[:2]
    if (ov.shape[0], ov.shape[1]) != (H, W):
        ov = cv2.resize(ov, (W, H), interpolation=cv2.INTER_LINEAR)

    bgr = base_bgr.astype(np.float32)
    ov_bgr = ov[:, :, :3].astype(np.float32)
    alpha = (ov[:, :, 3].astype(np.float32) / 255.0) * float(alpha_boost)
    alpha = np.clip(alpha, 0.0, 1.0)
    a3 = np.dstack([alpha, alpha, alpha])

    out = (bgr * (1.0 - a3) + ov_bgr * a3).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out)


# -------------- public runner ----------------

def run_people_heatmap(
    video: str,
    outdir: str = "outputs",
    model: str = "models/yolov8n.pt",
    stride: int = 2,
    conf: float = 0.4,
    resize_w: int = 960,
    radius: int = 12,
    alpha: float = 0.65,
    mask_thr: float = 0.05,
    mode: str = "balanced",
    bg: str = "median",
    detect_every: int = 5,
    auto_intensity: int = 1,
    gamma: float = 1.0,
    color_gain: float = 0.7,
    ref_image: str = None,
    ref_alpha: float = 0.65,      # zadržano zbog CLI kompatibilnosti
    ref_out: str = "heat_on_reference.png",
    ref_from_video: int = 0,
    ref_time: float = 0.0,
    ref_save: str = "reference_frame_hq.jpg",
    ref_alpha_boost: float = 1.0,
    min_area: int = 300,
    mog_history: int = 400,
    mog_var: float = 32.0,
    norm_p: float = 95.0,
    hm_power: float = 1.0,
    progress: int = 1
) -> Dict[str, Any]:
    """
    Pokreće generisanje heatmape kretanja i sve isto sprema kao CLI.
    Vraća putanje artefakata + statistiku i korištene parametre normalizacije.
    """
    os.makedirs(outdir, exist_ok=True)

    model_path = Path(model)
    if not model_path.exists():
        model_path = Path("yolov8n.pt")
    mdl = YOLO(str(model_path))

    # Pozadina
    bg_img = compute_background(video, sample_step=30, resize_w=resize_w,
                                mode=bg, show_progress=bool(progress))

    # Akumulacija
    heat_raw, stats = accumulate_heat(
        video, mdl,
        stride=stride, conf=conf, resize_w=resize_w, radius=radius,
        mode=mode, detect_every=detect_every, show_progress=bool(progress),
        mog_history=mog_history, mog_var=mog_var, min_area=min_area
    )

    # Normalizacija u [0..1]
    minutes = max(1.0, stats["duration_sec"] / 60.0)
    if auto_intensity:
        heat_rate = heat_raw / minutes
        nz = heat_rate[heat_rate > 0]
        p = float(np.clip(norm_p, 50.0, 99.9))
        p95 = float(np.percentile(nz, p)) if nz.size > 0 else 1e-6
        hm = np.clip(heat_rate / max(p95, 1e-6), 0.0, 1.0)
        gamma_used = float(np.clip(1.0 + 0.18 * np.log10(minutes), 1.0, 1.8))
        hm = hm ** (gamma_used * float(hm_power))
        color_gain_used = float(np.clip(color_gain * (0.95 / (1.0 + 0.35*np.log10(minutes))), 0.45, 0.9))
        thr_used = float(np.clip(mask_thr * (1.0 + 0.25*np.log10(minutes)), 0.03, 0.25))
    else:
        hm = heat_raw - heat_raw.min()
        mx = hm.max()
        if mx > 1e-6:
            hm = hm / mx
        if gamma and gamma != 1.0:
            hm = hm ** float(gamma)
        gamma_used = float(gamma)
        color_gain_used = float(color_gain)
        thr_used = float(mask_thr)

    # Spremanja
    overlay, heat_color, (heat_u8, _) = overlay_heat_on_bg(
        bg_img, hm, alpha=alpha, thr=thr_used,
        colormap=cv2.COLORMAP_TURBO, color_gain=color_gain_used
    )

    overlay_path = os.path.join(outdir, "heatmap_overlay.png")
    colored_path = os.path.join(outdir, "heatmap_colored.png")
    gray_path    = os.path.join(outdir, "heatmap_gray.png")
    npy_path     = os.path.join(outdir, "heatmap.npy")
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(colored_path, heat_color)
    cv2.imwrite(gray_path, heat_u8)
    np.save(npy_path, hm)

    # RGBA export
    rgba_path = os.path.join(outdir, "heatmap_transparent.png")
    save_transparent_png(heat_u8, bg_img.shape, thr=int(thr_used*255), out_path=rgba_path)

    # Finalni HQ overlay
    made_ref = False
    ref_out_path = os.path.join(outdir, ref_out)
    try:
        if ref_from_video:
            ref_bgr = grab_frame_hq(video, t_sec=ref_time)
            cv2.imwrite(os.path.join(outdir, ref_save), ref_bgr)
        else:
            if not ref_image:
                raise RuntimeError("Nije zadano ni ref_image ni ref_from_video=1.")
            ref_bgr = cv2.imread(ref_image, cv2.IMREAD_COLOR)
            if ref_bgr is None:
                raise RuntimeError(f"Ne mogu pročitati referentnu sliku: {ref_image}")

        composite_png_on_image(ref_bgr, rgba_path, ref_out_path, alpha_boost=ref_alpha_boost)
        made_ref = True
    except Exception as e:
        print("Upozorenje: finalni HQ overlay nije uspio:", e)

    # Preview
    preview = overlay.copy()
    cv2.putText(preview, "Heatmap of movement (people)", (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10,10,10), 3, cv2.LINE_AA)
    cv2.putText(preview, "Heatmap of movement (people)", (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,240,240), 1, cv2.LINE_AA)
    preview_path = os.path.join(outdir, "preview.jpg")
    cv2.imwrite(preview_path, preview)

    return {
        "stats": stats,
        "minutes": minutes,
        "gamma_used": float(gamma_used),
        "color_gain_used": float(color_gain_used),
        "thr_used": float(thr_used),
        "artifacts": {
            "overlay": overlay_path,
            "colored": colored_path,
            "gray": gray_path,
            "transparent_png": rgba_path,
            "npy": npy_path,
            "preview": preview_path,
            "ref_out": (ref_out_path if made_ref else None),
        },
        "params": {
            "video": video, "outdir": outdir, "model": str(model_path),
            "stride": stride, "conf": conf, "resize_w": resize_w, "radius": radius,
            "mode": mode, "bg": bg, "detect_every": detect_every,
            "auto_intensity": auto_intensity, "norm_p": norm_p, "hm_power": hm_power
        }
    }


# -------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Put do video fajla (obrada u nižoj rezoluciji radi brzine)")
    ap.add_argument("--model", default="models/yolov8n.pt", help="Put do YOLOv8 modela (npr. models/yolov8n.pt)")
    ap.add_argument("--outdir", default="outputs", help="Direktorij za izlaze")
    ap.add_argument("--stride", type=int, default=2, help="Preskakanje frameova (brže, 2=svaki drugi)")
    ap.add_argument("--conf", type=float, default=0.4, help="Confidence prag za detekciju")
    ap.add_argument("--resize_w", type=int, default=960, help="Resize širina za obradu (radi brzine)")
    ap.add_argument("--radius", type=int, default=12, help="Poluprečnik za tačke kretanja (utiče na širinu tragova)")
    ap.add_argument("--alpha", type=float, default=0.65, help="Jačina blendanja heatmape na pozadinu")
    ap.add_argument("--mask_thr", type=float, default=0.05, help="Prag gdje počinje crtati heatmapu na pozadini")
    ap.add_argument("--mode", choices=["fast","balanced","accurate"], default="balanced",
                    help="fast = samo background subtraction, balanced = YOLO povremeno + MOG2, accurate = YOLO svaki frame")
    ap.add_argument("--bg", choices=["median","first"], default="median",
                    help="Kako se računa pozadina (median = sporije, first = prvi frame, brže)")
    ap.add_argument("--detect_every", type=int, default=5, help="YOLO svaku N-tu sličicu (1=svaku)")

    # Automatsko prilagođavanje intenziteta prema trajanju
    ap.add_argument("--auto_intensity", type=int, default=1, choices=[0,1],
                    help="Automatski prilagodi intenzitet prema trajanju videa (1=DA)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Gamma za ručno podešavanje (koristi se ako isključiš auto_intensity)")
    ap.add_argument("--color_gain", type=float, default=0.7,
                    help="Osnovni gain boje (može biti auto smanjen za duge snimke)")

    # Referentna HQ slika / frame
    ap.add_argument("--ref_image", type=str, default=None,
                    help="Put do referentne HQ slike na koju se lijepi finalna heatmapa")
    ap.add_argument("--ref_alpha", type=float, default=0.65,
                    help="(Samo za stari način) Alpha pri slaganju heat_u8 na sliku")
    ap.add_argument("--ref_out", type=str, default="heat_on_reference.png",
                    help="Naziv izlaza za kompoziciju na referentnu sliku u outdir")
    ap.add_argument("--ref_from_video", type=int, default=0, choices=[0,1],
                    help="Ako je 1, uzmi HQ screenshot direktno iz videa (ignorira --ref_image).")
    ap.add_argument("--ref_time", type=float, default=0.0,
                    help="Vrijeme u sekundama za HQ screenshot iz videa (npr. 0.0 = prvi frame).")
    ap.add_argument("--ref_save", type=str, default="reference_frame_hq.jpg",
                    help="Gdje snimiti HQ screenshot ako se koristi --ref_from_video=1.")
    ap.add_argument("--ref_alpha_boost", type=float, default=1.0,
                    help="Skaliranje prozirnosti heatmap_transparent pri slaganju na HQ sliku (0.5=prozirnije, 2.0=jače)")

    # MOG2 i filtriranje šuma
    ap.add_argument("--min_area", type=int, default=300,
                    help="Minimalna površina konture iz MOG2 da bi se računala (px).")
    ap.add_argument("--mog_history", type=int, default=400,
                    help="History za MOG2 (veće = stabilnije, manje šuma).")
    ap.add_argument("--mog_var", type=float, default=32.0,
                    help="varThreshold za MOG2 (veće = manje osjetljivo).")
    ap.add_argument("--norm_p", type=float, default=95.0,
                    help="Percentil za normalizaciju (95–99).")
    ap.add_argument("--hm_power", type=float, default=1.0,
                    help="Dodatna nelinearnost poslije game (1.0=bez dodatka).")

    ap.add_argument("--progress", type=int, default=1, choices=[0,1], help="Prikaži progres i ETA (1) ili ne (0)")
    args = ap.parse_args()

    res = run_people_heatmap(
        video=args.video, outdir=args.outdir, model=args.model, stride=args.stride,
        conf=args.conf, resize_w=args.resize_w, radius=args.radius, alpha=args.alpha,
        mask_thr=args.mask_thr, mode=args.mode, bg=args.bg, detect_every=args.detect_every,
        auto_intensity=args.auto_intensity, gamma=args.gamma, color_gain=args.color_gain,
        ref_image=args.ref_image, ref_alpha=args.ref_alpha, ref_out=args.ref_out,
        ref_from_video=args.ref_from_video, ref_time=args.ref_time, ref_save=args.ref_save,
        ref_alpha_boost=args.ref_alpha_boost, min_area=args.min_area, mog_history=args.mog_history,
        mog_var=args.mog_var, norm_p=args.norm_p, hm_power=args.hm_power, progress=args.progress
    )

    print("\nSačuvano u:", args.outdir)
    print(" - heatmap_overlay.png (pozadina iz videa + obojena heatmapa, low-res)")
    print(" - heatmap_colored.png (samo heatmapa u boji, low-res)")
    print(" - heatmap_transparent.png (PNG sa alfa kanalom)")
    print(" - heatmap_gray.png i heatmap.npy (sirovi podaci)")
    if res["artifacts"]["ref_out"]:
        print(f" - {Path(res['artifacts']['ref_out']).name} (heatmap nalijepljena na referentnu HQ sliku)")

    print("\nStatistika:")
    st = res["stats"]
    print(f"   Trajanje videa: { _fmt_time(int(st['duration_sec'])) }  ({res['minutes']:.1f} min)")
    print(f"   FPS: {st['fps']:.2f}, total frames: {st['total_frames']}, processed: {st['processed_frames']}")
    print(f"   Mode: {st['mode']}, stride: {st['stride']}, detect_every: {st['detect_every']}")
    print(f"   Detections: {st['detections']}, weight_sum: {st['weight_sum']:.1f}")
    print(f"   Intensity → gamma: {res['gamma_used']:.2f}, color_gain: {res['color_gain_used']:.2f}, mask_thr: {res['thr_used']:.3f}")

if __name__ == "__main__":
    main()
