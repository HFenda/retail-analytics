# ml/heatmap.py
from typing import Dict, Any, Optional
import os
import numpy as np
import cv2

from ultralytics import YOLO
# Importaj funkcije iz tvog people_heatmap.py (isti folder: ml/)
from .people_heatmap import (
    compute_background,
    accumulate_heat,
    overlay_heat_on_bg,
    save_transparent_png,
    grab_frame_hq,
    composite_png_on_image,
)

def run_heatmap(
    video_path: str,
    outdir: str,
    *,
    model_path: str = "yolov8n.pt",
    resize_w: int = 640,
    stride: int = 2,
    conf: float = 0.40,
    radius: int = 12,
    mode: str = "balanced",           # "fast" | "balanced" | "accurate"
    detect_every: int = 5,
    bg: str = "median",               # "median" | "first"
    ref_from_video: int = 1,
    ref_time: float = 0.0,
    ref_out: str = "heat_on_reference.png",
    ref_alpha_boost: float = 0.55,
    norm_p: float = 98.5,
    hm_power: float = 1.2,
    mask_thr: float = 0.22,
    color_gain: float = 0.32,
    min_area: int = 350,
    mog_history: int = 700,
    mog_var: float = 26.0,
    progress: bool = False,
) -> Dict[str, Any]:
    """
    Pozove postojeće people_heatmap funkcije i generiše:
      - heatmap_overlay.png
      - heatmap_colored.png
      - heatmap_gray.png
      - heatmap_transparent.png
      - (opciono) heat_on_reference.png
      - heatmap.npy

    Vraća dict:
      {
        "paths": { "overlay": str, "colored": str, "gray": str, "transparent": str, "ref": Optional[str], "npy": str, "ref_hq": Optional[str] },
        "stats": { ... iz accumulate_heat ... },
        "norm":  { "gamma": float, "color_gain": float, "mask_thr": float }
      }
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Model (YOLO)
    model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")

    # 2) Pozadina (preview)
    bg_bgr = compute_background(
        video_path,
        sample_step=30,
        resize_w=resize_w,
        mode=bg,
        show_progress=bool(progress),
    )

    # 3) Akumulacija heat mape (RAW + statistika)
    heat_raw, stats = accumulate_heat(
        video_path,
        model,
        stride=stride,
        conf=conf,
        resize_w=resize_w,
        radius=radius,
        mode=mode,
        detect_every=detect_every,
        show_progress=bool(progress),
        mog_history=mog_history,
        mog_var=mog_var,
        min_area=min_area,
    )

    # 4) Automatsko mapiranje intenziteta (isti princip kao u CLI skripti)
    minutes = max(1.0, float(stats.get("duration_sec", 0.0)) / 60.0)
    heat_rate = heat_raw / minutes
    nz = heat_rate[heat_rate > 0]
    p = float(np.clip(norm_p, 50.0, 99.9))
    pxx = float(np.percentile(nz, p)) if nz.size > 0 else 1e-6
    hm = np.clip(heat_rate / max(pxx, 1e-6), 0.0, 1.0)
    gamma = float(np.clip(1.0 + 0.18 * np.log10(minutes), 1.0, 1.8))
    hm = hm ** (gamma * float(hm_power))
    color_gain_used = float(np.clip(color_gain * (0.95 / (1.0 + 0.35*np.log10(minutes))), 0.45, 0.9))
    thr_used = float(np.clip(mask_thr * (1.0 + 0.25*np.log10(minutes)), 0.03, 0.25))

    # 5) Overlay i exporti
    overlay, heat_color, (heat_u8, _) = overlay_heat_on_bg(
        bg_bgr, hm, alpha=0.65, thr=thr_used,
        colormap=cv2.COLORMAP_TURBO, color_gain=color_gain_used
    )

    path_overlay = os.path.join(outdir, "heatmap_overlay.png")
    path_colored = os.path.join(outdir, "heatmap_colored.png")
    path_gray    = os.path.join(outdir, "heatmap_gray.png")
    path_trans   = os.path.join(outdir, "heatmap_transparent.png")
    path_npy     = os.path.join(outdir, "heatmap.npy")

    cv2.imwrite(path_overlay, overlay)
    cv2.imwrite(path_colored, heat_color)
    cv2.imwrite(path_gray, heat_u8)
    np.save(path_npy, hm)

    save_transparent_png(heat_u8, bg_bgr.shape, thr=int(thr_used * 255), out_path=path_trans)

    # 6) (Opcionalno) kompozicija na HQ referentni frame/sliku
    path_ref: Optional[str] = None
    path_ref_hq: Optional[str] = None
    try:
        if ref_from_video == 1:
            ref_bgr = grab_frame_hq(video_path, t_sec=ref_time)
            path_ref_hq = os.path.join(outdir, "reference_frame_hq.jpg")
            cv2.imwrite(path_ref_hq, ref_bgr)
        else:
            ref_bgr = None  # korisnik bi morao proslijediti putanju do postojeće slike kroz poseban param ako želi

        if ref_from_video == 1:
            # koristimo tek snimljeni HQ frame
            path_ref = os.path.join(outdir, ref_out)
            composite_png_on_image(ref_bgr, path_trans, path_ref, alpha_boost=ref_alpha_boost)
    except Exception:
        # ako kompozicija ne uspije, samo preskoči (ostali artefakti su spremni)
        path_ref = None
        path_ref_hq = None

    return {
        "paths": {
            "overlay": path_overlay,
            "colored": path_colored,
            "gray": path_gray,
            "transparent": path_trans,
            "ref": path_ref,
            "ref_hq": path_ref_hq,
            "npy": path_npy,
        },
        "stats": {
            "total_frames": int(stats.get("total_frames", 0)),
            "fps": float(stats.get("fps", 0.0)),
            "duration_sec": float(stats.get("duration_sec", 0.0)),
            "processed_frames": int(stats.get("processed_frames", 0)),
            "detections": int(stats.get("detections", 0)),
            "weight_sum": float(stats.get("weight_sum", 0.0)),
            "stride": int(stats.get("stride", stride)),
            "detect_every": int(stats.get("detect_every", detect_every)),
            "mode": str(stats.get("mode", mode)),
        },
        "norm": {
            "gamma": float(gamma),
            "color_gain": float(color_gain_used),
            "mask_thr": float(thr_used),
            "percentile": float(p),
        },
    }
