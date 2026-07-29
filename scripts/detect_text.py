#!/usr/bin/env python3
"""
detect_text.py

"""
import argparse
import json
import os
import glob
import sys
import time

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Detect text/subtitle/watermark regions per frame")
    p.add_argument("--frames-dir", required=True, help="Folder berisi frame_%06d.png")
    p.add_argument("--output", required=True, help="Path file JSON output")
    p.add_argument("--stride", type=int, default=5, help="Jalankan OCR setiap N frame (default 5)")
    p.add_argument("--gpu", action="store_true", help="Gunakan GPU jika tersedia")
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--languages", default="en,id", help="Daftar bahasa EasyOCR, pisahkan koma")
    p.add_argument(
        "--roi",
        default="0.0,1.0",
        help='Batasi area scan vertikal "top_frac,bottom_frac" relatif tinggi frame, '
             'mis. "0.7,1.0" untuk 30%% bagian bawah saja (area subtitle umum). Default scan penuh.',
    )
    p.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Downscale lebar frame/ROI ke N piksel sebelum OCR (0 = tanpa resize). "
             "Disarankan 800-1000 untuk video 1080p+.",
    )
    p.add_argument("--canvas-size", type=int, default=1280, help="Parameter canvas_size EasyOCR (default lebih kecil dari bawaan 2560 untuk kecepatan)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import easyocr
    except ImportError:
        print("EasyOCR belum terpasang. Jalankan: pip install easyocr", file=sys.stderr)
        sys.exit(1)

    top_frac, bottom_frac = [float(x) for x in args.roi.split(",")]

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    reader = easyocr.Reader(langs, gpu=args.gpu)

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not frame_paths:
        print(f"Tidak ada frame ditemukan di {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    last_boxes = []
    t_start = time.time()

    for idx, fp in enumerate(frame_paths):
        run_ocr = (idx % args.stride == 0)

        if run_ocr:
            img = cv2.imread(fp)
            if img is None:
                results.append({"frame": idx, "boxes": last_boxes})
                continue

            h, w = img.shape[:2]
            y1_roi = int(h * top_frac)
            y2_roi = int(h * bottom_frac)
            crop = img[y1_roi:y2_roi, :]

            scale = 1.0
            ocr_input = crop
            if args.resize_width > 0 and crop.shape[1] > args.resize_width:
                scale = args.resize_width / crop.shape[1]
                new_w = args.resize_width
                new_h = max(1, int(crop.shape[0] * scale))
                ocr_input = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            detections = reader.readtext(ocr_input, canvas_size=args.canvas_size)
            boxes = []
            for (bbox, text, conf) in detections:
                if conf < args.min_confidence:
                    continue
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                # Balikkan skala resize, lalu geser balik sesuai offset ROI
                x1 = int(min(xs) / scale)
                y1 = int(min(ys) / scale) + y1_roi
                x2 = int(max(xs) / scale)
                y2 = int(max(ys) / scale) + y1_roi
                boxes.append([x1, y1, x2, y2])

            last_boxes = boxes
            results.append({"frame": idx, "boxes": boxes})
        else:
            # Tahan (hold) box dari frame OCR terakhir; refine_mask akan
            # menghaluskan transisinya secara temporal.
            results.append({"frame": idx, "boxes": last_boxes})

        if idx % 50 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(frame_paths) - idx - 1) / rate if rate > 0 else float("inf")
            print(f"[detect_text] frame {idx}/{len(frame_paths)} | {rate:.1f} frame/s | ETA {eta/60:.1f} menit")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"num_frames": len(frame_paths), "frames": results}, f)

    total_time = time.time() - t_start
    print(f"[detect_text] Selesai. {len(frame_paths)} frame diproses dalam {total_time/60:.1f} menit -> {args.output}")


if __name__ == "__main__":
    main()
