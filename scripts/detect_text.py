#!/usr/bin/env python3
"""
detect_text.py
Mendeteksi semua region teks (subtitle, watermark berbasis teks, caption, dsb)
di setiap frame menggunakan EasyOCR.

Untuk efisiensi di GitHub Actions (CPU-only), OCR tidak wajib dijalankan di
SETIAP frame. Gunakan --stride N untuk hanya menjalankan OCR setiap N frame,
lalu box hasilnya akan "ditahan" (hold) untuk frame-frame di antaranya.
refine_mask.py nanti akan mengoreksi hasil hold ini dengan temporal smoothing.

Output: satu file JSON berisi list box per frame:
[
  {"frame": 0, "boxes": [[x1,y1,x2,y2], ...]},
  ...
]
"""
import argparse
import json
import os
import glob
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Detect text/subtitle/watermark regions per frame")
    p.add_argument("--frames-dir", required=True, help="Folder berisi frame_%06d.png")
    p.add_argument("--output", required=True, help="Path file JSON output")
    p.add_argument("--stride", type=int, default=3, help="Jalankan OCR setiap N frame (default 3)")
    p.add_argument("--gpu", action="store_true", help="Gunakan GPU jika tersedia")
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--languages", default="en,id", help="Daftar bahasa EasyOCR, pisahkan koma")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import easyocr
    except ImportError:
        print("EasyOCR belum terpasang. Jalankan: pip install easyocr", file=sys.stderr)
        sys.exit(1)

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

    for idx, fp in enumerate(frame_paths):
        run_ocr = (idx % args.stride == 0)

        if run_ocr:
            img = cv2.imread(fp)
            if img is None:
                results.append({"frame": idx, "boxes": last_boxes})
                continue

            detections = reader.readtext(img)
            boxes = []
            for (bbox, text, conf) in detections:
                if conf < args.min_confidence:
                    continue
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                boxes.append([x1, y1, x2, y2])

            last_boxes = boxes
            results.append({"frame": idx, "boxes": boxes})
        else:
            # Tahan (hold) box dari frame OCR terakhir; refine_mask akan
            # menghaluskan transisinya secara temporal.
            results.append({"frame": idx, "boxes": last_boxes})

        if idx % 50 == 0:
            print(f"[detect_text] frame {idx}/{len(frame_paths)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"num_frames": len(frame_paths), "frames": results}, f)

    print(f"[detect_text] Selesai. {len(frame_paths)} frame diproses -> {args.output}")


if __name__ == "__main__":
    main()
