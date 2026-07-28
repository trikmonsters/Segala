#!/usr/bin/env python3
"""
generate_mask.py
Mengubah boxes.json (hasil detect_text.py) menjadi mask hitam-putih per frame
(putih = area yang akan dihapus/di-inpaint oleh ProPainter).

Opsional: --static-region "x1,y1,x2,y2" untuk menandai area watermark logo
yang selalu ada di posisi tetap (mis. watermark aplikasi editing di pojok),
karena watermark berupa logo/gambar tidak akan terdeteksi oleh OCR teks.
"""
import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Generate frame-wise masks from OCR boxes")
    p.add_argument("--boxes-json", required=True)
    p.add_argument("--frames-dir", required=True, help="Untuk mengambil ukuran frame")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--padding", type=int, default=8, help="Padding piksel di sekitar box")
    p.add_argument(
        "--static-region",
        default=None,
        help="Area watermark tetap 'x1,y1,x2,y2' (relatif ke resolusi frame), opsional, bisa diulang dipisah ';'",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.boxes_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_paths = sorted(
        [os.path.join(args.frames_dir, x) for x in os.listdir(args.frames_dir)
         if x.lower().endswith((".png", ".jpg"))]
    )
    if not frame_paths:
        raise SystemExit(f"Tidak ada frame di {args.frames_dir}")

    sample = cv2.imread(frame_paths[0])
    h, w = sample.shape[:2]

    static_regions = []
    if args.static_region:
        for chunk in args.static_region.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            x1, y1, x2, y2 = [int(v) for v in chunk.split(",")]
            static_regions.append((x1, y1, x2, y2))

    os.makedirs(args.output_dir, exist_ok=True)

    frames_info = data["frames"]
    for entry in frames_info:
        idx = entry["frame"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for (x1, y1, x2, y2) in entry.get("boxes", []):
            x1 = max(0, x1 - args.padding)
            y1 = max(0, y1 - args.padding)
            x2 = min(w, x2 + args.padding)
            y2 = min(h, y2 + args.padding)
            mask[y1:y2, x1:x2] = 255

        for (x1, y1, x2, y2) in static_regions:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 255

        out_path = os.path.join(args.output_dir, f"mask_{idx:06d}.png")
        cv2.imwrite(out_path, mask)

        if idx % 50 == 0:
            print(f"[generate_mask] frame {idx}/{len(frames_info)}")

    print(f"[generate_mask] Selesai. {len(frames_info)} mask ditulis ke {args.output_dir}")


if __name__ == "__main__":
    main()
