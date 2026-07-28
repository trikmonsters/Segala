#!/usr/bin/env python3
"""
refine_mask.py
Menghaluskan urutan mask:
1. Morphological close + dilate per frame -> menutup lubang kecil di dalam box teks
   dan memberi margin aman untuk inpainting.
2. Temporal union dalam window kecil (mis. +/-2 frame) -> mask tidak "berkedip"
   (flicker) saat OCR gagal mendeteksi teks di satu-dua frame tertentu.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Refine mask sequence (morphology + temporal smoothing)")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dilate-kernel", type=int, default=9)
    p.add_argument("--temporal-window", type=int, default=2, help="Radius window (frame) untuk union temporal")
    return p.parse_args()


def main():
    args = parse_args()

    mask_paths = sorted(glob.glob(os.path.join(args.input_dir, "mask_*.png")))
    if not mask_paths:
        raise SystemExit(f"Tidak ada mask di {args.input_dir}")

    masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]
    n = len(masks)

    kernel = np.ones((args.dilate_kernel, args.dilate_kernel), np.uint8)
    closed = []
    for m in masks:
        c = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        c = cv2.dilate(c, kernel, iterations=1)
        closed.append(c)

    os.makedirs(args.output_dir, exist_ok=True)
    w = args.temporal_window

    for i in range(n):
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        union = closed[lo]
        for j in range(lo + 1, hi):
            union = cv2.bitwise_or(union, closed[j])

        out_path = os.path.join(args.output_dir, f"mask_{i:06d}.png")
        cv2.imwrite(out_path, union)

        if i % 50 == 0:
            print(f"[refine_mask] frame {i}/{n}")

    print(f"[refine_mask] Selesai. {n} mask disempurnakan -> {args.output_dir}")


if __name__ == "__main__":
    main()
