# =====================================================================
# BLENDING + RENDER FINAL -- DIJALANKAN DI RUNNER GHA (CPU, bukan Colab)
# =====================================================================
# Ini adalah "Stage 2" yang dulu jalan di Colab, dipindah ke sini karena
# tidak butuh GPU sama sekali (murni OpenCV + ffmpeg). Dipindah supaya
# sesi Colab bisa langsung ditutup begitu ProPainter (Stage 1) selesai,
# tanpa perlu tetap hidup lebih lama lagi -- menghindari sesi mati/
# di-revoke di tengah proses seperti yang beberapa kali terjadi
# sebelumnya saat Stage 2 masih ikut jalan di Colab.
#
# Logika feather-blending PERSIS SAMA dengan versi Colab sebelumnya,
# hanya sumber framenya beda:
#   - Frame asli   -> dibaca langsung dari video_raw.mp4 (GHA sudah
#                     punya file ini, identik dengan yang diupload ke
#                     Colab, tidak perlu transfer ulang).
#   - Frame hasil ProPainter -> dari propainter_inpainted.mp4 (video
#                     kompak yang di-render & didownload dari Colab).
#   - Mask         -> dari propainter_masks.mkv (lossless, dari Colab).
# Ketiganya dibaca frame-by-frame secara paralel (VideoCapture), tanpa
# perlu menyimpan ribuan file PNG perantara ke disk.

import os
import glob
import shutil
import subprocess
import json
import sys

import cv2
import numpy as np

META_PATH = "stage1_output/stage1_meta.json"
ORIG_VIDEO = "input/video_raw.mp4"
INPAINT_VIDEO = "stage1_output/propainter_inpainted.mp4"
MASKS_VIDEO = "stage1_output/propainter_masks.mkv"

FEATHER_RADIUS = 8
SHARPEN_AMOUNT = 0.35

for p in (META_PATH, ORIG_VIDEO, INPAINT_VIDEO, MASKS_VIDEO):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise RuntimeError(f"File yang dibutuhkan tidak ada/kosong: {p}")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

fps = float(meta["fps"])
width = int(meta["width"])
height = int(meta["height"])
expected_frames = int(meta["frame_idx"])

print(f"Metadata Stage 1: fps={fps}, {width}x{height}, expected_frames={expected_frames}")

orig_cap = cv2.VideoCapture(ORIG_VIDEO)
inpaint_cap = cv2.VideoCapture(INPAINT_VIDEO)
mask_cap = cv2.VideoCapture(MASKS_VIDEO)

if not orig_cap.isOpened():
    raise RuntimeError(f"Tidak bisa membuka {ORIG_VIDEO}")
if not inpaint_cap.isOpened():
    raise RuntimeError(f"Tidak bisa membuka {INPAINT_VIDEO}")
if not mask_cap.isOpened():
    raise RuntimeError(f"Tidak bisa membuka {MASKS_VIDEO}")

BLEND_DIR = "propainter_frames_blended"
if os.path.exists(BLEND_DIR):
    shutil.rmtree(BLEND_DIR)
os.makedirs(BLEND_DIR)

print("\n[4/5] Blending hasil inpaint dengan frame asli (feathered, biar tidak kelihatan kotak)...")

i = 0
while True:
    ret_o, orig_img = orig_cap.read()
    ret_i, inpaint_img = inpaint_cap.read()
    ret_m, mask_img_raw = mask_cap.read()

    if not ret_o or not ret_i:
        break

    inpaint_img = cv2.resize(inpaint_img, (width, height), interpolation=cv2.INTER_CUBIC)

    if not ret_m:
        blended = inpaint_img
    else:
        mask_img = cv2.cvtColor(mask_img_raw, cv2.COLOR_BGR2GRAY)
        if mask_img.shape != (height, width):
            mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_bin = (mask_img > 127).astype(np.uint8)
        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
        alpha = np.clip(dist / FEATHER_RADIUS, 0.0, 1.0)[..., None]

        blended = inpaint_img.astype(np.float32) * alpha + orig_img.astype(np.float32) * (1 - alpha)

        if SHARPEN_AMOUNT > 0:
            blur_for_sharpen = cv2.GaussianBlur(blended, (0, 0), 1.0)
            sharpened = cv2.addWeighted(
                blended, 1 + SHARPEN_AMOUNT, blur_for_sharpen, -SHARPEN_AMOUNT, 0
            )
            sharpen_mask = (alpha > 0.05).astype(np.float32)
            blended = sharpened * sharpen_mask + blended * (1 - sharpen_mask)

        blended = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(BLEND_DIR, f"{i:05d}.png"), blended)
    i += 1

    if i % 50 == 0:
        print(f"  Blended: {i}/{expected_frames}", flush=True)

orig_cap.release()
inpaint_cap.release()
mask_cap.release()

if i == 0:
    raise RuntimeError("Tidak ada frame yang berhasil di-blend.")

if i != expected_frames:
    print(f"⚠️  Peringatan: jumlah frame yang diproses ({i}) != jumlah frame Stage 1 "
          f"({expected_frames}). Melanjutkan dengan {i} frame yang berhasil dibaca.")

print(f"Total frame di-blend: {i}")

# --------------------- RENDER VIDEO AKHIR ---------------------
print(f"\n[5/5] Rendering video akhir ({width}x{height})...")

os.makedirs("output", exist_ok=True)
OUTPUT_PATH = "output/video_processed.mp4"

result = subprocess.run(
    [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(BLEND_DIR, "%05d.png"),
        "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
        OUTPUT_PATH,
    ],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("========== FFMPEG STDOUT ==========")
    print(result.stdout[-3000:])
    print("========== FFMPEG STDERR ==========")
    print(result.stderr[-3000:])
    raise RuntimeError("Render ffmpeg gagal.")

shutil.rmtree(BLEND_DIR, ignore_errors=True)

if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0:
    raise RuntimeError(f"{OUTPUT_PATH} tidak dihasilkan/kosong.")

print(f"\n✨ SELESAI (blending+render di GHA)! Video output: {os.path.abspath(OUTPUT_PATH)}")
print(f"Ukuran: {os.path.getsize(OUTPUT_PATH)} bytes")
