# =====================================================================
# PROPAINTER PIPELINE v3 - STAGE 2/2
# SEAMLESS FEATHER BLENDING + RENDER VIDEO AKHIR
# =====================================================================
# Dijalankan lewat panggilan "colab exec" KEDUA, ke sesi Colab yang SAMA
# dengan Stage 1 (kernel stateful -> semua variabel dari Stage 1 seperti
# COMBINED_DIR, FRAMES_DIR, MASKS_DIR, frame_idx, width, height, fps,
# BASENAME, CONFIG, run_checked masih tersedia di sini tanpa perlu
# didefinisikan ulang). Tahap ini jauh lebih singkat & jarang diam lama,
# jadi risiko kena bug reply-timeout colab-cli jauh lebih kecil.
#
# Re-import di bawah ini sengaja tetap ditulis (walau kemungkinan besar
# sudah ada di namespace dari Stage 1) sebagai jaga-jaga kalau Stage 2
# pernah dijalankan sendiri/di-retry di sesi baru.

import os, glob, shutil, time
import cv2
import numpy as np

# --------------------- 5. SEAMLESS FEATHER BLENDING (kunci: tidak kelihatan editan) ---------------------
print("\n[4/5] Blending hasil inpaint dengan frame asli (feathered, biar tidak kelihatan kotak)...")

PROPAINTER_FRAMES = COMBINED_DIR
source_pngs = sorted(glob.glob(os.path.join(PROPAINTER_FRAMES, "*.png")))
if len(source_pngs) == 0:
    raise RuntimeError(f"Folder {PROPAINTER_FRAMES} tidak berisi file .png.")

n_frames = min(len(source_pngs), frame_idx)
if len(source_pngs) != frame_idx:
    print(f"⚠️  Peringatan: jumlah frame hasil ProPainter ({len(source_pngs)}) "
          f"!= jumlah frame input ({frame_idx}). Memproses {n_frames} frame yang cocok.")

BLEND_DIR = os.path.abspath("propainter_frames_blended")
if os.path.exists(BLEND_DIR):
    shutil.rmtree(BLEND_DIR)
os.makedirs(BLEND_DIR)

feather_radius = max(1, CONFIG["feather_radius"])

for i in range(n_frames):
    inpaint_path = source_pngs[i]
    orig_path = os.path.join(FRAMES_DIR, f"{i:05d}.png")
    mask_path = os.path.join(MASKS_DIR, f"{i:05d}.png")

    inpaint_img = cv2.imread(inpaint_path)
    inpaint_img = cv2.resize(inpaint_img, (width, height), interpolation=cv2.INTER_CUBIC)

    if not os.path.exists(orig_path) or not os.path.exists(mask_path):
        cv2.imwrite(os.path.join(BLEND_DIR, f"{i:05d}.png"), inpaint_img)
        continue

    orig_img = cv2.imread(orig_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        cv2.imwrite(os.path.join(BLEND_DIR, f"{i:05d}.png"), inpaint_img)
        continue
    if mask_img.shape != (height, width):
        mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_img > 127).astype(np.uint8)
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    alpha = np.clip(dist / feather_radius, 0.0, 1.0)[..., None]

    blended = inpaint_img.astype(np.float32) * alpha + orig_img.astype(np.float32) * (1 - alpha)

    if CONFIG["sharpen_amount"] > 0:
        blur_for_sharpen = cv2.GaussianBlur(blended, (0, 0), 1.0)
        sharpened = cv2.addWeighted(blended, 1 + CONFIG["sharpen_amount"], blur_for_sharpen, -CONFIG["sharpen_amount"], 0)
        sharpen_mask = (alpha > 0.05).astype(np.float32)
        blended = sharpened * sharpen_mask + blended * (1 - sharpen_mask)

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(BLEND_DIR, f"{i:05d}.png"), blended)

    if (i + 1) % 50 == 0:
        print(f"  Blended: {i + 1}/{n_frames}", flush=True)

# --------------------- 6. RENDER VIDEO AKHIR ---------------------
print(f"\n[5/5] Rendering video akhir ({width}x{height})...")

temp_no_audio = "temp_propainter.mp4"

run_checked([
    'ffmpeg', '-y', '-framerate', str(fps),
    '-i', os.path.join(BLEND_DIR, '%05d.png'),
    '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p',
    temp_no_audio
])

OUTPUT_NAME = f"{BASENAME}_ProPainter_Success.mp4"

if os.path.exists('audio_orig.aac'):
    run_checked([
        'ffmpeg', '-y', '-i', temp_no_audio, '-i', 'audio_orig.aac',
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', OUTPUT_NAME
    ])
    os.remove('audio_orig.aac')
else:
    os.rename(temp_no_audio, OUTPUT_NAME)

if os.path.exists(temp_no_audio):
    os.remove(temp_no_audio)

shutil.rmtree(BLEND_DIR, ignore_errors=True)

print(f"\n✨ SELESAI! Video output: {os.path.abspath(OUTPUT_NAME)}")
print(f"OUTPUT_FILE={os.path.abspath(OUTPUT_NAME)}")
