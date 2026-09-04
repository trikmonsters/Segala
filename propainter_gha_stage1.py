# =====================================================================
# PROPAINTER PIPELINE v3 - STAGE 1/2
# EXTRACTION + MASK + PROPAINTER INFERENCE (bagian terberat/terlama)
# =====================================================================
# Dipecah dari script tunggal menjadi 2 tahap yang dijalankan lewat 2
# panggilan "colab exec" TERPISAH ke sesi Colab yang SAMA (kernel-nya
# stateful, jadi semua variabel & file di sini tetap tersedia untuk
# Stage 2). Tujuannya: memperpendek durasi SETIAP panggilan RPC tunggal,
# supaya lebih kecil peluangnya kena bug upstream google-colab-cli#14
# (koneksi jadi tidak responsif untuk eksekusi yang berjalan lama).
#
# Tambahan fix di stage ini: proses ProPainter per-chunk sebelumnya
# betul-betul "diam" (tidak ada output sama sekali) selama subprocess
# GPU bekerja, karena outputnya di-buffer penuh (capture_output=True)
# bukan di-stream. Diam lama seperti itu adalah pemicu paling mungkin
# dari bug #14 di atas. Sekarang subprocess ProPainter dijalankan lewat
# run_with_heartbeat() yang tetap mencetak "... masih berjalan (Ns)"
# secara berkala walau belum ada output baru dari ProPainter sendiri.
#
# Ringkasan perilaku deteksi/masking tetap SAMA seperti sebelumnya:
#   - Mask HANYA dari kotak hasil OCR (EasyOCR) di 40% bawah frame,
#     TIDAK ada SAM/segmentasi objek -- supaya orang/objek lain aman.
#   - ProPainter dipecah per-chunk (chunk_frames) + auto-retry OOM.

import subprocess, os, sys, time, gc, glob, shutil
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --------------------- CONFIG (silakan diubah sesuai kebutuhan) ---------------------
CONFIG = {
    # -- OCR speed --
    "ocr_sample_interval": 1,
    "ocr_downscale": 0.65,
    "ocr_conf_min": 0.035,
    "ocr_text_threshold": 0.5,
    "ocr_low_text": 0.3,
    "ocr_grace_frames": 2,

    # -- Mask quality --
    "mask_dilate_iterations": 1,
    "mask_close_kernel": 5,
    "box_padding_pct": 0.08,

    # -- Feather blending (dipakai di Stage 2, disimpan di CONFIG supaya
    # ikut tersedia lewat kernel yang sama) --
    "feather_radius": 8,

    # -- ProPainter resolusi --
    "propainter_max_dim": 1024,
    "subvideo_length": 12,
    "neighbor_length": 5,
    "ref_stride": 8,
    "oom_retry_max": 3,

    # -- Sharpening (dipakai di Stage 2) --
    "sharpen_amount": 0.35,

    # -- RAM safety --
    "chunk_frames": 120,
}

r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
assert r.returncode == 0, "GPU tidak terdeteksi! Set Runtime > Change runtime type > T4 GPU."
print("GPU OK:", subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'],
                                 capture_output=True, text=True).stdout.strip())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def run_checked(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print("\n❌ COMMAND FAILED:", " ".join(cmd))
        print("---- STDOUT ----")
        print(result.stdout[-3000:])
        print("---- STDERR ----")
        print(result.stderr[-3000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


class _HeartbeatResult:
    """Bentuk hasil yang menyerupai subprocess.CompletedProcess, supaya
    kode pemanggil di bawah (cek OOM, cek weight issue, dst) tidak perlu
    diubah -- stdout & stderr sengaja diisi konten yang SAMA (gabungan
    stdout+stderr asli) karena kita menggabungkan kedua stream saat
    streaming, bukan menangkapnya terpisah."""
    def __init__(self, returncode, combined_output):
        self.returncode = returncode
        self.stdout = combined_output
        self.stderr = combined_output


def run_with_heartbeat(cmd, env=None, heartbeat_sec=15, label=""):
    """Jalankan subprocess sambil tetap mencetak sesuatu ke kernel secara
    berkala, supaya koneksi colab-cli tidak dianggap 'diam' terlalu lama
    (fix bug upstream google-colab-cli#14) selama ProPainter bekerja di
    GPU tanpa ada output baru untuk waktu yang lama."""
    log_path = f"_heartbeat_{int(time.time() * 1000)}.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 text=True, env=env)
        start = time.time()
        while proc.poll() is None:
            time.sleep(heartbeat_sec)
            if proc.poll() is not None:
                break
            elapsed = int(time.time() - start)
            print(f"    ... {label} masih berjalan ({elapsed}s)", flush=True)
        returncode = proc.returncode

    with open(log_path, "r", errors="replace") as logf:
        combined_output = logf.read()
    os.remove(log_path)

    return _HeartbeatResult(returncode, combined_output)


# --------------------- 1. DETEKSI FILE INPUT ---------------------
# GHA meng-upload video ke /content/video_raw.mp4 sebelum script ini dijalankan.
# Bisa dioverride dengan environment variable INPUT_VIDEO.
INPUT_PATH = os.environ.get("INPUT_VIDEO", "/content/video_raw.mp4")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Video input tidak ditemukan: {INPUT_PATH}. "
        "Pastikan GitHub Actions sudah menjalankan 'colab upload'."
    )

BASENAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]

# --------------------- 2. INSTALL DEPS & PROPAINTER ---------------------
print("\n[1/5] Menginstal ProPainter & Dependensi...")
get_ipython().system('apt-get -qq update && apt-get -qq install -y ffmpeg libgl1 libglib2.0-0')

if not os.path.exists('ProPainter'):
    get_ipython().system('git clone https://github.com/sczhou/ProPainter.git')

get_ipython().run_line_magic('cd', 'ProPainter')
get_ipython().system('pip -q install -r requirements.txt')
get_ipython().system('pip -q install easyocr')

os.makedirs('weights', exist_ok=True)

get_ipython().run_line_magic('cd', '..')

WEIGHT_URLS = {
    "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
}


def ensure_weight(name, url, min_bytes=1024):
    wp = os.path.join("ProPainter", "weights", name)
    if os.path.exists(wp) and os.path.getsize(wp) >= min_bytes:
        return True
    print(f"  Mengunduh manual: {name} ...")
    result = subprocess.run(['wget', '-q', '-O', wp, url], capture_output=True, text=True)
    ok = result.returncode == 0 and os.path.exists(wp) and os.path.getsize(wp) >= min_bytes
    if not ok and os.path.exists(wp):
        os.remove(wp)
    return ok

# --------------------- 3. EXTRACTION & DETEKSI MASK (subtitle only, aman) ---------------------
import cv2
import numpy as np
import easyocr

print("\n[2/5] Ekstraksi Frame & Generasi Mask (khusus teks subtitle)...")
FRAMES_DIR = "propainter_frames"
MASKS_DIR = "propainter_masks"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

subprocess.run(['ffmpeg', '-y', '-i', INPUT_PATH, '-vn', '-acodec', 'copy', 'audio_orig.aac'],
                stderr=subprocess.DEVNULL)

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Tidak bisa membuka video input: {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

reader = easyocr.Reader(['en', 'id'], gpu=True)

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (CONFIG["mask_close_kernel"], CONFIG["mask_close_kernel"])
)

frame_idx = 0
last_boxes = []
miss_streak = 0
ocr_scale = CONFIG["ocr_downscale"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_name = f"{frame_idx:05d}.png"
    cv2.imwrite(os.path.join(FRAMES_DIR, frame_name), frame)

    small = cv2.resize(frame, None, fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_AREA)
    results_ocr = reader.readtext(
        small,
        text_threshold=CONFIG["ocr_text_threshold"],
        low_text=CONFIG["ocr_low_text"],
    )
    boxes = []
    for (bbox, text, prob) in results_ocr:
        if prob > CONFIG["ocr_conf_min"]:
            pts = np.array(bbox, dtype=np.float32) / ocr_scale
            x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
            x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())
            if y_max > int(height * 0.40):
                h_pad = max(3, int((y_max - y_min) * CONFIG["box_padding_pct"]))
                w_pad = max(3, int((x_max - x_min) * CONFIG["box_padding_pct"]))
                boxes.append([max(0, x_min - w_pad), max(0, y_min - h_pad),
                              min(width, x_max + w_pad), min(height, y_max + h_pad)])

    if boxes:
        last_boxes = boxes
        miss_streak = 0
    elif last_boxes and miss_streak < CONFIG["ocr_grace_frames"]:
        boxes = last_boxes
        miss_streak += 1
    else:
        last_boxes = []
        miss_streak = 0

    mask = np.zeros((height, width), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    if boxes:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.dilate(mask, kernel_dilate, iterations=CONFIG["mask_dilate_iterations"])

    cv2.imwrite(os.path.join(MASKS_DIR, frame_name), mask)
    frame_idx += 1

    del frame, mask
    if frame_idx % 30 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Frame tersimpan: {frame_idx}")

cap.release()

if frame_idx == 0:
    raise RuntimeError("Tidak ada frame yang berhasil diekstrak dari video input!")

print("Membersihkan VRAM secara menyeluruh dari EasyOCR...")
del reader
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --------------------- 4. INPAINTING VIA PROPAINTER (chunked, RAM-safe, OOM auto-retry) ---------------------
print("\n[3/5] Menjalankan ProPainter per-chunk (agar RAM tidak habis)...")
start_time = time.time()

ABS_FRAMES = os.path.abspath("propainter_frames")
ABS_MASKS = os.path.abspath("propainter_masks")
COMBINED_DIR = os.path.abspath("propainter_frames_combined")
if os.path.exists(COMBINED_DIR):
    shutil.rmtree(COMBINED_DIR)
os.makedirs(COMBINED_DIR)

_longest_side = max(width, height)
_scale = min(1.0, CONFIG["propainter_max_dim"] / _longest_side)
pp_width = max(8, int(round(width * _scale / 8)) * 8)
pp_height = max(8, int(round(height * _scale / 8)) * 8)
print(f"  Resolusi asli: {width}x{height} -> resolusi kerja ProPainter: {pp_width}x{pp_height}")

get_ipython().run_line_magic('cd', 'ProPainter')


def build_cmd(frames_dir, masks_dir, output_dir, subvideo_length, neighbor_length):
    return [
        "python", "inference_propainter.py",
        "--video", frames_dir,
        "--mask", masks_dir,
        "--output", output_dir,
        "--width", str(pp_width),
        "--height", str(pp_height),
        "--subvideo_length", str(subvideo_length),
        "--neighbor_length", str(neighbor_length),
        "--ref_stride", str(CONFIG["ref_stride"]),
        "--fp16",
        "--save_frames",
    ]


def find_best_frames_dir(root, exclude_dirs):
    exclude_abs = {os.path.abspath(d) for d in exclude_dirs}
    best_dir, best_count = None, 0
    for cur_root, _dirs, cur_files in os.walk(root):
        if os.path.abspath(cur_root) in exclude_abs:
            continue
        pngs = [f for f in cur_files if f.lower().endswith('.png')]
        if len(pngs) > best_count:
            best_count = len(pngs)
            best_dir = cur_root
    return best_dir, best_count


env_vars = os.environ.copy()
env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sub_len = CONFIG["subvideo_length"]
nbr_len = CONFIG["neighbor_length"]
weight_fallback_tried = False

chunk_size = max(1, CONFIG["chunk_frames"])
chunk_ranges = [(s, min(s + chunk_size, frame_idx)) for s in range(0, frame_idx, chunk_size)]
print(f"  Video dipecah menjadi {len(chunk_ranges)} chunk (~{chunk_size} frame/chunk).")

global_out_idx = 0

for chunk_i, (start_f, end_f) in enumerate(chunk_ranges):
    n_chunk = end_f - start_f
    print(f"\n  --- Chunk {chunk_i + 1}/{len(chunk_ranges)}: frame {start_f}-{end_f - 1} ({n_chunk} frame) ---")

    CHUNK_FRAMES_DIR = os.path.abspath(f"chunk_{chunk_i}_frames")
    CHUNK_MASKS_DIR = os.path.abspath(f"chunk_{chunk_i}_masks")
    CHUNK_OUT_DIR = os.path.abspath(f"chunk_{chunk_i}_out")
    for d in (CHUNK_FRAMES_DIR, CHUNK_MASKS_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    for local_i, global_i in enumerate(range(start_f, end_f)):
        name = f"{global_i:05d}.png"
        os.symlink(os.path.join(ABS_FRAMES, name), os.path.join(CHUNK_FRAMES_DIR, f"{local_i:05d}.png"))
        os.symlink(os.path.join(ABS_MASKS, name), os.path.join(CHUNK_MASKS_DIR, f"{local_i:05d}.png"))

    result = None
    for attempt in range(CONFIG["oom_retry_max"] + 1):
        cmd = build_cmd(CHUNK_FRAMES_DIR, CHUNK_MASKS_DIR, CHUNK_OUT_DIR, sub_len, nbr_len)
        print(f"    Percobaan {attempt + 1}: subvideo_length={sub_len}, neighbor_length={nbr_len}")

        # FIX (bug google-colab-cli#14): dulu pakai subprocess.run(capture_output=True)
        # yang membuat kernel TOTAL DIAM selama GPU bekerja (bisa beberapa menit per
        # chunk tanpa output apapun) -- ini pemicu paling mungkin dari koneksi
        # colab-cli jadi tidak responsif. Sekarang pakai run_with_heartbeat supaya
        # kernel tetap mencetak sesuatu secara berkala walau ProPainter sendiri diam.
        result = run_with_heartbeat(
            cmd, env=env_vars, heartbeat_sec=15,
            label=f"chunk {chunk_i + 1}/{len(chunk_ranges)}"
        )

        if result.returncode == 0:
            break

        stderr_low = result.stderr.lower()

        if "out of memory" in stderr_low or "cuda error" in stderr_low:
            if attempt == CONFIG["oom_retry_max"]:
                print(result.stderr[-3000:])
                break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            sub_len = max(4, sub_len // 2)
            nbr_len = max(2, nbr_len // 2)
            print("    ⚠️  CUDA OOM terdeteksi, menurunkan parameter dan mencoba lagi...")
            time.sleep(2)
            continue

        looks_like_weight_issue = any(
            kw in stderr_low for kw in ["urlopen", "urlerror", "connection", "download", ".pth", "no such file"]
        )
        if looks_like_weight_issue and not weight_fallback_tried:
            weight_fallback_tried = True
            print("    ⚠️  Kemungkinan gagal unduh bobot otomatis, mencoba fallback manual...")
            all_ok = True
            for wname, wurl in WEIGHT_URLS.items():
                if not ensure_weight(wname, wurl):
                    all_ok = False
                    print(f"    ❌ Gagal juga mengunduh manual: {wname}")
            if all_ok:
                print("    ✅ Fallback manual berhasil, mencoba inference lagi...")
                continue

        break

    if result is None or result.returncode != 0:
        print("\n❌ PROPAINTER ERROR LOGS (chunk gagal):")
        print(result.stdout[-3000:] if result else "")
        raise RuntimeError(f"ProPainter gagal pada chunk {chunk_i + 1}. Lihat log di atas!")

    chunk_frames_out, found_count = find_best_frames_dir(CHUNK_OUT_DIR, exclude_dirs=[CHUNK_FRAMES_DIR, CHUNK_MASKS_DIR])
    if not chunk_frames_out or found_count == 0:
        raise RuntimeError(f"Tidak ditemukan frame hasil ProPainter untuk chunk {chunk_i + 1}.")

    chunk_pngs = sorted(glob.glob(os.path.join(chunk_frames_out, "*.png")))
    for p in chunk_pngs:
        shutil.move(p, os.path.join(COMBINED_DIR, f"{global_out_idx:05d}.png"))
        global_out_idx += 1

    shutil.rmtree(CHUNK_FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(CHUNK_MASKS_DIR, ignore_errors=True)
    shutil.rmtree(CHUNK_OUT_DIR, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

get_ipython().run_line_magic('cd', '..')

print(f"\nProPainter selesai untuk semua chunk dalam: {time.time() - start_time:.1f} detik "
      f"(setting akhir: subvideo_length={sub_len}, neighbor_length={nbr_len})")

print("\n=== STAGE1_DONE ===")
print(f"STAGE1_FRAME_COUNT={frame_idx}")
print(f"STAGE1_COMBINED_COUNT={len(glob.glob(os.path.join(COMBINED_DIR, '*.png')))}")

# Penanda di FILESYSTEM (bukan cuma di output kernel) -- supaya GHA tetap
# bisa memverifikasi Stage 1 selesai meski balasan RPC "colab exec" hilang
# akibat bug google-colab-cli#14. File di disk tidak terpengaruh oleh
# putusnya koneksi WebSocket, beda dengan pesan/print yang tidak sampai.
with open("/content/STAGE1_DONE.marker", "w") as _mf:
    _mf.write(str(frame_idx))
