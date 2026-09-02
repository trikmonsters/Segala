# =====================================================================
# PROPAINTER PIPELINE v3 - SUBTITLE-ONLY REMOVAL (SAFE, NO OVER-MASKING)
# =====================================================================
# Ringkasan perilaku versi ini:
#
#   [DETEKSI: HANYA TEKS SUBTITLE]
#   - Mask dibangun MURNI dari kotak hasil deteksi OCR (EasyOCR) di 40%
#     bagian bawah frame -- zona khas subtitle. Tidak ada SAM, tidak ada
#     segmentasi objek, tidak ada scan watermark/logo di area lain.
#   - Kenapa SAM dihapus: SAM pernah "menebak" objek besar di sekitar
#     kotak teks (termasuk orang) dan menghasilkan mask yang jauh lebih
#     besar dari yang dimaksud -> video jadi rusak/orang ikut terhapus.
#     Kotak OCR + padding tipis + morphological smoothing (close+dilate
#     kecil) sudah cukup untuk menutup teks tanpa risiko itu.
#   - Logo/watermark brand SENGAJA TIDAK disentuh (sesuai permintaan) --
#     cukup teks subtitle yang hilang, gambar/logo dibiarkan apa adanya.
#
#   [REALISTIC / TIDAK KELIHATAN EDITAN]
#   - Tahap "Seamless Feather Blending": mask di-feather pakai distance
#     transform (bukan GaussianBlur biasa) -- bagian DALAM mask (teks
#     yang benar-benar terdeteksi) selalu alpha=1.0 penuh, transisi halus
#     hanya di cincin tepi luar mask. Jadi tepi kotak tidak kelihatan
#     TANPA membuat teks aslinya numpang balik/transparan.
#
#   [KECEPATAN]
#   - OCR dijalankan tiap frame tapi di resolusi diperkecil (ocr_downscale)
#     lalu koordinat di-scale balik -- OCR tetap akurat per-frame (tidak
#     reuse box basi) tapi lebih cepat dari OCR di resolusi penuh.
#
#   [ANTI OOM VRAM & RAM]
#   - ProPainter dipecah jadi beberapa CHUNK (chunk_frames) supaya RAM
#     sistem tidak habis (ProPainter memuat seluruh video ke RAM sekaligus
#     kalau diproses utuh).
#   - Tiap chunk dibungkus auto-retry: kalau "CUDA out of memory" muncul,
#     subvideo_length & neighbor_length otomatis diturunkan lalu dicoba
#     ulang (maks oom_retry_max kali).
#   - Bobot ProPainter di-download otomatis oleh script bawaan; ada
#     fallback wget manual kalau itu gagal karena masalah jaringan.

import subprocess, os, sys, time, gc, glob, shutil
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --------------------- CONFIG (silakan diubah sesuai kebutuhan) ---------------------
CONFIG = {
    # -- OCR speed --
    "ocr_sample_interval": 1,
    "ocr_downscale": 0.65,        # dinaikkan dari 0.5 -> teks kecil lebih kebaca, tetap lebih cepat dari full-res
    "ocr_conf_min": 0.035,        # diturunkan dari 0.05 -> tangkap teks tipis/kecil yang confidence-nya rendah
    "ocr_text_threshold": 0.5,    # ambang internal EasyOCR (default 0.7) - lebih rendah = lebih sensitif
    "ocr_low_text": 0.3,          # ambang internal EasyOCR (default 0.4) - lebih rendah = lebih sensitif
    "ocr_grace_frames": 2,        # kalau OCR gagal total di 1-2 frame berturut-turut, pakai box terakhir
                                   # (aman sekarang karena mask cuma kotak, bukan SAM yg bisa "nyasar")

    # -- Mask quality --
    # PENTING (fix "over-removal"/orang ke-mask): mask HANYA berupa kotak
    # dari hasil deteksi OCR (tidak pakai SAM). SAM dihapus karena bisa
    # "menebak" objek lain di sekitar kotak (mis. orang) dan menghasilkan
    # mask raksasa yang salah. Kotak + padding kecil + smoothing sudah
    # cukup untuk menutup teks subtitle tanpa berlebihan.
    "mask_dilate_iterations": 1,  # kecil saja, cukup utk anti-alias tepi teks
    "mask_close_kernel": 5,       # morphological closing, haluskan kontur mask (harus ganjil)
    "box_padding_pct": 0.08,      # padding tipis di sekitar box teks (bukan logo, bukan objek lain)

    # -- Feather blending (kunci "tidak kelihatan editan", TANPA subtitle numpang balik) --
    # feather_radius = lebar cincin transisi di TEPI LUAR mask (piksel). Bagian
    # DALAM mask selalu alpha=1.0 penuh (dijamin oleh distance transform),
    # jadi subtitle asli TIDAK PERNAH tembus balik ke tengah area yang di-inpaint.
    "feather_radius": 8,

    # -- ProPainter resolusi: dihitung OTOMATIS dari resolusi video asli (lihat
    # bagian 4), dibatasi oleh propainter_max_dim supaya tidak OOM. Kalau video
    # aslinya <= max_dim, diproses di resolusi ASLI (tidak ada blur upscale
    # sama sekali). auto-retry tetap menurunkan subvideo/neighbor length kalau OOM.
    "propainter_max_dim": 1024,   # sisi terpanjang maksimum saat diproses ProPainter
    "subvideo_length": 12,
    "neighbor_length": 5,
    "ref_stride": 8,
    "oom_retry_max": 3,

    # -- Sharpening ringan HANYA di area yang di-inpaint (menutupi sedikit
    # blur akibat ProPainter bekerja di resolusi lebih rendah dari original) --
    "sharpen_amount": 0.35,

    # -- RAM safety (FIX: ProPainter memuat SELURUH video ke RAM sekaligus,
    # bukan streaming, meski --subvideo_length dipakai. Ini penyebab crash
    # "used all available RAM" di Colab. Solusinya: pecah video jadi
    # beberapa chunk, proses ProPainter terpisah per chunk, gabung hasilnya.) --
    "chunk_frames": 120,  # jumlah frame per chunk. Turunkan (mis. 60) kalau masih RAM habis.
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


# --------------------- 1. DETEKSI FILE INPUT ---------------------
VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v')

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

# CATATAN PENTING (fix bug sebelumnya):
# ID gdown di versi lama TIDAK VALID (bukan file ID asli), makanya selalu
# gagal. Ternyata inference_propainter.py SUDAH otomatis mengunduh bobot
# (ProPainter.pth, raft-things.pth, recurrent_flow_completion.pth) sendiri
# dari https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ saat
# pertama kali dijalankan, kalau file belum ada di folder weights/. Jadi
# kita tidak perlu download manual sama sekali -- biarkan script bawaan
# ProPainter yang menanganinya di tahap [3/5].
os.makedirs('weights', exist_ok=True)

get_ipython().run_line_magic('cd', '..')

WEIGHT_URLS = {
    "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
}


def ensure_weight(name, url, min_bytes=1024):
    """Fallback manual download kalau auto-download bawaan ProPainter gagal
    (mis. karena jaringan). Dipanggil setelah percobaan pertama inference,
    bukan sebelumnya -- supaya tidak menduplikasi proses auto-download."""
    wp = os.path.join("ProPainter", "weights", name)
    if os.path.exists(wp) and os.path.getsize(wp) >= min_bytes:
        return True
    print(f"  Mengunduh manual: {name} ...")
    result = subprocess.run(['wget', '-q', '-O', wp, url], capture_output=True, text=True)
    ok = result.returncode == 0 and os.path.exists(wp) and os.path.getsize(wp) >= min_bytes
    if not ok and os.path.exists(wp):
        os.remove(wp)  # hapus file kosong/gagal
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

    # --- OCR tiap frame (di resolusi diperkecil, tapi tidak terlalu kecil
    # supaya teks subtitle yang kecil/tipis tetap kebaca). Threshold internal
    # EasyOCR juga dilonggarkan (ocr_text_threshold/ocr_low_text) supaya
    # tidak gampang miss pada teks kontras rendah. ---
    small = cv2.resize(frame, None, fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_AREA)
    results_ocr = reader.readtext(
        small,
        text_threshold=CONFIG["ocr_text_threshold"],
        low_text=CONFIG["ocr_low_text"],
    )
    boxes = []
    for (bbox, text, prob) in results_ocr:
        if prob > CONFIG["ocr_conf_min"]:
            pts = np.array(bbox, dtype=np.float32) / ocr_scale  # scale balik ke ukuran asli
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
        # OCR gagal total frame ini tapi baru saja ada teks -> kemungkinan
        # besar teksnya masih ada, cuma missed. Pakai box terakhir sebentar
        # supaya tidak "berkedip" lolos 1-2 frame.
        boxes = last_boxes
        miss_streak += 1
    else:
        last_boxes = []
        miss_streak = 0

    # --- Mask = kotak-kotak hasil OCR saja (TIDAK ada SAM, TIDAK ada
    # tebak-tebakan objek). Ini menjamin mask tidak pernah melebar ke luar
    # area teks yang benar-benar terdeteksi -> orang/objek lain aman. ---
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

# --- Hitung resolusi kerja ProPainter dari resolusi video ASLI ---
# Kalau video aslinya sudah <= propainter_max_dim, proses di resolusi ASLI
# (tidak ada upscale sama sekali -> tidak ada blur tambahan). Kalau lebih
# besar, diskalakan turun secukupnya saja (bukan dipotong jauh ke ukuran
# tetap seperti sebelumnya), lalu dibulatkan ke kelipatan 8.
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

# Setting ProPainter dimulai dari CONFIG dan dibawa antar-chunk (kalau satu
# chunk kena OOM lalu turun settingnya, chunk berikutnya mulai dari setting
# yang sudah aman itu, bukan mengulang dari awal lagi).
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

    # Symlink (bukan copy) supaya tidak menggandakan pemakaian disk/waktu
    for local_i, global_i in enumerate(range(start_f, end_f)):
        name = f"{global_i:05d}.png"
        os.symlink(os.path.join(ABS_FRAMES, name), os.path.join(CHUNK_FRAMES_DIR, f"{local_i:05d}.png"))
        os.symlink(os.path.join(ABS_MASKS, name), os.path.join(CHUNK_MASKS_DIR, f"{local_i:05d}.png"))

    result = None
    for attempt in range(CONFIG["oom_retry_max"] + 1):
        cmd = build_cmd(CHUNK_FRAMES_DIR, CHUNK_MASKS_DIR, CHUNK_OUT_DIR, sub_len, nbr_len)
        print(f"    Percobaan {attempt + 1}: subvideo_length={sub_len}, neighbor_length={nbr_len}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env_vars)

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
        print(result.stderr[-3000:] if result else "")
        raise RuntimeError(f"ProPainter gagal pada chunk {chunk_i + 1}. Lihat log di atas!")

    chunk_frames_out, found_count = find_best_frames_dir(CHUNK_OUT_DIR, exclude_dirs=[CHUNK_FRAMES_DIR, CHUNK_MASKS_DIR])
    if not chunk_frames_out or found_count == 0:
        raise RuntimeError(f"Tidak ditemukan frame hasil ProPainter untuk chunk {chunk_i + 1}.")

    chunk_pngs = sorted(glob.glob(os.path.join(chunk_frames_out, "*.png")))
    for p in chunk_pngs:
        shutil.move(p, os.path.join(COMBINED_DIR, f"{global_out_idx:05d}.png"))
        global_out_idx += 1

    # Bersihkan folder sementara chunk ini segera -> membebaskan disk & memori
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

    # --- Feather via distance transform (FIX: mencegah subtitle asli tembus balik) ---
    # Piksel yang jauh dari tepi mask (di dalam area yang di-inpaint) dijamin
    # alpha=1.0 PENUH. Transisi halus (0->1) hanya terjadi di cincin selebar
    # feather_radius piksel di TEPI LUAR mask. Ini beda dari GaussianBlur biasa,
    # yang dulu mengencerkan alpha di SELURUH area mask (termasuk bagian yang
    # seharusnya solid) -> itu penyebab subtitle asli ikut numpang balik.
    mask_bin = (mask_img > 127).astype(np.uint8)
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    alpha = np.clip(dist / feather_radius, 0.0, 1.0)[..., None]

    blended = inpaint_img.astype(np.float32) * alpha + orig_img.astype(np.float32) * (1 - alpha)

    # --- Sharpening ringan HANYA di area yang dipengaruhi inpaint (alpha>0) ---
    # ProPainter bekerja di resolusi pp_width x pp_height lalu di-upscale;
    # kalau upscale-nya signifikan, hasilnya sedikit lebih lembek dari
    # sekitarnya yang detail (tangan/bulu/tekstur). Unsharp mask ringan ini
    # mengembalikan sedikit ketajaman tanpa memengaruhi area di luar mask.
    if CONFIG["sharpen_amount"] > 0:
        blur_for_sharpen = cv2.GaussianBlur(blended, (0, 0), 1.0)
        sharpened = cv2.addWeighted(blended, 1 + CONFIG["sharpen_amount"], blur_for_sharpen, -CONFIG["sharpen_amount"], 0)
        sharpen_mask = (alpha > 0.05).astype(np.float32)
        blended = sharpened * sharpen_mask + blended * (1 - sharpen_mask)

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(BLEND_DIR, f"{i:05d}.png"), blended)

    if (i + 1) % 50 == 0:
        print(f"  Blended: {i + 1}/{n_frames}")

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