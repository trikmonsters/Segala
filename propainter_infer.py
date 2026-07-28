#!/usr/bin/env python3
"""
propainter_infer.py
Dijalankan DI DALAM Kaggle Kernel (GPU). Tugasnya:
1. Clone repo resmi ProPainter.
2. Install dependencies.
3. Jalankan inference_propainter.py memakai frames + masks yang sudah
   disiapkan oleh GitHub Actions (di-mount Kaggle sebagai dataset input).
4. Salin hasil output.mp4 ke /kaggle/working/output/ supaya bisa diambil
   lewat `kaggle kernels output` dari GitHub Actions.

Catatan:
- Weight ProPainter (ProPainter.pth, recurrent_flow_completion.pth, raft-things.pth)
  akan otomatis terunduh saat inferensi pertama kali dijalankan (sesuai README resmi
  ProPainter), jadi TIDAK perlu didownload manual di sini.
- Dataset input Kaggle diharapkan berstruktur:
    /kaggle/input/<dataset-slug>/work/frames/frame_000000.png ...
    /kaggle/input/<dataset-slug>/work/masks_refined/mask_000000.png ...
"""
import glob
import os
import shutil
import subprocess
import sys

KAGGLE_INPUT_ROOT = "/kaggle/input"
WORKDIR = "/kaggle/working"
REPO_DIR = os.path.join(WORKDIR, "ProPainter")
OUTPUT_DIR = os.path.join(WORKDIR, "output")


def run(cmd, cwd=None):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def find_input_dataset_dir():
    """Cari folder dataset yang di-attach ke kernel ini (nama slug bisa apa saja)."""
    candidates = [d for d in glob.glob(os.path.join(KAGGLE_INPUT_ROOT, "*")) if os.path.isdir(d)]
    if not candidates:
        raise SystemExit("Tidak ada dataset input yang ter-attach ke kernel ini.")
    # Pakai dataset pertama yang punya folder work/frames
    for c in candidates:
        if os.path.isdir(os.path.join(c, "work", "frames")):
            return c
    return candidates[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(REPO_DIR):
        run(f"git clone --depth 1 https://github.com/sczhou/ProPainter.git {REPO_DIR}")

    run(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=REPO_DIR)

    dataset_dir = find_input_dataset_dir()
    frames_dir = os.path.join(dataset_dir, "work", "frames")
    masks_dir = os.path.join(dataset_dir, "work", "masks_refined")

    if not os.path.isdir(frames_dir) or not os.path.isdir(masks_dir):
        raise SystemExit(
            f"Struktur dataset tidak sesuai. Diharapkan {frames_dir} dan {masks_dir}"
        )

    print(f"[propainter_infer] frames: {frames_dir}")
    print(f"[propainter_infer] masks : {masks_dir}")

    results_dir = os.path.join(REPO_DIR, "results")
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    # --fp16 : hemat memori, cocok untuk GPU 16GB (T4/P100) di Kaggle free tier
    # --subvideo_length : proses per-batch supaya tidak OOM pada video panjang
    run(
        f"{sys.executable} inference_propainter.py "
        f"--video {frames_dir} "
        f"--mask {masks_dir} "
        f"--fp16 "
        f"--subvideo_length 60 "
        f"--save_fps 30",
        cwd=REPO_DIR,
    )

    # Cari hasil output (ProPainter menyimpan sebagai <nama_input>/inpaint_out.mp4
    # atau <nama_input>.mp4 tergantung mode input; kita cari file mp4 apa saja
    # di dalam results/ dan salin sebagai output.mp4)
    produced = glob.glob(os.path.join(results_dir, "**", "*.mp4"), recursive=True)
    if not produced:
        raise SystemExit("ProPainter tidak menghasilkan file mp4 di folder results/.")

    final_path = os.path.join(OUTPUT_DIR, "output.mp4")
    shutil.copy(produced[0], final_path)
    print(f"[propainter_infer] Output final -> {final_path}")

    with open(os.path.join(OUTPUT_DIR, "STATUS_OK"), "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()
