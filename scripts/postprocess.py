#!/usr/bin/env python3
"""
postprocess.py
1. Transkripsi audio dari output.mp4 memakai faster-whisper (jalan di CPU
   GitHub Actions runner, tidak butuh API key tambahan untuk Whisper).
2. Kirim transkrip ke Gemini API untuk menghasilkan judul + deskripsi rilis.
3. Tulis release_notes.md yang dipakai job release di workflow.
"""
import argparse
import os
import sys


def transcribe(video_path, model_size="small"):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(video_path, beam_size=5)

    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())

    full_text = " ".join(text_parts).strip()
    return full_text, info.language


def generate_release_notes(transcript, api_key, source_url):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
Kamu membantu menulis catatan rilis (release notes) singkat untuk sebuah video
hasil proses otomatis "hapus subtitle/watermark/teks" dari video sumber berikut:
{source_url}

Berikut transkrip audio videonya (mungkin kosong jika video tanpa suara):
\"\"\"{transcript[:4000]}\"\"\"

Tulis dalam format:
JUDUL: <judul singkat, maksimal 12 kata>
DESKRIPSI:
<2-4 kalimat ringkas tentang isi video, netral, tanpa klaim berlebihan>
""".strip()

    response = model.generate_content(prompt)
    return response.text.strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path output.mp4")
    p.add_argument("--source-url", default="")
    p.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--output", required=True, help="Path release_notes.md")
    return p.parse_args()


def main():
    args = parse_args()

    transcript = ""
    try:
        transcript, lang = transcribe(args.video, model_size=args.whisper_model)
        print(f"[postprocess] Transkrip ({lang}): {transcript[:200]}...")
    except Exception as e:
        print(f"[postprocess] Transkripsi gagal/skip: {e}", file=sys.stderr)

    notes = None
    if args.gemini_api_key:
        try:
            notes = generate_release_notes(transcript, args.gemini_api_key, args.source_url)
        except Exception as e:
            print(f"[postprocess] Gemini gagal/skip: {e}", file=sys.stderr)

    if not notes:
        notes = (
            "JUDUL: Video hasil auto-remove subtitle/watermark\n"
            "DESKRIPSI:\n"
            f"Diproses otomatis dari sumber: {args.source_url}\n"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(notes + "\n")

    if transcript:
        with open(os.path.join(os.path.dirname(args.output) or ".", "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

    print(f"[postprocess] release notes ditulis ke {args.output}")


if __name__ == "__main__":
    main()
