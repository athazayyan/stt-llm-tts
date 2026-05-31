"""
dedup_audio.py
--------------
Hapus file duplikat yang punya suffix (1), (2), dst.
Contoh: '2341_audio01(1).wav' → duplikat dari '2341_audio01.wav'

Jalankan SEBELUM repair_audio.py dan eda_audio.py.
"""

import os
import re
import json
import hashlib


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "..", "data", "audio")
LOG_DIR   = os.path.join(BASE_DIR, "..", "log")
os.makedirs(LOG_DIR, exist_ok=True)

# Pattern: nama file yang punya (angka) sebelum ekstensi
# contoh: 2341_audio01(1).wav, 2341_audio01 (2).wav
DUPE_PATTERN = re.compile(r"^(.+?)\s*\(\d+\)(\.wav)$", re.IGNORECASE)


def file_hash(path: str, chunk=65536) -> str:
    """MD5 hash isi file untuk cek apakah benar-benar identik."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk_data := f.read(chunk):
            h.update(chunk_data)
    return h.hexdigest()


def main():
    all_files = sorted(os.listdir(AUDIO_DIR))
    wav_files = [f for f in all_files if f.lower().endswith(".wav")]

    print(f"Total file .wav: {len(wav_files)}\n")

    removed = []
    kept    = []
    no_original = []  # duplikat tapi original tidak ada

    for filename in wav_files:
        m = DUPE_PATTERN.match(filename)
        if not m:
            continue  # bukan duplikat

        base_name    = m.group(1)       # '2341_audio01'
        ext          = m.group(2)       # '.wav'
        original     = base_name + ext  # '2341_audio01.wav'
        dupe_path    = os.path.join(AUDIO_DIR, filename)
        original_path = os.path.join(AUDIO_DIR, original)

        print(f"Duplikat ditemukan: {filename}")
        print(f"  → Original       : {original}")

        if not os.path.exists(original_path):
            # Original tidak ada → rename duplikat jadi original
            os.rename(dupe_path, original_path)
            print(f"  → Original tidak ada, RENAME ke {original}")
            no_original.append({"dupe": filename, "action": f"renamed to {original}"})
            continue

        # Bandingkan hash — kalau benar-benar sama, hapus duplikat
        hash_dupe     = file_hash(dupe_path)
        hash_original = file_hash(original_path)

        if hash_dupe == hash_original:
            os.remove(dupe_path)
            print(f"  → Hash identik, HAPUS {filename}")
            removed.append({"file": filename, "reason": "identical hash"})
        else:
            # Isi berbeda (mungkin beda sample rate karena repair sebagian)
            # Pertahankan yang ukurannya lebih besar (biasanya kualitas lebih tinggi)
            size_dupe     = os.path.getsize(dupe_path)
            size_original = os.path.getsize(original_path)

            if size_dupe > size_original:
                # Duplikat lebih besar → ganti original dengan duplikat
                os.replace(dupe_path, original_path)
                print(f"  → Hash beda, duplikat lebih besar → GANTI original dengan duplikat")
                removed.append({"file": filename, "reason": "replaced original (larger)"})
            else:
                # Original lebih besar atau sama → hapus duplikat
                os.remove(dupe_path)
                print(f"  → Hash beda, original lebih besar → HAPUS duplikat")
                removed.append({"file": filename, "reason": "original is larger, dupe removed"})

        kept.append(original)
        print()

    # Ringkasan
    print("=" * 50)
    print(f"Duplikat dihapus/dihandle : {len(removed)}")
    print(f"Rename (original hilang)  : {len(no_original)}")

    if not removed and not no_original:
        print("Tidak ada duplikat ditemukan.")

    # Cek sisa file
    remaining = sorted(f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav"))
    print(f"Total file tersisa        : {len(remaining)}")

    # Simpan log
    log = {
        "removed":     removed,
        "renamed":     no_original,
        "total_after": len(remaining),
    }
    log_path = os.path.join(LOG_DIR, "dedup_log.json")
    with open(log_path, "w", encoding="utf8") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)
    print(f"Log disimpan: {log_path}")


if __name__ == "__main__":
    main()