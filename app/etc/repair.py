"""
repair_audio.py
---------------
Konversi file audio yang "Format not recognised" ke WAV PCM 16-bit 16kHz.
Butuh: pip install pydub soundfile
Butuh: ffmpeg terinstall di sistem (https://ffmpeg.org/download.html)

Cara pakai:
    python repair_audio.py
"""

import os
import json
import shutil
import soundfile as sf
from pydub import AudioSegment


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR   = os.path.join(BASE_DIR, "..", "data", "audio")
BACKUP_DIR  = os.path.join(BASE_DIR, "..", "data", "audio_backup")
LOG_DIR     = os.path.join(BASE_DIR, "..", "log")

os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# Target format setelah konversi
TARGET_SR       = 16000   # Hz — standar untuk STT
TARGET_CHANNELS = 1       # mono
TARGET_BIT      = 16      # PCM 16-bit


def is_broken(audio_path: str) -> bool:
    """Return True jika soundfile tidak bisa baca file ini."""
    try:
        sf.info(audio_path)
        return False
    except Exception:
        return True


def repair_file(audio_path: str) -> tuple[bool, str]:
    """
    Coba baca dengan pydub (pakai ffmpeg di belakang layar),
    lalu export ulang sebagai WAV PCM standar.
    File asli dibackup dulu ke audio_backup/.
    """
    filename = os.path.basename(audio_path)

    # Backup file asli
    backup_path = os.path.join(BACKUP_DIR, filename)
    if not os.path.exists(backup_path):
        shutil.copy2(audio_path, backup_path)

    try:
        # pydub pakai ffmpeg untuk decode — bisa handle M4A, AAC,
        # OGG, MP3, bahkan WAV dengan codec aneh
        audio = AudioSegment.from_file(audio_path)

        # Normalize ke target format
        audio = (
            audio
            .set_channels(TARGET_CHANNELS)
            .set_frame_rate(TARGET_SR)
            .set_sample_width(TARGET_BIT // 8)   # bytes
        )

        # Overwrite file lama dengan WAV PCM yang bersih
        audio.export(audio_path, format="wav")
        return True, "repaired"

    except Exception as e:
        # Kembalikan backup jika repair gagal
        shutil.copy2(backup_path, audio_path)
        return False, str(e)


def main():
    audio_files = sorted(
        f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")
    )

    print(f"Scan {len(audio_files)} file audio...\n")

    broken   = [f for f in audio_files if is_broken(os.path.join(AUDIO_DIR, f))]
    ok_count = len(audio_files) - len(broken)

    print(f"OK       : {ok_count}")
    print(f"Broken   : {len(broken)}")

    if not broken:
        print("\nSemua file sudah dalam format yang benar.")
        return

    print(f"\nBackup file asli ke: {BACKUP_DIR}")
    print("Mulai repair...\n")

    results = []
    for i, filename in enumerate(broken, 1):
        path = os.path.join(AUDIO_DIR, filename)
        print(f"[{i:>3}/{len(broken)}] {filename} ...", end=" ", flush=True)

        success, msg = repair_file(path)
        status = "OK" if success else f"GAGAL: {msg}"
        print(status)

        results.append({
            "file":    filename,
            "success": success,
            "note":    msg,
        })

    # Ringkasan
    n_ok   = sum(r["success"] for r in results)
    n_fail = len(results) - n_ok

    print(f"\nRepair selesai: {n_ok} berhasil, {n_fail} gagal")

    if n_fail:
        print("\nFile yang GAGAL direpair (kemungkinan corrupt total):")
        for r in results:
            if not r["success"]:
                print(f"  - {r['file']}: {r['note']}")

    # Simpan log
    log_path = os.path.join(LOG_DIR, "repair_log.json")
    with open(log_path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nLog disimpan: {log_path}")

    # Verifikasi ulang setelah repair
    print("\nVerifikasi ulang file yang sudah direpair...")
    still_broken = []
    for r in results:
        if r["success"]:
            path = os.path.join(AUDIO_DIR, r["file"])
            if is_broken(path):
                still_broken.append(r["file"])

    if still_broken:
        print(f"⚠ Masih bermasalah setelah repair: {still_broken}")
    else:
        print("Semua file yang direpair sudah bisa dibaca.")


if __name__ == "__main__":
    main()