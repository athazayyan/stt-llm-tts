import os
import json
import wave
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from collections import defaultdict


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "..", "data", "audio")
LOG_DIR   = os.path.join(BASE_DIR, "..", "log")
os.makedirs(LOG_DIR, exist_ok=True)


# ─── Konfigurasi ──────────────────────────────────────────────────────────────
SILENCE_THRESHOLD_DB = -40   # dB, di bawah ini dianggap silence
TOP_N_PRINT          = 10    # jumlah baris yang dicetak di preview


# ─── Helper ───────────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> tuple[str, str]:
    """
    '2128_audio01.wav' → speaker_id='2128', utterance_id='audio01'
    """
    base = filename.replace(".wav", "")
    parts = base.split("_", 1)
    speaker_id   = parts[0] if len(parts) > 0 else "unknown"
    utterance_id = parts[1] if len(parts) > 1 else "unknown"
    return speaker_id, utterance_id


def check_playable(audio_path: str) -> tuple[bool, str]:
    """Cek apakah file bisa dibaca/diputar."""
    try:
        info = sf.info(audio_path)
        _ = info.samplerate   # trigger actual read
        return True, "ok"
    except Exception as e:
        return False, str(e)


def get_wav_bitdepth(audio_path: str) -> int | str:
    """Baca bit depth dari header WAV (wave module, tanpa decode audio)."""
    try:
        with wave.open(audio_path, "rb") as wf:
            return wf.getsampwidth() * 8   # sampwidth dalam byte → bit
    except Exception:
        return "n/a"


def extract_features(audio_path: str) -> dict:
    """
    Ekstrak semua fitur dari satu file audio.
    Return dict kosong jika file corrupt/tidak bisa diputar.
    """
    playable, err_msg = check_playable(audio_path)
    filename          = os.path.basename(audio_path)
    speaker_id, utterance_id = parse_filename(filename)

    base = {
        "filename":     filename,
        "speaker_id":   speaker_id,
        "utterance_id": utterance_id,
        "playable":     playable,
        "error":        err_msg if not playable else "",
    }

    if not playable:
        return base

    # ── Metadata dari header (cepat, tanpa decode penuh) ──────────────────────
    info      = sf.info(audio_path)
    bit_depth = get_wav_bitdepth(audio_path)

    base.update({
        "sample_rate_hz":  info.samplerate,
        "channels":        info.channels,
        "bit_depth":       bit_depth,
        "duration_sec":    round(info.duration, 3),
        "total_samples":   info.frames,
    })

    # ── Load audio untuk analisis sinyal (mono, sr asli) ─────────────────────
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Amplitude & RMS
    rms         = float(np.sqrt(np.mean(y ** 2)))
    rms_db      = float(20 * np.log10(rms + 1e-9))
    peak_amp    = float(np.max(np.abs(y)))
    peak_db     = float(20 * np.log10(peak_amp + 1e-9))

    # Silence ratio
    frame_len   = 2048
    hop_len     = 512
    rms_frames  = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_frames_db = 20 * np.log10(rms_frames + 1e-9)
    silence_ratio = float(np.mean(rms_frames_db < SILENCE_THRESHOLD_DB))

    # Spectral features
    zcr         = float(np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_len)))
    spec_centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)
    ))
    spec_bandwidth = float(np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_len)
    ))
    spec_rolloff = float(np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_len)
    ))

    base.update({
        "rms":                  round(rms, 6),
        "rms_db":               round(rms_db, 2),
        "peak_amplitude":       round(peak_amp, 6),
        "peak_db":              round(peak_db, 2),
        "silence_ratio":        round(silence_ratio, 4),
        "zcr_mean":             round(zcr, 6),
        "spectral_centroid_hz": round(spec_centroid, 2),
        "spectral_bandwidth_hz":round(spec_bandwidth, 2),
        "spectral_rolloff_hz":  round(spec_rolloff, 2),
    })

    return base


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    audio_files = sorted(
        f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")
    )

    if not audio_files:
        print(f"[ERROR] Tidak ada file .wav di: {AUDIO_DIR}")
        return

    print(f"Ditemukan {len(audio_files)} file audio\n")

    records = []
    for i, file in enumerate(audio_files, 1):
        path = os.path.join(AUDIO_DIR, file)
        print(f"[{i:>3}/{len(audio_files)}] {file} ...", end=" ", flush=True)
        rec = extract_features(path)
        records.append(rec)
        status = "OK" if rec["playable"] else f"GAGAL: {rec['error']}"
        print(status)

    df = pd.DataFrame(records)

    # ── Simpan CSV lengkap ────────────────────────────────────────────────────
    csv_path = os.path.join(LOG_DIR, "eda_audio.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV disimpan: {csv_path}")

    # ── Simpan JSON lengkap ───────────────────────────────────────────────────
    json_path = os.path.join(LOG_DIR, "eda_audio.json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    print(f"JSON disimpan: {json_path}\n")

    # ── Report ke terminal ────────────────────────────────────────────────────
    df_ok = df[df["playable"] == True].copy()

    print("=" * 60)
    print("RINGKASAN EDA AUDIO")
    print("=" * 60)

    # 1. Playability
    n_ok   = df["playable"].sum()
    n_fail = len(df) - n_ok
    print(f"\n[1] PLAYABILITY")
    print(f"    Bisa diputar  : {n_ok}/{len(df)}")
    if n_fail:
        print(f"    GAGAL ({n_fail})    :")
        for _, row in df[~df["playable"]].iterrows():
            print(f"      - {row['filename']}: {row['error']}")

    # 2. Sample rate & bit depth
    print(f"\n[2] SAMPLE RATE & BIT DEPTH")
    sr_counts = df_ok["sample_rate_hz"].value_counts()
    bd_counts = df_ok["bit_depth"].value_counts()
    print(f"    Sample rate (Hz) : {sr_counts.to_dict()}")
    print(f"    Bit depth (bit)  : {bd_counts.to_dict()}")
    print(f"    Channels         : {df_ok['channels'].value_counts().to_dict()}")

    # 3. Durasi
    print(f"\n[3] DURASI (detik)")
    print(f"    Min    : {df_ok['duration_sec'].min():.2f}")
    print(f"    Max    : {df_ok['duration_sec'].max():.2f}")
    print(f"    Mean   : {df_ok['duration_sec'].mean():.2f}")
    print(f"    Median : {df_ok['duration_sec'].median():.2f}")
    print(f"    Total  : {df_ok['duration_sec'].sum():.2f} detik "
          f"({df_ok['duration_sec'].sum()/60:.2f} menit)")

    # 4. Amplitude & RMS
    print(f"\n[4] AMPLITUDE & RMS")
    print(f"    RMS rata-rata    : {df_ok['rms'].mean():.5f}")
    print(f"    RMS dB rata-rata : {df_ok['rms_db'].mean():.2f} dB")
    print(f"    Peak dB rata-rata: {df_ok['peak_db'].mean():.2f} dB")
    low_vol = df_ok[df_ok["rms_db"] < -30]
    if not low_vol.empty:
        print(f"    ⚠ Volume rendah (<-30dB): {len(low_vol)} file")
        for _, r in low_vol.head(5).iterrows():
            print(f"      - {r['filename']} ({r['rms_db']:.1f} dB)")

    # 5. Silence ratio
    print(f"\n[5] SILENCE RATIO (threshold {SILENCE_THRESHOLD_DB} dB)")
    print(f"    Mean   : {df_ok['silence_ratio'].mean():.3f}")
    print(f"    Median : {df_ok['silence_ratio'].median():.3f}")
    high_silence = df_ok[df_ok["silence_ratio"] > 0.5]
    if not high_silence.empty:
        print(f"    ⚠ Silence >50%: {len(high_silence)} file")
        for _, r in high_silence.head(5).iterrows():
            print(f"      - {r['filename']} ({r['silence_ratio']:.2f})")

    # 6. Spectral features
    print(f"\n[6] SPECTRAL FEATURES")
    print(f"    ZCR mean             : {df_ok['zcr_mean'].mean():.6f}")
    print(f"    Spectral centroid Hz : {df_ok['spectral_centroid_hz'].mean():.1f}")
    print(f"    Spectral bandwidth Hz: {df_ok['spectral_bandwidth_hz'].mean():.1f}")
    print(f"    Spectral rolloff Hz  : {df_ok['spectral_rolloff_hz'].mean():.1f}")

    # 7. Distribusi per speaker
    print(f"\n[7] DISTRIBUSI PER SPEAKER ID")
    spk = df_ok.groupby("speaker_id").agg(
        jumlah_audio=("filename", "count"),
        durasi_total=("duration_sec", "sum"),
        durasi_mean=("duration_sec", "mean"),
        rms_db_mean=("rms_db", "mean"),
        silence_mean=("silence_ratio", "mean"),
    ).round(2)
    print(spk.to_string())

    # 8. Distribusi per utterance ID
    print(f"\n[8] DISTRIBUSI PER UTTERANCE ID")
    utt = df_ok.groupby("utterance_id").agg(
        jumlah_speaker=("speaker_id", "nunique"),
        durasi_mean=("duration_sec", "mean"),
        durasi_std=("duration_sec", "std"),
        rms_db_mean=("rms_db", "mean"),
        silence_mean=("silence_ratio", "mean"),
    ).round(3)
    print(utt.to_string())

    print("\n" + "=" * 60)
    print("EDA selesai.")
    print("=" * 60)


if __name__ == "__main__":
    main()