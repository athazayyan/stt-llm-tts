import os
import uuid
import tempfile
import subprocess
import re  # <-- WAJIB: Untuk membersihkan timestamp [00:00:00]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(BASE_DIR, "..", "models", "whisper.cpp")

WHISPER_BINARY     = os.path.join(WHISPER_DIR, "build", "bin", "Release", "whisper-cli.exe")
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-small.bin")

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{uuid.uuid4()}{file_ext}")

        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        cmd = [
            WHISPER_BINARY,
            "-m", WHISPER_MODEL_PATH,
            "-f", audio_path,
            "-l", "auto",
            "-t", "8",
            "--prompt", "Saya sedang mempersiapkan dokumen dan checklist perjalanan umrah dan hajj dari Indonesia. Ya akhi, uridu check schedule penerbangan paling aman, tolong bantu book flight langsung menuju Jeddah lalu arrange transport min Makkah ila Madinah ghadan secara step by step."  
        ]

        try:
            # Ambil output langsung dari terminal (stdout) sebagai string
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return f"[ERROR] Whisper failed: {e}\nDetails: {e.stderr}"

        # PROSES PARSING: Ambil teks baris demi baris dari terminal
        full_output = res.stdout if res.stdout else ""
        lines = full_output.splitlines()

        transcript_lines = []
        for line in lines:
            # Hanya ambil baris yang berisi penanda waktu '-->'
            if "-->" in line:
                # Hilangkan format [00:00:00.000 --> 00:00:03.960] dengan Regex
                clean_line = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]", "", line).strip()
                if clean_line:
                    transcript_lines.append(clean_line)

        # Gabungkan potongan teks menjadi satu kalimat utuh
        if transcript_lines:
            return " ".join(transcript_lines)
        else:
            return f"[ERROR] Whisper tidak menghasilkan teks.\nLog:\n{res.stderr}\n{res.stdout}"