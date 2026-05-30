# Multilingual Code-Switching Speech-to-Speech System (ID-EN-AR)

Sistem Speech-to-Speech (S2S) interaktif yang dirancang untuk menangani fenomena code-switching (pencampuran bahasa) antara Bahasa Indonesia, Inggris, dan Arab. Sistem ini bekerja secara end-to-end untuk menerima input suara pengguna melalui mikrofon, memprosesnya melalui model bahasa (LLM), dan mengembalikan respons cerdas dalam bentuk audio kembali.

Sistem menyediakan dua mode respons utama:
1. **preserve**: Mempertahankan pola code-switching pencampuran bahasa (ID/EN/AR) secara natural menggunakan aturan penandaan tag khusus.
2. **normalize**: Mentransformasikan seluruh jawaban ke dalam bentuk Bahasa Indonesia yang baku dan formal.

---

## Alur Arsitektur Sistem

1. **Speech-to-Text (STT)**: Transkripsi audio lokal berlatensi rendah berbasis c++ menggunakan Whisper.cpp (Model ggml-small.bin).
2. **Large Language Model (LLM)**: Pemrosesan konteks jawaban menggunakan Gemini (Gemma-4-31b-it) dengan instruksi sistem berlapis untuk menjamin keakuratan gaya bahasa.
3. **Text-to-Speech (TTS)**: Sintesis suara teks menjadi file audio .wav menggunakan Coqui TTS dengan model kustom Wibowo serta modul normalisasi fonetik Arab-Inggris terintegrasi.

---

## Struktur Proyek

```text
.
├── .gitignore
├── README.md
├── analisis_pipeline.py
├── app
│   ├── coqui_utils
│   │   ├── checkpoint_1260000-inference.pth
│   │   ├── config.json
│   │   └── speakers.pth
│   ├── en_transliteration.json
│   ├── llm.py
│   ├── main.py
│   ├── stt.py
│   └── tts.py
├── data
│   ├── corpus
│   │   ├── audio
│   │   ├── output
│   │   └── transcripts
│   │       └── scripts.json
│   └── manifests
├── gradio_app
│   └── app.py
├── log
│   ├── llm_hasil_normalize.txt
│   ├── llm_hasil_preserve.txt
│   ├── llm_log_normalize.json
│   ├── llm_log_preserve.json
│   ├── stt_log.json
│   ├── stt_log_no_prompt.json
│   ├── summary.json
│   ├── tts_log_normalize.json
│   └── tts_log_preserve.json
└── requirements.txt
```

## Panduan Instalasi dan Konfigurasi Awal

### 1. Kloning Repositori

```bash
git clone https://github.com/athazayyan/stt-llm-tts.git
cd UAS-Praktikum-Pemrosesan-Bahasa-Alami
```

### 2. Membuat dan Mengaktifkan Virtual Environment

```bash
python -m venv uas-nlp

.\uas-nlp\Scripts\Activate.ps1

.\uas-nlp\Scripts\activate.bat
```

### 3. Instalasi Dependensi / Library

```bash
pip install -r requirements.txt
```

### 4. Pengaturan Environment Variable (.env)

```ini
GEMINI_API_KEY=isi_api_key_gemini_anda_di_sini
```

## Cara Menjalankan Sistem Interaktif (Gradio App)

### Langkah 1: Jalankan Backend Core Engine (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Langkah 2: Jalankan Antarmuka Pengguna (Gradio Frontend)

```bash
python gradio_app/app.py
```

Akses:

`http://localhost:7860`

## Cara Menjalankan Skrip Evaluasi dan Analisis Pipeline

```bash
python analisis_pipeline.py
```

Skrip akan membaca seluruh file JSON di folder `log/`, mengolah metrik performa masing-masing komponen (STT, LLM, TTS), dan memperbarui file `log/summary.json`.
