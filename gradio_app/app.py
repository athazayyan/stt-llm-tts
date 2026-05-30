import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
import numpy as np

BACKEND_URL = "http://localhost:8000/voice-chat"

def voice_chat(audio, mode):
    if audio is None:
        return None, "❌ Tidak ada audio.", "", ""

    sr, audio_data = audio

    # Pastikan format int16 agar valid sebagai .wav
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)

    # Simpan audio input ke file temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    # Kirim ke FastAPI
    with open(audio_path, "rb") as f:
        files    = {"audio": ("voice.wav", f, "audio/wav")}
        params   = {"mode": mode}
        try:
            response = requests.post(BACKEND_URL, files=files, params=params, timeout=60)
        except requests.exceptions.ConnectionError:
            return None, "❌ Backend tidak bisa dihubungi. Pastikan FastAPI sudah jalan.", "", ""

    # Cleanup input temp
    os.unlink(audio_path)

    if response.status_code == 200:
        # Ambil metadata dari response header
        transcript    = response.headers.get("X-Transcript", "-")
        response_text = response.headers.get("X-Response-Text", "-")
        latency       = response.headers.get("X-Latency", "-")

        # Simpan audio output
        output_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path, transcript, response_text, f"{latency} detik"
    else:
        err = response.json().get("error", response.text)
        return None, f"❌ Error: {err}", "", ""


# ── UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Voice CS Chatbot") as demo:
    gr.Markdown("# 🎙️ Voice Chatbot — Code-Switching ID/EN/AR")
    gr.Markdown("Rekam pertanyaan, sistem akan menjawab dengan suara.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                label="🎤 Rekam Pertanyaan"
            )
            mode_select = gr.Radio(
                choices=["preserve", "normalize"],
                value="preserve",
                label="Mode Respons",
                info="preserve = pertahankan code-switching | normalize = Bahasa Indonesia baku"
            )
            submit_btn = gr.Button("🔁 Proses", variant="primary")

        with gr.Column(scale=1):
            audio_output   = gr.Audio(type="filepath", label="🔊 Balasan Asisten")
            transcript_box = gr.Textbox(label="📝 Transkripsi (STT)", interactive=False)
            response_box   = gr.Textbox(label="💬 Teks Respons (LLM)", interactive=False)
            latency_box    = gr.Textbox(label="⏱️ Latency", interactive=False)

    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, mode_select],
        outputs=[audio_output, transcript_box, response_box, latency_box]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)