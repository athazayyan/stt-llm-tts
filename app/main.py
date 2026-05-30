import os
import time
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech

app = FastAPI(title="Code-Switching Speech-to-Speech System")

@app.get("/")
def root():
    return {"status": "running", "message": "Voice CS System aktif"}

@app.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    mode: str = Query("preserve", enum=["preserve", "normalize"])
):
    start_time = time.time()

    # ------------------------------------------------------------------------
    # STEP 1: Speech-to-Text (STT) - Whisper.cpp
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("START PIPELINE - Memproses Permintaan Suara Baru")
    print(f"   [Mode Respons] -> {mode}")
    print("="*50)
    print("--- [STEP 1: RUNNING STT (WHISPER)] ---")
    
    audio_bytes = await audio.read()
    transcript = transcribe_speech_to_text(audio_bytes)
    
    if transcript.startswith("[ERROR]"):
        print(f"[STT FAILED]: {transcript}")
        return JSONResponse(status_code=500, content={"error": transcript})
    
    print(f"[STT RESULT] -> User berkata: \"{transcript}\"")

    # ------------------------------------------------------------------------
    # STEP 2: Large Language Model (LLM) - Gemini (Gemma-4-31b-it)
    # ------------------------------------------------------------------------
    print("\n--- [STEP 2: RUNNING LLM (GEMINI)] ---")
    
    response_text = generate_response(transcript, mode=mode)
    
    if response_text.startswith("[ERROR]"):
        print(f"[LLM FAILED]: {response_text}")
        return JSONResponse(status_code=500, content={"error": response_text})
    
    print(f"[LLM RESULT] -> Jawaban Asisten: \"{response_text}\"")

    # ------------------------------------------------------------------------
    # STEP 3: Text-to-Speech (TTS) - Coqui TTS (Wibowo)
    # ------------------------------------------------------------------------
    print("\n--- [STEP 3: RUNNING TTS (COQUI)] ---")
    
    output_audio_path = transcribe_text_to_speech(response_text)
    
    if output_audio_path.startswith("[ERROR]"):
        print(f"[TTS FAILED]: {output_audio_path}")
        return JSONResponse(status_code=500, content={"error": output_audio_path})
    
    print(f"[TTS RESULT] -> File audio sukses dibuat di: {output_audio_path}")

    # ------------------------------------------------------------------------
    # Selesai & Hitung Latency
    # ------------------------------------------------------------------------
    latency = round(time.time() - start_time, 2)
    print("\n" + "="*50)
    print(f"PIPELINE SUCCESS - Total Waktu Eksekusi: {latency} detik")
    print("="*50 + "\n")

    return FileResponse(
        output_audio_path,
        media_type="audio/wav",
        headers={
            "X-Transcript": transcript,
            "X-Response-Text": response_text,
            "X-Latency": str(latency)
        }
    )