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

    # Step 1: STT
    audio_bytes = await audio.read()
    transcript = transcribe_speech_to_text(audio_bytes)
    if transcript.startswith("[ERROR]"):
        return JSONResponse(status_code=500, content={"error": transcript})

    # Step 2: LLM
    llm_input = transcript
    if mode == "normalize":
        llm_input = f"[Normalize ke Bahasa Indonesia baku]: {transcript}"
    
    response_text = generate_response(llm_input)
    if response_text.startswith("[ERROR]"):
        return JSONResponse(status_code=500, content={"error": response_text})

    # Step 3: TTS
    output_audio_path = transcribe_text_to_speech(response_text)
    if output_audio_path.startswith("[ERROR]"):
        return JSONResponse(status_code=500, content={"error": output_audio_path})

    latency = round(time.time() - start_time, 2)
    print(f"[PIPELINE] transcript={transcript!r} | response={response_text!r} | latency={latency}s")

    return FileResponse(
        output_audio_path,
        media_type="audio/wav",
        headers={
            "X-Transcript": transcript,
            "X-Response-Text": response_text,
            "X-Latency": str(latency)
        }
    )