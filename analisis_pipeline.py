import os
import json
import time
import unicodedata
import re
import shutil
import argparse
from pathlib import Path

from app.stt import transcribe_speech_to_text
from app.llm import generate_response, FALLBACK_MSG
from app.tts import transcribe_text_to_speech

# ── Path config ───────────────────────────────────────────────────────────────
AUDIO_DIR      = Path("data/corpus/audio")
TRANSCRIPT_REF = Path("data/corpus/transcripts/scripts.json")
OUTPUT_DIR     = Path("data/corpus/output")
LOG_DIR        = Path("log")
STT_LOG        = LOG_DIR / "stt_log.json"

def get_llm_log_path(mode: str) -> Path:
    return LOG_DIR / f"llm_log_{mode}.json"

def get_tts_log_path(mode: str) -> Path:
    return LOG_DIR / f"tts_log_{mode}.json"

def get_output_dir(mode: str) -> Path:
    p = OUTPUT_DIR / mode
    os.makedirs(p, exist_ok=True)
    return p

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def compute_wer(ref: str, hyp: str) -> float:
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    if not r: return 0.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return round(d[len(r)][len(h)] / len(r), 4)

def compute_cer(ref: str, hyp: str) -> float:
    r = list(normalize_text(ref).replace(" ", ""))
    h = list(normalize_text(hyp).replace(" ", ""))
    if not r: return 0.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return round(d[len(r)][len(h)] / len(r), 4)

def get_utterance_id(filename_stem: str) -> str:
    raw   = filename_stem.split("_")[-1].lower()
    match = re.search(r'\d+', raw)
    if match:
        return "audio" + str(int(match.group()))
    return "audio1"

def load_ref_scripts() -> dict:
    with open(TRANSCRIPT_REF, "r", encoding="utf-8") as f:
        return json.load(f)

def save_log(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[LOG] Disimpan → {path}")

# ── STAGE 1: STT ──────────────────────────────────────────────────────────────
def run_stt(audio_files: list, ref_scripts: dict) -> dict:
    results   = {}
    errors    = []
    total_wer = total_cer = success = 0

    print(f"\n{'='*60}")
    print(f"[STT] Memproses {len(audio_files)} file...")
    print(f"{'='*60}")

    for i, audio_path in enumerate(audio_files, 1):
        stem         = audio_path.stem
        utterance_id = get_utterance_id(stem)
        ref_text     = ref_scripts.get(utterance_id, "")

        print(f"[{i}/{len(audio_files)}] {audio_path.name}", end=" ... ")
        t0 = time.time()

        try:
            with open(audio_path, "rb") as f:
                transcript = transcribe_speech_to_text(f.read())

            if transcript.startswith("[ERROR]"):
                raise RuntimeError(transcript)

            latency   = round(time.time() - t0, 2)
            wer_score = compute_wer(ref_text, transcript)
            cer_score = compute_cer(ref_text, transcript)

            results[stem] = {
                "file": audio_path.name, "utterance_id": utterance_id,
                "reference": ref_text,   "transcript": transcript,
                "wer": wer_score,        "cer": cer_score,
                "latency_s": latency,    "status": "ok"
            }
            total_wer += wer_score
            total_cer += cer_score
            success   += 1
            print(f"OK | WER={wer_score} CER={cer_score} | {latency}s")

        except Exception as e:
            latency = round(time.time() - t0, 2)
            results[stem] = {
                "file": audio_path.name, "utterance_id": utterance_id,
                "reference": ref_text,   "transcript": None,
                "wer": None,             "cer": None,
                "latency_s": latency,    "status": "error", "error": str(e)
            }
            errors.append(audio_path.name)
            print(f"ERROR: {e}")

    avg_wer = round(total_wer / success, 4) if success else None
    avg_cer = round(total_cer / success, 4) if success else None

    output = {
        "summary": {
            "total": len(audio_files), "success": success,
            "errors": len(errors),     "error_files": errors,
            "avg_wer": avg_wer,        "avg_cer": avg_cer
        },
        "results": results
    }
    save_log(output, STT_LOG)
    print(f"\n[STT SELESAI] {success}/{len(audio_files)} | Avg WER={avg_wer} CER={avg_cer}")
    return output

# ── STAGE 2: LLM ──────────────────────────────────────────────────────────────
def run_llm(mode: str = "normalize") -> None:
    LLM_LOG = get_llm_log_path(mode)
    LLM_TXT = LOG_DIR / f"llm_hasil_{mode}.txt"

    if not STT_LOG.exists():
        raise FileNotFoundError("stt_log.json tidak ditemukan. Jalankan --stage stt dulu.")

    with open(STT_LOG, "r", encoding="utf-8") as f:
        stt_results = json.load(f)["results"]

    # Resume: load yang sudah selesai
    log_data = {"results": {}}
    if LLM_LOG.exists():
        try:
            with open(LLM_LOG, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if "results" in loaded:
                    log_data = loaded
        except Exception:
            pass

    done    = set(log_data["results"].keys())
    total   = sum(1 for r in stt_results.values() if r["status"] == "ok")
    success = len(done)
    idx     = 0

    print(f"\n{'='*60}")
    print(f"[LLM] Total={total} | Selesai={success} | Sisa={total-success} | mode={mode}")
    print(f"{'='*60}\n")

    txt_file = open(LLM_TXT, "a", encoding="utf-8")

    for stem, entry in stt_results.items():
        if entry["status"] != "ok" or stem in done:
            if stem in done:
                print(f"[SKIP] {stem}")
            continue

        idx        += 1
        transcript  = entry["transcript"]
        print(f"[{success+1}/{total}] {entry['file']} ... ", end="")
        t0 = time.time()

        response_text = generate_response(transcript, mode=mode)
        latency       = round(time.time() - t0, 2)

        # Tulis ke TXT
        txt_file.write(f"{stem}\n")
        txt_file.write(f"USER: {transcript}\n")
        txt_file.write(f"RESPONSE: {response_text}\n\n")
        txt_file.flush()

        # Simpan ke JSON untuk kebutuhan stage TTS
        log_data["results"][stem] = {
            "file":          entry["file"],
            "utterance_id":  entry["utterance_id"],
            "response_text": response_text,
            "status":        "ok"
        }
        with open(LLM_LOG, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        success += 1
        print(f"OK | {latency}s")

        if idx % 10 == 0:
            print(f"[LLM] Jeda 5 detik...")
            time.sleep(5)

    txt_file.close()
    print(f"\n[LLM SELESAI] {success}/{total} | mode={mode} → {LLM_TXT}")

# ── STAGE 3: TTS ──────────────────────────────────────────────────────────────
def run_tts(mode: str = "normalize") -> dict:
    LLM_LOG  = get_llm_log_path(mode)
    TTS_LOG  = get_tts_log_path(mode)
    OUT_DIR  = get_output_dir(mode)

    if not LLM_LOG.exists():
        raise FileNotFoundError(
            f"{LLM_LOG} tidak ditemukan. Jalankan --stage llm --mode {mode} dulu."
        )

    with open(LLM_LOG, "r", encoding="utf-8") as f:
        llm_results = json.load(f)["results"]

    # Resume: load yang sudah selesai
    done = set()
    if TTS_LOG.exists():
        try:
            with open(TTS_LOG, "r", encoding="utf-8") as f:
                existing = json.load(f).get("results", {})
            done = {k for k, v in existing.items() if v.get("status") == "ok"}
            print(f"[TTS] Resume: {len(done)} file sudah diproses.")
        except Exception:
            pass

    results = {}
    errors  = []
    success = len(done)
    total   = sum(1 for r in llm_results.values() if r.get("status") == "ok")
    idx     = 0

    print(f"\n{'='*60}")
    print(f"[TTS] Total={total} | Selesai={success} | Sisa={total-success} | mode={mode}")
    print(f"{'='*60}\n")

    for stem, llm_entry in llm_results.items():
        if llm_entry.get("status") != "ok":
            results[stem] = {"status": "skipped", "reason": "LLM error"}
            continue

        if stem in done:
            print(f"[SKIP] {stem}")
            results[stem] = {"status": "ok", "skipped": True}
            continue

        idx     += 1
        raw_text = llm_entry["response_text"]
        print(f"[{success+1}/{total}] {llm_entry['file']}", end=" ... ")
        t0 = time.time()

        try:
            tts_path   = transcribe_text_to_speech(raw_text)
            latency    = round(time.time() - t0, 2)

            if tts_path.startswith("[ERROR]"):
                raise RuntimeError(tts_path)

            saved = OUT_DIR / f"{stem}_response.wav"
            shutil.copy(tts_path, saved)

            results[stem] = {
                "file":          llm_entry["file"],
                "utterance_id":  llm_entry["utterance_id"],
                "input_text":    raw_text,
                "output_path":   str(saved),
                "latency_s":     latency,
                "status":        "ok"
            }
            success += 1
            print(f"OK | {latency}s → {saved.name}")

        except Exception as e:
            latency = round(time.time() - t0, 2)
            results[stem] = {
                "file":          llm_entry["file"],
                "utterance_id":  llm_entry["utterance_id"],
                "input_text":    raw_text,
                "output_path":   None,
                "latency_s":     latency,
                "status":        "error",
                "error":         str(e)
            }
            errors.append(llm_entry["file"])
            print(f"ERROR: {e}")

        # Simpan setiap file — aman kalau tiba-tiba berhenti
        save_log({
            "summary": {
                "total": total, "success": success,
                "errors": len(errors), "error_files": errors, "mode": mode
            },
            "results": results
        }, TTS_LOG)

    print(f"\n[TTS SELESAI] {success}/{total} | mode={mode}")
    print(f"  Audio → {OUT_DIR}")
    print(f"  Log   → {TTS_LOG}")
    return results

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Analisis STT → LLM → TTS")
    parser.add_argument("--stage", required=True,
                        choices=["stt", "llm", "tts", "all"],
                        help="Stage yang dijalankan")
    parser.add_argument("--mode",  default="normalize",
                        choices=["preserve", "normalize"],
                        help="Mode LLM (default: normalize)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Batasi jumlah file audio")
    parser.add_argument("--file",  type=str, default=None,
                        help="Proses 1 file saja (hanya untuk --stage stt)")
    args = parser.parse_args()

    ref_scripts = load_ref_scripts()

    if args.file:
        audio_files = [Path(args.file)]
    else:
        audio_files = sorted(AUDIO_DIR.glob("*.wav"))
        if args.limit:
            audio_files = audio_files[:args.limit]

    if args.stage == "stt":
        run_stt(audio_files, ref_scripts)

    elif args.stage == "llm":
        run_llm(mode=args.mode)

    elif args.stage == "tts":
        run_tts(mode=args.mode)

    elif args.stage == "all":
        run_stt(audio_files, ref_scripts)
        run_llm(mode=args.mode)
        run_tts(mode=args.mode)