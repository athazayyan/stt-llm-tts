import os
import re
import json
import uuid
import tempfile

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
COQUI_DIR         = os.path.join(BASE_DIR, "coqui_utils")
COQUI_MODEL_PATH  = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")
COQUI_SPEAKER     = "wibowo"
EN_TRANS_FILE     = os.path.join(BASE_DIR, "en_transliteration.json")

with open(EN_TRANS_FILE, "r", encoding="utf-8") as f:
    EN_TRANS = json.load(f)

_tts_instance = None

def _get_tts():
    global _tts_instance
    if _tts_instance is None:
        original_dir = os.getcwd()
        os.chdir(COQUI_DIR)
        try:
            from TTS.api import TTS
            print("[TTS] Loading model...")
            _tts_instance = TTS(
                model_path=COQUI_MODEL_PATH,
                config_path=COQUI_CONFIG_PATH,
                progress_bar=False
            )
            print("[TTS] Model loaded!")
        finally:
            os.chdir(original_dir)
    return _tts_instance

# Transliterasi karakter Arab berdiakritik → ASCII
AR_DIACRITIC_MAP = {
    'ā': 'a', 'ī': 'i', 'ū': 'u',
    'ḥ': 'h', 'ṭ': 't', 'ṣ': 's',
    'ḍ': 'd', 'ẓ': 'z', 'ʿ': '',
     '\u2019': '',
    'á': 'a', 'é': 'e', 'í': 'i',
    'ó': 'o', 'ú': 'u', 'ñ': 'n',
    'ḏ': 'd', 'ġ': 'g', 'ḫ': 'h',
    'ṯ': 't', 'ḳ': 'k', 'ẗ': 't',
    'g': 'k',
}

def normalize_diacritics(text: str) -> str:
    for char, repl in AR_DIACRITIC_MAP.items():
        text = text.replace(char, repl)
    return text

def extract_segments(text: str) -> list:
    """
    Pisah teks jadi segmen berdasarkan tag bahasa.
    Return list of (lang, text) tuples.
    Contoh: "Bisa, <en>take the train</en>, bagus"
    → [('id','Bisa,'), ('en','take the train'), ('id',', bagus')]
    """
    pattern = r'(<(?:en|ar)>.*?</(?:en|ar)>)'
    parts   = re.split(pattern, text)
    segments = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r'<(en|ar)>(.*?)</(?:en|ar)>', part, re.DOTALL)
        if m:
            segments.append((m.group(1), m.group(2).strip()))
        else:
            segments.append(('id', part))
    return segments

def clean_segment(text: str) -> str:
    """Bersihkan satu segmen teks untuk TTS."""
    # Transliterasi diakritik Arab
    text = normalize_diacritics(text)
    text = text.lower()

    # Transliterasi kata Inggris dari kamus
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            bigram = words[i] + " " + words[i+1]
            if bigram in EN_TRANS:
                result.append(EN_TRANS[bigram])
                i += 2
                continue
        w = re.sub(r"[^\w'-]", "", words[i])
        if w in EN_TRANS:
            mapped = EN_TRANS[w]
            if mapped:
                result.append(mapped)
        else:
            result.append(words[i])
        i += 1
    text = " ".join(result)

    # Ganti karakter tidak di vocab
    char_map = {'c': 'ts', 'q': 'k', 'v': 'f', 'x': 'ks'}
    for char, repl in char_map.items():
        text = text.replace(char, repl)

    # Allowed chars — vocab model + tanda baca
    allowed = set("abdefghijklmnoprstuwyzŋɔəɛɡɪɲʃʊʒʔˈ ,!.?'-")
    text = ''.join(c if c in allowed else ' ' for c in text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_tts_text(text: str) -> str:
    """
    Proses teks dengan segmentasi per bahasa,
    gabungkan kembali dengan jeda koma antar segmen.
    """
    segments = extract_segments(text)
    cleaned  = []
    for lang, seg in segments:
        seg_clean = clean_segment(seg)
        if seg_clean:
            cleaned.append(seg_clean)

    # Gabung dengan koma sebagai jeda natural antar segmen
    result = ", ".join(cleaned)
    # Bersihkan koma dobel
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def transcribe_text_to_speech(text: str) -> str:
    tmp_dir     = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    try:
        clean = clean_tts_text(text)
        print(f"  [TTS] Input bersih: {clean[:120]}")
        tts = _get_tts()
        tts.tts_to_file(
            text=clean,
            speaker=COQUI_SPEAKER,
            file_path=output_path
        )
        return output_path
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return "[ERROR] Failed to synthesize speech"