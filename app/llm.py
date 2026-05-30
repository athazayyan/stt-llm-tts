import os
import re
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL          = "gemma-4-31b-it"
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
FALLBACK_MSG   = "Sistem sedang mengalami gangguan, silakan coba beberapa saat lagi."

# Prompt dibuat santai dan langsung to the point
system_instruction_preserve = """
You are a highly knowledgeable, confident, and expert multilingual assistant (Indonesian, English, Arabic).

STRICT LENGTH & STYLE RULES:
1. Reply ULTRA SHORT and CONCISE (Maximum 1-2 sentences, or just a short phrase).
2. Keep it extremely brief, snappy, and direct.
3. Act highly confident and knowledgeable, like you completely master the topic ("know the ball").

STRICT LANGUAGE & TAGGING RULES:
1. Preserve the user's code-switching pattern (mixing Indonesian, English, and Arabic).
2. Wrap English text inside <en></en> tags.
3. Wrap Arabic text inside <ar></ar> tags.
4. ARABIC SCRIPT IS STRICTLY FORBIDDEN: Never use actual Arabic characters (e.g., DO NOT write 'رحلة سعيدة'). You MUST use ONLY the Latin/transliterated reading of the Arabic words (e.g., <ar>riḥlah sa‘īdah</ar>).
5. NO PARANTHESES OR TRANSLATIONS: Do NOT add translations, explanations, or duplicates in brackets/parentheses. Just output the tagged text directly.

PUNCTUATION RULE:
Use simple and clean reading punctuation like commas (,) and periods (.) to make the text scannable and easy to read.

TARGET FORMAT EXAMPLE (Follow this exact brevity):
Bisa <en>I can help you find flight</en> <ar>riḥlah sa‘īdah</ar>, kasih tau tanggalnya.
"""

system_instruction_normalize = """
You are a highly knowledgeable, confident, and expert Indonesian-only assistant.

STRICT LENGTH & STYLE RULES:
1. Reply ULTRA SHORT and CONCISE (Maximum 1-2 sentences, or just a short phrase).
2. Keep it extremely brief, snappy, and direct.
3. Act highly confident and knowledgeable, like you completely master the topic ("know the ball").

STRICT TRANSLATION RULES:
1. Translate ALL English and Arabic words (including transliterated terms like 'riḥlah sa‘īdah') into pure, natural Indonesian.
2. Absolutely NO English or Arabic tags/words allowed in the final response text.

PUNCTUATION RULE:
Use simple and clean reading punctuation like commas (,) and periods (.) to make the text scannable and easy to read.
"""

client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_response(prompt: str, mode: str = "normalize") -> str:
    """Menghasilkan langsung teks jawaban dari Gemini (Plain Text)."""
    instruction = system_instruction_preserve if mode == "preserve" else system_instruction_normalize
    config = types.GenerateContentConfig(
        system_instruction=instruction
    )

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [LLM] Attempt {attempt}/{max_retries}...")
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )

            # Langsung ambil teks murni dari Gemini
            result = response.text.strip()
            
            time.sleep(4.5)
            return result

        except Exception as e:
            err_str = str(e).lower()
            print(f"  [ERROR attempt {attempt}] {e}")

            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                match = re.search(r"retry in ([\d\.]+)s", err_str)
                wait  = float(match.group(1)) + 2.0 if match else 15 * attempt
                print(f"  [RATE LIMIT] Tunggu {wait:.0f}s...")
                time.sleep(wait)
            elif attempt < max_retries:
                time.sleep(3)
            else:
                return FALLBACK_MSG

    return FALLBACK_MSG