from google import genai
from google.genai import types
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

GROQ_MODEL = "llama-3.3-70b-versatile"


def _call_groq(prompt: str, temperature: float) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def get_gemini_response(prompt: str, temperature: float = 0.0) -> str:
    """
    Revised fallback logic:

    503 UNAVAILABLE  → try next Gemini model (brief wait, it's a server spike)
    429 RATE LIMITED → immediately jump to Groq, do NOT wait 50 seconds
                       Groq is fast and free — use it as a real peer, not last resort

    Why this change?
    Waiting 50s per 429 across 8-10 calls = 400-500s per query.
    Groq responds in under 2 seconds. The old strategy was burning minutes
    waiting for Gemini when a perfectly good provider was sitting idle.
    """
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=2000,
                )
            )
            return response.text.strip() if response.text else ""

        except Exception as e:
            error_str = str(e)

            if "503" in error_str or "UNAVAILABLE" in error_str:
                # Server spike — worth waiting briefly and trying next model
                print(f"  [LLM] {model_name} unavailable (503), trying next...")
                time.sleep(2)
                continue

            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Rate limited — do NOT wait, go to Groq immediately
                print(f"  [LLM] {model_name} rate limited (429) → switching to Groq now")
                break  # Exit Gemini loop entirely, go straight to Groq

            else:
                raise  # Real error — crash loudly

    print(f"  [LLM] Using Groq ({GROQ_MODEL})")
    try:
        return _call_groq(prompt, temperature)
    except Exception as groq_error:
        raise RuntimeError(f"All providers failed. Groq error: {groq_error}")