from dotenv import load_dotenv
import os

load_dotenv()

# === NEW Google GenAI SDK (recommended) ===
from google import genai
from google.genai import types

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

# Test Gemini
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",   # or gemini-2.5-flash-lite
        contents="Say hello in one word"
    )
    print("Gemini:", response.text)
except Exception as e:
    print("Gemini Error:", e)
    
# === Groq (this part is fine) ===
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    r2 = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        max_tokens=10
    )
    print("Groq:", r2.choices[0].message.content)
except Exception as e:
    print("Groq Error:", e)