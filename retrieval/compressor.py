from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

# Initialize the Gemini client once
client = genai.Client(api_key=GEMINI_API_KEY)


def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    For each chunk, asks Gemini to extract ONLY the sentences
    relevant to the query. Everything else is discarded.

    Why one call per chunk?
    - Better traceability
    - Avoids context confusion
    - Failures are isolated
    """

    compressed = []

    for chunk in chunks:
        prompt = f"""You are a precise legal document analyst.

Given the QUERY and the PASSAGE below, extract ONLY the sentences from the 
passage that are directly relevant to answering the query.

Rules:
- Copy sentences verbatim. Do not paraphrase, summarize, or modify them.
- If multiple sentences are relevant, include all of them in order.
- If NO sentences are relevant, respond with exactly: NOT RELEVANT
- Do not add any explanations, headers, bullet points, or commentary.

QUERY: {query}

PASSAGE:
{chunk['text']}

Relevant sentences:"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",   # You can also use "gemini-2.5-flash" if available
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,        # Deterministic output
                    max_output_tokens=1000,
                )
            )

            extracted = response.text.strip() if response.text else ""

            # Filter out irrelevant chunks
            if extracted == "NOT RELEVANT" or len(extracted) < 15:
                continue

            # Create compressed chunk
            compressed_chunk = chunk.copy()
            compressed_chunk["original_text"] = chunk["text"]
            compressed_chunk["text"] = extracted
            compressed_chunk["compressed"] = True
            compressed.append(compressed_chunk)

        except Exception as e:
            print(f"⚠️ Compression failed for chunk {chunk.get('chunk_id', 'unknown')}: {e}")
            # Fallback: keep original chunk if compression fails
            chunk_copy = chunk.copy()
            chunk_copy["compressed"] = False
            compressed.append(chunk_copy)

    print(f"✅ Compression: {len(chunks)} chunks → {len(compressed)} compressed chunks")
    return compressed


# ============================
# Test Code
# ============================
if __name__ == "__main__":
    dummy_chunks = [
        {
            "chunk_id": "1",
            "text": """Section 4 deals with termination of employment. An employer shall 
provide thirty days written notice before terminating a permanent employee. 
The notice shall state the reason for termination clearly. Additionally, 
all employees are entitled to fourteen days of annual leave per calendar year. 
Leave encashment shall be calculated at the basic salary rate.""",
            "source": "employment_act.pdf",
            "page": 4,
            "chunk_index": 10,
            "rerank_score": 0.92
        },
        {
            "chunk_id": "2",
            "text": """This document is confidential. Unauthorized distribution is prohibited.
Employees must maintain secrecy of company trade secrets at all times.""",
            "source": "employment_act.pdf",
            "page": 5,
            "chunk_index": 15,
            "rerank_score": 0.45
        }
    ]

    query = "How many days notice must an employer give before terminating an employee?"

    print("Running compression test...\n")
    result = compress_chunks(query, dummy_chunks)

    print("\n--- Compressed Output ---")
    for r in result:
        print(f"Chunk ID       : {r.get('chunk_id')}")
        print(f"Original length: {len(r.get('original_text', r['text']))} chars")
        print(f"Compressed len : {len(r['text'])} chars")
        print(f"Compressed     : {r['text'][:300]}..." if len(r['text']) > 300 else f"Compressed     : {r['text']}")
        print("-" * 80)