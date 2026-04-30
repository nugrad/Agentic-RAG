# from google import genai
# from google.genai import types
import os
from dotenv import load_dotenv
from agent.llm import get_gemini_response

# load_dotenv()

# # Load API key
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in environment variables!")

# # Initialize the Gemini client once
# client = genai.Client(api_key=GEMINI_API_KEY)
def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Old approach: 1 LLM call per chunk = N calls total. Kills rate limits.

    New approach: 1 LLM call for ALL chunks together.
    We pass all chunks in a numbered list, LLM extracts relevant
    sentences per chunk in one shot, we parse the response.

    Trade-off: slightly more complex prompt/parsing.
    Benefit: 5x-10x fewer LLM calls. Night and day on free tier.
    """
    if not chunks:
        return []

    # Build a single prompt with all chunks numbered
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n[CHUNK {i+1}]\n{chunk['text']}\n"

    prompt = f"""You are a legal document analyst.

Given the QUERY and multiple PASSAGES below, extract ONLY the sentences 
from each passage that are directly relevant to answering the query.

Rules:
- For each chunk, output its number and extracted sentences
- Copy sentences verbatim — do not paraphrase
- If a chunk has NO relevant sentences, write: [CHUNK N]: NOT RELEVANT
- Do not add explanations or commentary
- Follow the exact output format shown

QUERY: {query}

PASSAGES:
{chunks_text}

OUTPUT FORMAT (follow exactly):
[CHUNK 1]: <extracted sentences or NOT RELEVANT>
[CHUNK 2]: <extracted sentences or NOT RELEVANT>
...and so on for each chunk

Output:"""

    try:
        raw = get_gemini_response(prompt, temperature=0)
        return _parse_batch_response(raw, chunks)

    except Exception as e:
        # If batch compression fails entirely, return chunks uncompressed
        # Better to have uncompressed chunks than no context at all
        print(f"  [COMPRESSOR] Batch compression failed: {e}. Using raw chunks.")
        for chunk in chunks:
            chunk["compressed"] = False
        return chunks


def _parse_batch_response(raw: str, chunks: list[dict]) -> list[dict]:
    """
    Parses the batched LLM response back into individual chunk dicts.

    Expected format from LLM:
    [CHUNK 1]: Some relevant sentence from chunk 1.
    [CHUNK 2]: NOT RELEVANT
    [CHUNK 3]: Another relevant sentence.

    We match by chunk number, skip NOT RELEVANT ones,
    and attach compressed text back to the original chunk dict.
    """
    import re

    compressed = []

    for i, chunk in enumerate(chunks):
        # Look for [CHUNK N]: content in the response
        pattern = rf"\[CHUNK {i+1}\]:\s*(.+?)(?=\[CHUNK \d+\]|$)"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)

        if not match:
            # LLM didn't output this chunk number — keep original
            chunk["compressed"] = False
            compressed.append(chunk)
            continue

        extracted = match.group(1).strip()

        if "NOT RELEVANT" in extracted.upper() or len(extracted) < 20:
            # Genuinely irrelevant — drop this chunk
            continue

        compressed_chunk = chunk.copy()
        compressed_chunk["original_text"] = chunk["text"]
        compressed_chunk["text"] = extracted
        compressed_chunk["compressed"] = True
        compressed.append(compressed_chunk)

    print(f"  ✅ Compression: {len(chunks)} chunks → {len(compressed)} kept")
    return compressed


