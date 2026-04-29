from sentence_transformers import SentenceTransformer
import numpy as np

# BGE base — good balance of quality and speed
# First run downloads ~90MB model, cached after that
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Load once at module level — expensive to reload repeatedly
# This stays in memory for the duration of your script
_model = None


def get_model() -> SentenceTransformer:
    """
    Lazy loader — only loads model when first called.
    Prevents loading 90MB model if you import this file for other reasons.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")
    return _model


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    """
    Takes your list of chunk dicts (from chunker.py output).
    Adds an 'embedding' key to each dict.
    Returns the same list — now enriched with vectors.

    Why batch_size=32?
    Embedding one chunk at a time is slow.
    Embedding all at once may OOM on large datasets.
    32 is safe for CPU and most laptops with 8GB RAM.

    BGE Query vs Passage distinction:
    - When embedding CHUNKS (passages) → no instruction prefix needed
    - When embedding a USER QUERY at search time → add the prefix
    We handle the query prefix separately in retrieval. Here we embed passages.
    """
    model = get_model()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")

    # encode() handles batching internally when you pass batch_size
    # show_progress_bar gives you a live progress indicator
    # normalize_embeddings=True → vectors scaled to unit length
    # Required for cosine similarity to work correctly with BGE
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Attach embedding back to each chunk dict
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()  # list for JSON compatibility

    print(f"Done. Each vector has {len(chunks[0]['embedding'])} dimensions.")
    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embeds a single user query at search time.

    BGE requires a specific instruction prefix for queries — not for passages.
    This is documented in the BGE model card and measurably improves retrieval.
    Skipping this prefix degrades search quality.
    """
    model = get_model()
    instruction = "Represent this sentence for searching relevant passages: "
    
    embedding = model.encode(
        instruction + query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embedding.tolist()


if __name__ == "__main__":
    # Quick test — embed 2 law-domain sentences, check similarity
    model = get_model()

    s1 = "An employer may terminate employment with thirty days notice."
    s2 = "Employment can be ended by giving one month advance notice."
    s3 = "The contract shall be governed by the laws of Pakistan."

    vecs = model.encode([s1, s2, s3], normalize_embeddings=True)

    # Cosine similarity = dot product of normalized vectors
    sim_12 = np.dot(vecs[0], vecs[1])
    sim_13 = np.dot(vecs[0], vecs[2])

    print(f"Similarity (s1 vs s2 — same meaning): {sim_12:.4f}")
    print(f"Similarity (s1 vs s3 — different topic): {sim_13:.4f}")
    # s1 vs s2 should be significantly higher than s1 vs s3
    # If not — something is wrong with your embeddings