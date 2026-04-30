from sentence_transformers import CrossEncoder

# Free, runs locally, ~80MB download
# Trained on MS MARCO — large passage ranking dataset
# Good enough for legal domain out of the box
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker() -> CrossEncoder:
    """
    Lazy load — same pattern as embedder.py.
    CrossEncoder is a different class from SentenceTransformer.
    It takes (query, passage) pairs — not individual texts.
    """
    global _reranker
    if _reranker is None:
        print(f"Loading cross-encoder: {MODEL_NAME}")
        _reranker = CrossEncoder(MODEL_NAME)
        print("Cross-encoder loaded.")
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Takes hybrid search results (20 chunks).
    Scores each (query, chunk) pair with cross-encoder.
    Returns top_k chunks sorted by cross-encoder score.

    Why top_k=5?
    LLMs work best with focused, dense context.
    5 chunks × ~400 chars ≈ 2000 chars — clean, manageable context.
    More than 5 and you risk diluting the signal.

    The cross-encoder score is NOT a probability (not 0-1).
    It's a raw logit — can be negative. Higher = more relevant.
    Only use it for ranking, not as an absolute quality measure.
    """
    reranker = get_reranker()

    # Build (query, chunk_text) pairs — this is what cross-encoder expects
    pairs = [(query, chunk["text"]) for chunk in chunks]

    scores = reranker.predict(pairs)

    # Attach score to each chunk
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # Sort by rerank score descending, return top_k
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    top = reranked[:top_k]

    print(f"Reranked {len(chunks)} → kept top {len(top)}")
    return top


if __name__ == "__main__":
    # Simulate what reranker receives from hybrid search
    dummy_chunks = [
        {"chunk_id": "1", "text": "An employer shall give thirty days notice before termination.", "source": "act.pdf", "page": 4, "chunk_index": 10},
        {"chunk_id": "2", "text": "Employees are entitled to annual leave of fourteen days.", "source": "act.pdf", "page": 7, "chunk_index": 22},
        {"chunk_id": "3", "text": "Termination without cause requires compensation equivalent to one month salary.", "source": "act.pdf", "page": 5, "chunk_index": 15},
        {"chunk_id": "4", "text": "The contract shall be governed under the laws of Pakistan.", "source": "contract.pdf", "page": 1, "chunk_index": 3},
        {"chunk_id": "5", "text": "Summary dismissal is permitted in cases of gross misconduct as defined in Section 14.", "source": "act.pdf", "page": 9, "chunk_index": 30},
    ]

    query = "What notice period is required before terminating an employee?"
    results = rerank(query, dummy_chunks, top_k=3)

    print("\n--- Reranked Top 3 ---")
    for i, r in enumerate(results):
        print(f"\nRank {i+1} | Score: {r['rerank_score']:.4f}")
        print(f"Text: {r['text']}")