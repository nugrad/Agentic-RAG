from retrieval.bm25_search import BM25Index
from retrieval.vector_search import vector_search


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion.

    k=60 is the standard constant — chosen empirically in the original RRF paper.
    Lower k = top ranks matter more (aggressive).
    Higher k = ranks matter less, more uniform blending.
    60 is safe. Don't change it without a reason.

    We use chunk_id as the unique key to merge results across both lists.
    A chunk may appear in vector results but not BM25 — that's fine.
    Its RRF score will only have the vector component.
    """
    rrf_scores = {}  # chunk_id → {score, chunk_data}

    # Process vector results
    for rank, chunk in enumerate(vector_results):
        cid = chunk["chunk_id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = {"score": 0.0, "chunk": chunk}
        rrf_scores[cid]["score"] += 1.0 / (k + rank + 1)

    # Process BM25 results
    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = {"score": 0.0, "chunk": chunk}
        rrf_scores[cid]["score"] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Return enriched chunk dicts with rrf_score attached
    final = []
    for item in sorted_results:
        chunk = item["chunk"].copy()
        chunk["rrf_score"] = item["score"]
        final.append(chunk)

    return final


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    top_k: int = 20
) -> list[dict]:
    """
    Runs both vector and BM25 search, merges with RRF.
    Returns top_k results from the merged list.

    top_k=20 here because reranker will cut this down to 5.
    We cast a wide net at this stage — recall matters more than precision.
    Precision comes from the reranker.
    """
    print(f"Running hybrid search for: '{query}'")

    vec_results = vector_search(query, top_k=top_k)
    bm25_results = bm25_index.search(query, top_k=top_k)

    print(f"  Vector results: {len(vec_results)}")
    print(f"  BM25 results:   {len(bm25_results)}")

    merged = reciprocal_rank_fusion(vec_results, bm25_results)
    top = merged[:top_k]

    print(f"  After RRF merge: {len(top)} unique chunks")
    return top


if __name__ == "__main__":
    from retrieval.bm25_search import load_bm25_index

    bm25 = load_bm25_index()
    results = hybrid_search(
        "what is the notice period for termination of employment",
        bm25_index=bm25,
        top_k=10
    )

    print("\n--- Hybrid Top 10 ---")
    for i, r in enumerate(results):
        print(f"\nRank {i+1} | RRF: {r['rrf_score']:.6f}")
        print(f"Source: {r['source']}, Page: {r['page']}")
        print(f"Text: {r['text'][:200]}...")