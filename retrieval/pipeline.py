from retrieval.bm25_search import BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from retrieval.compressor import compress_chunks


def retrieve(
    query: str,
    bm25_index: BM25Index,
    hybrid_top_k: int = 20,
    rerank_top_k: int = 5
) -> list[dict]:
    """
    Full retrieval pipeline for one query.

    Stage 1 — Hybrid Search  : cast wide net, get 20 candidates
    Stage 2 — Reranking      : cross-encoder scores, keep top 5
    Stage 3 — Compression    : extract only relevant sentences

    Returns compressed, reranked chunks ready to be passed to the LLM.
    This is the ONLY function the agent layer will call.
    """
    print(f"\n{'='*50}")
    print(f"RETRIEVAL PIPELINE: {query}")
    print(f"{'='*50}")

    # Stage 1
    candidates = hybrid_search(query, bm25_index, top_k=hybrid_top_k)

    # Stage 2
    reranked = rerank(query, candidates, top_k=rerank_top_k)

    # Stage 3
    final = compress_chunks(query, reranked)

    print(f"\nFinal context chunks ready: {len(final)}")
    return final


if __name__ == "__main__":
    from retrieval.bm25_search import load_bm25_index

    bm25 = load_bm25_index()
    query = "What are the legal grounds for terminating an employment contract in Pakistan?"

    results = retrieve(query, bm25)

    print("\n\n===== FINAL CONTEXT FOR LLM =====")
    for i, chunk in enumerate(results):
        print(f"\n[Chunk {i+1}] Source: {chunk['source']}, Page: {chunk['page']}")
        print(f"Rerank Score: {chunk['rerank_score']:.4f}")
        print(f"Text:\n{chunk['text']}")
        print("-" * 40)