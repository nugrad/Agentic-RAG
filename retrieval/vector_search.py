import chromadb
from ingestion.embedder import embed_query
from ingestion.indexer import CHROMA_DIR, COLLECTION_NAME


def get_collection():
    """
    Returns the ChromaDB collection.
    Called at search time — ChromaDB handles connection pooling internally.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def vector_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Embeds the query using BGE (with instruction prefix).
    Queries ChromaDB for top_k nearest chunks.
    Returns list of chunk dicts with vector_rank attached.
    """
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for rank, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        chunks.append({
            "chunk_id": results["ids"][0][rank],
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "chunk_index": meta["chunk_index"],
            "vector_distance": float(dist),
            "vector_rank": rank,
        })

    return chunks


if __name__ == "__main__":
    results = vector_search("termination of employment notice period", top_k=5)

    print("\n--- Vector Top 5 ---")
    for r in results:
        print(f"\nRank {r['vector_rank'] + 1} | Distance: {r['vector_distance']:.4f}")
        print(f"Source: {r['source']}, Page: {r['page']}")
        print(f"Text: {r['text'][:200]}...")