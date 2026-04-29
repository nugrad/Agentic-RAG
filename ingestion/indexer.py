import chromadb
from chromadb.config import Settings
import json
import os
from pathlib import Path

# Where ChromaDB persists its data on disk
# Delete this folder to start fresh / re-index
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "pakistan_law"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    PersistentClient = saves index to disk at CHROMA_DIR.
    Survives restarts. Not EphemeralClient (in-memory only).
    """
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Gets existing collection or creates it fresh.
    
    cosine distance = standard for normalized BGE embeddings.
    'l2' (euclidean) would give wrong results with normalized vectors.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def index_chunks(chunks: list[dict]) -> None:
    """
    Stores all chunks in ChromaDB.

    ChromaDB expects four parallel lists:
    - ids         : unique string per chunk
    - embeddings  : the vector
    - documents   : raw text (ChromaDB stores this for you)
    - metadatas   : dict of filterable fields

    Why not store embedding inside metadata?
    ChromaDB separates vectors from metadata by design —
    vectors go into the HNSW index, metadata stays in a flat store.
    They're linked by ID internally.

    Why batch upsert?
    Upserting 1000 chunks one by one = 1000 DB writes.
    Batching = 1 write per batch. Much faster.
    upsert = insert if not exists, update if exists.
    Safe to re-run without duplicating data.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    BATCH_SIZE = 100
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start: start + BATCH_SIZE]

        ids = [chunk["chunk_id"] for chunk in batch]
        embeddings = [chunk["embedding"] for chunk in batch]
        documents = [chunk["text"] for chunk in batch]
        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"Indexed {min(start + BATCH_SIZE, total)}/{total} chunks")

    print(f"\nCollection '{COLLECTION_NAME}' now has "
          f"{collection.count()} chunks stored.")


def verify_index(query_embedding: list[float], n_results: int = 3) -> None:
    """
    Sanity check — query the index with a real embedding.
    Call this after indexing to confirm retrieval works.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    print("\n--- Retrieval Verification ---")
    for i in range(len(results["documents"][0])):
        print(f"\nResult {i+1}")
        print(f"  Source  : {results['metadatas'][0][i]['source']}")
        print(f"  Page    : {results['metadatas'][0][i]['page']}")
        print(f"  Distance: {results['distances'][0][i]:.4f}")
        print(f"  Text    : {results['documents'][0][i][:200]}...")


if __name__ == "__main__":
    # Load already-embedded chunks and index them
    CHUNKS_FILE = "data/chunks_embedded.json"

    if not os.path.exists(CHUNKS_FILE):
        print(f"ERROR: {CHUNKS_FILE} not found. Run run_ingestion.py first.")
        exit(1)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index_chunks(chunks)