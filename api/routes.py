from fastapi import APIRouter, Depends, HTTPException
from api.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    HealthResponse, IngestRequest, IngestResponse
)
from api.dependencies import get_bm25_index, app_state
from agent.agent import run_agent
from retrieval.bm25_search import BM25Index
import traceback

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Always expose a health endpoint.
    Used by Docker, load balancers, and monitoring tools
    to know if the server is ready to accept traffic.
    Returns 200 if healthy, implicitly 500 if it crashes.
    """
    return HealthResponse(
        status="healthy" if app_state.models_loaded else "loading",
        bm25_chunks=len(app_state.bm25_index.chunks) if app_state.bm25_index else 0,
        models_loaded=app_state.models_loaded
    )


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    bm25_index: BM25Index = Depends(get_bm25_index)
):
    """
    Main endpoint. Receives a legal question, returns an answer with sources.

    Depends(get_bm25_index) = FastAPI automatically calls get_bm25_index()
    and passes the result as the bm25_index parameter.
    This is dependency injection — the route doesn't care how the index
    was created, just that it gets one.

    Why try/except here?
    If the agent crashes internally (LLM provider down, bad retrieval, etc.)
    we return a clean 500 error to the caller — not a raw Python traceback.
    Raw tracebacks in API responses are a security and UX problem.
    """
    try:
        result = run_agent(
            user_query=request.question,
            bm25_index=bm25_index
        )

        sources = [
            SourceDocument(source=s["source"], page=s["page"])
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            iterations=result["iterations"],
            forced=result.get("forced", False)
        )

    except Exception as e:
        # Log full traceback server-side for debugging
        traceback.print_exc()
        # Return clean error to caller
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed to process query: {str(e)}"
        )


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    bm25_index: BM25Index = Depends(get_bm25_index)
):
    """
    Triggers re-ingestion of PDFs from the specified directory.
    After ingestion, rebuilds the BM25 index in memory.

    Why expose this as an endpoint?
    In production, you'd add new legal documents without restarting
    the server. This endpoint handles that — upload PDFs, call /ingest,
    system is updated.

    Note: Vector index (ChromaDB) re-indexing is included.
    This can take 30-120 seconds depending on document count.
    In production you'd make this async — for now synchronous is fine.
    """
    try:
        from ingestion.loader import load_all_pdfs
        from ingestion.chunker import chunk_pages
        from ingestion.embedder import embed_chunks
        from ingestion.indexer import index_chunks
        import json

        print(f"[INGEST] Loading PDFs from {request.pdf_dir}...")
        pages = load_all_pdfs(request.pdf_dir)
        chunks = chunk_pages(pages)

        print("[INGEST] Embedding chunks...")
        chunks = embed_chunks(chunks)

        # Save chunks for BM25
        with open("data/chunks.json", "w", encoding="utf-8") as f:
            json.dump(
                [{k: v for k, v in c.items() if k != "embedding"} for c in chunks],
                f, ensure_ascii=False
            )

        print("[INGEST] Indexing into ChromaDB...")
        index_chunks(chunks)

        # Rebuild BM25 index in memory with new chunks
        print("[INGEST] Rebuilding BM25 index...")
        from retrieval.bm25_search import BM25Index as BM25
        app_state.bm25_index = BM25(
            [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
        )

        return IngestResponse(
            message="Ingestion complete. BM25 and vector index updated.",
            chunks_created=len(chunks)
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")