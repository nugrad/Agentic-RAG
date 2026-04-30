"""
Why a separate dependencies file?

Your BM25 index takes ~1 second to build.
Your embedding model takes ~3 seconds to load.
Your cross-encoder takes ~2 seconds to load.

If you loaded these inside the route function:
- Every request waits 6 seconds just for model loading
- You'd be loading a 90MB model thousands of times

FastAPI's dependency injection system lets you load once
at startup and inject the same object into every route.
This is standard production practice — not optional.
"""

from retrieval.bm25_search import BM25Index, load_bm25_index
from ingestion.embedder import get_model as get_embedding_model
from retrieval.reranker import get_reranker


class AppState:
    """
    Holds all shared resources that should live for the
    duration of the server process.

    We use a simple class instead of global variables
    because it's easier to test and reason about.
    """
    bm25_index: BM25Index = None
    models_loaded: bool = False


# Single instance — shared across all requests
app_state = AppState()


def get_bm25_index() -> BM25Index:
    """
    FastAPI dependency — injected into routes that need it.
    Returns the already-loaded BM25 index.
    Raises if called before startup completed (shouldn't happen).
    """
    if app_state.bm25_index is None:
        raise RuntimeError("BM25 index not loaded. Server not ready.")
    return app_state.bm25_index


def load_all_models():
    """
    Called once at server startup.
    Pre-loads all models into memory so first request isn't slow.
    Order matters: BM25 first (fast), then neural models (slow).
    """
    print("[STARTUP] Building BM25 index...")
    app_state.bm25_index = load_bm25_index("data/chunks.json")

    print("[STARTUP] Loading BGE embedding model...")
    get_embedding_model()  # Triggers lazy load, caches in module

    print("[STARTUP] Loading cross-encoder reranker...")
    get_reranker()  # Triggers lazy load, caches in module

    app_state.models_loaded = True
    print("[STARTUP] All models ready. Server accepting requests.")