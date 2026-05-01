from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import router
from api.dependencies import load_all_models
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan = code that runs at startup and shutdown.

    Everything before 'yield' runs when server starts.
    Everything after 'yield' runs when server shuts down.

    Why lifespan instead of @app.on_event("startup")?
    on_event is deprecated in newer FastAPI. Lifespan is the
    current recommended pattern. Use what the framework recommends.

    Models load here — once — before any request is served.
    """
    print("[STARTUP] Agentic RAG server starting...")
    load_all_models()
    print("[STARTUP] Server ready.\n")

    yield  # Server runs here, handling requests

    # Shutdown cleanup (if needed in future — close DB connections etc.)
    print("[SHUTDOWN] Server shutting down.")


app = FastAPI(
    title="Agentic RAG — Pakistan Employment & Contract Law",
    description=(
        "An agentic retrieval-augmented generation system for querying "
        "Pakistani employment and contract law documents. "
        "Uses BGE embeddings, hybrid search, cross-encoder reranking, "
        "and a ReAct agent loop."
    ),
    version="1.0.0",
    lifespan=lifespan
)
app = FastAPI(
    title="PAKEmployee GPT",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve frontend folder as static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# # Root hits the HTML file directly
@app.get("/")
def root():
    return FileResponse("frontend/index.html")

app.include_router(router, prefix="/api/v1")


# Root redirect so hitting / gives something useful
# @app.get("/")
# def root():
#     return {
#         "message": "Agentic RAG API is running",
#         "docs": "/docs",
#         "health": "/api/v1/health"
#     }