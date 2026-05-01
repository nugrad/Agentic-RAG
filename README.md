# PAKEmployee GPT 🇵🇰⚖️

> An end-to-end **Agentic RAG system** for querying Pakistani Employment and Contract Law documents — built with hybrid retrieval, cross-encoder reranking, a ReAct agent loop, and evaluated with RAGAS.

---

## Overview

PAKEmployee GPT is a production-grade legal research assistant that lets users ask natural language questions about Pakistani employment and contract law. Rather than a simple keyword search or a basic RAG pipeline, it uses an **agentic retrieval loop** — the system can reason about whether its retrieved context is sufficient, refine its search queries, and iterate before producing a cited answer.

**Domain:** Pakistani Employment Law, Contract Law, Labour Regulations  
**Data:** Sindh Terms of Employment Act, Commercial Employment Ordinance, Pakistan Labour Law, and related statutes

---

## Architecture

```
PDF Documents
      ↓
Context-Aware Chunking          (semantic boundaries, 500 chars, 100 overlap)
      ↓
BGE Embeddings                  (BAAI/bge-base-en-v1.5, 768-dim, normalized)
      ↓
Dual Indexing
  ├── ChromaDB Vector Index     (cosine similarity, persistent)
  └── BM25 Index                (keyword search, in-memory)
      ↓
User Query
      ↓
Hybrid Retrieval                (Vector + BM25 merged via Reciprocal Rank Fusion)
      ↓
Cross-Encoder Reranking         (ms-marco-MiniLM-L-6-v2, top 5 kept)
      ↓
Context Compression             (LLM extracts relevant sentences only)
      ↓
ReAct Agent Loop                (Reason → Act → Observe → Repeat, max 3 iterations)
  ├── Tool: retrieve(query)
  ├── Tool: refine_query(original, reason)
  └── Tool: final_answer(answer, sources)
      ↓
LLM Generation                  (Gemini 2.0 Flash → Groq llama-3.3-70b fallback)
      ↓
RAGAS Evaluation                (Faithfulness, Answer Relevancy, Context Precision, Recall)
      ↓
FastAPI + Streamlit Frontend
```

---

## RAGAS Evaluation Results

Evaluated on 15 hand-crafted question-answer pairs from the actual legal documents.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Faithfulness** | **0.93** | Answers grounded in retrieved context — minimal hallucination |
| **Answer Relevancy** | **0.83** | Answers directly address what was asked |
| **Context Precision** | 0.50 | Retrieval includes some noise — expected in dense legal text |
| **Context Recall** | 0.67 | Most required information is retrieved |

> **Judge LLM:** Groq `llama-3.3-70b-versatile` · **Embeddings:** BGE `bge-base-en-v1.5` (local)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `BAAI/bge-base-en-v1.5` via sentence-transformers |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Keyword Search | rank-bm25 (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (primary) | Google Gemini 2.0 Flash |
| LLM (fallback) | Groq llama-3.3-70b-versatile |
| PDF Parsing | PyMuPDF (fitz) |
| Evaluation | RAGAS + Groq judge + BGE embeddings |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit / HTML+JS |
| Containerization | Docker |

---

## Project Structure

```
agentic-rag/
│
├── data/
│   ├── pdfs/                   # Raw Pakistani law PDFs
│   ├── chunks.json             # Processed chunks (no embeddings)
│   ├── chunks_embedded.json    # Chunks with BGE vectors
│   └── chroma_db/              # Persisted ChromaDB index
│
├── ingestion/
│   ├── loader.py               # PDF text extraction (PyMuPDF)
│   ├── chunker.py              # Context-aware chunking + cleaning
│   ├── embedder.py             # BGE embedding (batch, normalized)
│   ├── indexer.py              # ChromaDB upsert with metadata
│   └── run_ingestion.py        # Full ingestion orchestrator
│
├── retrieval/
│   ├── vector_search.py        # ChromaDB semantic search
│   ├── bm25_search.py          # BM25 keyword search
│   ├── hybrid.py               # RRF merger
│   ├── reranker.py             # Cross-encoder scoring
│   ├── compressor.py           # LLM-based context compression
│   └── pipeline.py             # Single retrieve() entry point
│
├── agent/
│   ├── llm.py                  # Gemini + Groq fallback chain
│   ├── tools.py                # retrieve, refine_query, final_answer
│   ├── prompts.py              # ReAct system prompt + history builder
│   └── agent.py                # ReAct loop, parser, force-answer
│
├── api/
│   ├── main.py                 # FastAPI app + lifespan startup
│   ├── routes.py               # /health, /query, /ingest endpoints
│   ├── schemas.py              # Pydantic request/response models
│   └── dependencies.py        # Shared model loading (load once)
│
├── evaluation/
│   ├── test_dataset.json       # 15 hand-crafted QA pairs
│   └── evaluator.py            # RAGAS pipeline + scoring
│
├── frontend/
│   └── index.html              # Neon chatbot UI (served by FastAPI)
│
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Running

### Prerequisites
- Python 3.11+
- Docker (optional)
- Gemini API key — [aistudio.google.com](https://aistudio.google.com)
- Groq API key — [console.groq.com](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### 3. Add your PDFs

Place Pakistani law PDF documents in `data/pdfs/`

### 4. Run ingestion

```bash
python -m ingestion.run_ingestion
```

This loads PDFs → chunks → embeds with BGE → indexes in ChromaDB + BM25. Takes 2-5 minutes depending on document count.

### 5. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`  
Frontend at `http://localhost:8000/`

### 6. (Optional) Run Streamlit frontend

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`

---

## API Endpoints

### `GET /api/v1/health`
```json
{
  "status": "healthy",
  "bm25_chunks": 2124,
  "models_loaded": true
}
```

### `POST /api/v1/query`

**Request:**
```json
{
  "question": "What is the notice period for terminating a permanent employee in Pakistan?",
  "max_iterations": 3
}
```

**Response:**
```json
{
  "answer": "An employer must give one month's notice before terminating a permanent employee. In lieu of notice, one month's wages calculated on the basis of average wages earned during the last three months shall be paid.",
  "sources": [
    { "source": "Sindh Terms of Employment standing orders Act 2015.pdf", "page": 12 },
    { "source": "Pak_labour_law.pdf", "page": 5 }
  ],
  "iterations": 2,
  "forced": false
}
```

### `POST /api/v1/ingest`
Triggers re-ingestion of PDFs without server restart.

---

## How the Agent Works

Unlike a standard RAG pipeline that fires one query and returns whatever it finds, the ReAct agent **decides what to do** at each step:

```
Question received
      ↓
[REASON] What do I need to find?
      ↓
[ACT]    retrieve("notice period termination Pakistan")
      ↓
[OBSERVE] Are these chunks sufficient to answer?
      ├── YES → final_answer() with citations
      └── NO  → [REASON] What was missing?
                      ↓
               [ACT] refine_query("notice period", "results too general")
                      ↓
               [ACT] retrieve("one month notice permanent employee ordinance")
                      ↓
               [OBSERVE] Sufficient? → final_answer()
```

Maximum 3 iterations. If still insufficient after 3 loops — answers honestly with what was found and flags `forced: true`.

---

## Running Evaluation

```bash
python -m evaluation.evaluator
```

Requires `evaluation/test_dataset.json` with format:
```json
[
  {
    "question": "What is the notice period for termination?",
    "ground_truth": "One month's notice is required..."
  }
]
```

Results saved to `evaluation/results.json`.

---

## Docker

```bash
# Build
docker build -t agentic-rag .

# Run
docker run -p 8000:8000 --env-file .env --name pakemployee agentic-rag

# Stop
docker stop pakemployee && docker rm pakemployee
```

> **Note:** Ensure `data/chunks.json` and `data/chroma_db/` exist before running Docker — they are mounted from the host.

---

## Key Design Decisions

**Why BGE over OpenAI embeddings?**  
BGE `bge-base-en-v1.5` is free, runs locally, requires no API calls, and consistently ranks at the top of the MTEB benchmark. For a legal domain with structured text, it outperforms many paid alternatives.

**Why hybrid retrieval over vector-only?**  
Legal text is dense with specific section numbers, act names, and clause references. BM25 captures exact keyword matches that semantic search misses. RRF merges both ranked lists without scale mismatch.

**Why a ReAct loop over a fixed pipeline?**  
Legal questions are rarely simple. Multi-hop questions (probation → termination → compensation) require multiple retrieval passes. A fixed pipeline fires once and accepts whatever it gets. The agent loops, refines, and verifies.

**Why Groq as fallback?**  
Gemini free tier limits are 10 RPM. A single agentic query makes 6-10 LLM calls (compression + reasoning). Groq provides 30 RPM free with sub-2s response times — treating it as a peer provider rather than last resort eliminates rate limit delays.

---

## Limitations

- Context Precision (0.50) reflects the density of Pakistani legal text — overlapping clauses across documents make clean chunk isolation difficult
- Free tier API limits mean concurrent users will hit rate limits
- ChromaDB is single-node — not suitable for horizontal scaling without migration to Qdrant or Pinecone
- No authentication on API endpoints — add API key middleware before public deployment

---

## License

MIT License — see `LICENSE` for details.

---

*Built as a portfolio project demonstrating end-to-end Agentic RAG system design, evaluation, and deployment.*