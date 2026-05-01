from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """
    What the caller sends to POST /query.

    question: the user's legal question
    max_iterations: optionally override agent loop limit (default 3)

    Why validate here instead of in the route?
    Pydantic catches bad input before it reaches your agent.
    You get automatic error messages like:
    {"detail": "question field required"} instead of a Python crash.
    """
    question: str = Field(
        ...,  # required — no default
        min_length=5,
        max_length=1000,
        description="The legal question to answer"
    )
    max_iterations: Optional[int] = Field(
        default=3,
        ge=1,  # greater than or equal to 1
        le=5,  # less than or equal to 5
        description="Max agent retrieval attempts"
    )


class SourceDocument(BaseModel):
    """
    A single cited source in the response.
    Separated from the answer so callers can render citations cleanly.
    """
    source: str
    page: int


class QueryResponse(BaseModel):
    """
    What the API returns after processing a question.

    answer      : the final answer text (with citations inline)
    sources     : structured list of cited documents
    iterations  : how many retrieval loops the agent used
    forced      : True if agent hit max iterations without a clean answer
    """
    answer: str
    sources: list[SourceDocument]
    iterations: int
    forced: bool = False


class HealthResponse(BaseModel):
    status: str
    bm25_chunks: int
    models_loaded: bool


# class IngestRequest(BaseModel):
#     """
#     For the ingest endpoint — trigger re-ingestion of PDFs.
#     pdf_dir defaults to the standard data/pdfs path.
#     """
#     pdf_dir: Optional[str] = Field(
#         default="data/pdfs",
#         description="Directory containing PDF files to ingest"
#     )


# class IngestResponse(BaseModel):
#     message: str
#     chunks_created: int