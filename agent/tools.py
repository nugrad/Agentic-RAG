from retrieval.pipeline import retrieve
from retrieval.bm25_search import BM25Index
from agent.llm import get_gemini_response


def tool_retrieve(query: str, bm25_index: BM25Index) -> dict:
    """
    Tool 1 — retrieve(query)

    Calls the full retrieval pipeline:
    hybrid search → rerank → compress.

    Returns a dict with:
    - 'chunks'     : the actual chunk dicts (for source tracking)
    - 'observation': formatted string the LLM reads in the next loop

    Why return both?
    The LLM needs a readable string to reason about.
    The agent needs the raw chunks to build citations at the end.
    Keeping both avoids re-parsing the observation string later.
    """
    chunks = retrieve(query, bm25_index)

    if not chunks:
        return {
            "chunks": [],
            "observation": "No relevant passages found for this query."
        }

    # Build a readable observation string for the LLM
    # Numbered so the LLM can reference specific passages in its reasoning
    lines = []
    for i, chunk in enumerate(chunks):
        lines.append(
            f"[Passage {i+1}]\n"
            f"Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"Content: {chunk['text']}\n"
        )

    return {
        "chunks": chunks,
        "observation": "\n".join(lines)
    }


def tool_refine_query(original_query: str, reason: str) -> dict:
    """
    Tool 2 — refine_query(original_query, reason)

    The agent calls this when it retrieved results but they
    didn't answer the question well enough.

    The agent provides:
    - original_query : what it searched before
    - reason         : why the results were insufficient

    We call the LLM to produce a better query.
    We do NOT let the agent write its own refined query directly —
    because the agent prompt is already complex. Offloading query
    refinement to a separate focused prompt gives cleaner results.

    Returns the refined query string wrapped in a dict
    (consistent return shape across all tools).
    """
    prompt = f"""You are a legal search query optimizer.

A search was performed with the following query but did not return useful results.
Your job is to write a better search query.

Original query: {original_query}
Why it failed: {reason}

Rules:
- Write ONE improved query only
- Use specific legal terminology where appropriate
- Expand abbreviations (e.g. NIRC → National Industrial Relations Commission)
- Break vague terms into specific ones
- Do not add explanation — output the query only

Improved query:"""

    refined = get_gemini_response(prompt, temperature=0)
    return {
        "refined_query": refined,
        "observation": f"Refined query produced: '{refined}'"
    }


def tool_final_answer(answer: str, sources: list[dict]) -> dict:
    """
    Tool 3 — final_answer(answer, sources)

    Called when the agent is satisfied it can answer the question.
    Formats the final response with citations.

    sources = list of chunk dicts accumulated across all retrieval steps.
    We deduplicate by (source filename, page) — same page cited twice = one citation.

    This is the ONLY way the agent should terminate.
    If the loop hits max iterations, we force-call this with whatever we have.
    You are a legal assistant for Pakistani employment and contract law.

    Answer the question below in 2-4 sentences maximum.
    Be direct. Answer ONLY what was asked.
    Use ONLY information from the provided context.
    Do not explain background. Do not add caveats unless they are in the context.
    If the context does not answer the question, say exactly: "The provided documents do not contain information about this."
    """
    # Deduplicate sources by (source, page)
    seen = set()
    unique_sources = []
    for chunk in sources:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append({
                "source": chunk["source"],
                "page": chunk["page"]
            })

    citation_lines = [
        f"  - {s['source']}, Page {s['page']}"
        for s in unique_sources
    ]

    formatted = (
        f"{answer}\n\n"
        f"Sources:\n" + "\n".join(citation_lines)
        if citation_lines
        else answer
    )

    return {
        "final_answer": formatted,
        "sources": unique_sources,
        "done": True
    }