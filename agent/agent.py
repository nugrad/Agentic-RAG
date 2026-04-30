import json
from retrieval.bm25_search import BM25Index
from agent.llm import get_gemini_response
from agent.prompts import build_prompt
from agent.tools import tool_retrieve, tool_refine_query, tool_final_answer


MAX_ITERATIONS = 3  # Hard ceiling on retrieval attempts


def parse_llm_response(response_text: str) -> dict:
    """
    The LLM should return valid JSON. In practice it sometimes wraps
    it in markdown code fences (```json ... ```). We strip those.

    If parsing fails entirely — we do NOT crash the agent.
    We return a fallback that forces the agent to answer with what it has.
    Crashing on a JSON parse error in production is unacceptable.
    """
    cleaned = response_text.strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        cleaned = "\n".join(lines[1:-1])

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  [AGENT] JSON parse failed. Raw response:\n{response_text}")
        # Graceful fallback — force final answer
        return {
            "thought": "Could not parse structured response. Ending loop.",
            "action": "final_answer",
            "action_input": {
                "answer": "I encountered an issue processing the response. Please try again.",
                "sources": []
            }
        }


def run_agent(user_query: str, bm25_index: BM25Index) -> dict:
    """
    The ReAct loop.

    State we maintain across iterations:
    - history       : all thoughts/actions/observations so far
    - all_chunks    : every chunk retrieved across all iterations
                      (for citation building at the end)
    - iterations    : current loop count

    Loop mechanics:
    1. Build prompt with current history
    2. Call LLM → get JSON response
    3. Parse JSON → extract thought, action, action_input
    4. Execute the tool
    5. Append action + observation to history
    6. If action was final_answer → break
    7. If iterations == MAX_ITERATIONS → force final_answer
    8. Else → loop back to step 1

    Why a while loop and not recursion?
    Recursion has a stack depth limit. While loop is explicit and
    easier to add a hard stop to. Simpler to debug.
    """
    history = []
    all_chunks = []
    iterations = 0

    print(f"\n{'='*60}")
    print(f"AGENT START: {user_query}")
    print(f"{'='*60}")

    while iterations < MAX_ITERATIONS:
        iterations += 1
        print(f"\n--- Iteration {iterations}/{MAX_ITERATIONS} ---")

        # Build prompt with full history
        prompt = build_prompt(user_query, history)

        # Call LLM
        raw_response = get_gemini_response(prompt, temperature=0)
        print(f"[THOUGHT+ACTION RAW]: {raw_response[:300]}...")

        # Parse LLM response
        parsed = parse_llm_response(raw_response)

        thought = parsed.get("thought", "")
        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})

        print(f"[THOUGHT]: {thought}")
        print(f"[ACTION]:  {action}")

        # Append action to history so LLM sees it next iteration
        history.append({
            "role": "action",
            "content": f"{action}({json.dumps(action_input)})"
        })

        # --- Execute Tool ---

        if action == "retrieve":
            query = action_input.get("query", user_query)
            result = tool_retrieve(query, bm25_index)

            # Accumulate chunks for citation tracking
            all_chunks.extend(result["chunks"])

            observation = result["observation"]
            print(f"[OBSERVATION]: Retrieved {len(result['chunks'])} chunks")

        elif action == "refine_query":
            original = action_input.get("original_query", user_query)
            reason = action_input.get("reason", "results were not relevant")
            result = tool_refine_query(original, reason)

            observation = result["observation"]
            print(f"[OBSERVATION]: {observation}")

        elif action == "final_answer":
            answer = action_input.get("answer", "No answer generated.")
            result = tool_final_answer(answer, all_chunks)

            print(f"\n{'='*60}")
            print("AGENT COMPLETE")
            print(f"{'='*60}")
            print(result["final_answer"])

            return {
                "answer": result["final_answer"],
                "sources": result["sources"],
                "iterations": iterations,
                "history": history
            }

        else:
            # Unknown action — agent went off script
            # Log it and force termination on next iteration
            observation = f"Unknown action '{action}'. Use only: retrieve, refine_query, final_answer."
            print(f"[WARNING]: {observation}")

        # Append observation to history
        history.append({
            "role": "observation",
            "content": observation
        })

    # --- Max iterations reached without final_answer ---
    # Force an answer rather than returning nothing
    print(f"\n[AGENT] Max iterations reached. Forcing final answer.")

    forced_answer = _force_answer(user_query, all_chunks)
    result = tool_final_answer(forced_answer, all_chunks)

    return {
        "answer": result["final_answer"],
        "sources": result["sources"],
        "iterations": iterations,
        "history": history,
        "forced": True  # Flag so caller knows this was a forced termination
    }


def _force_answer(user_query: str, chunks: list[dict]) -> str:
    """
    Called when agent hits max iterations without calling final_answer.
    Uses whatever chunks were collected to generate a best-effort answer.
    Honest about the limitation — does not fabricate confidence.
    """
    if not chunks:
        return (
            "I was unable to find sufficient information in the available "
            "legal documents to answer your question. Please consult a "
            "qualified legal professional."
        )

    context = "\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in chunks[:5]  # Use top 5 collected chunks
    ])

    prompt = f"""Based ONLY on the legal passages below, answer the question.
If the passages don't fully answer the question, say what you found and what is missing.
Do not add information beyond what is in the passages.

Question: {user_query}

Passages:
{context}

Answer:"""

    return get_gemini_response(prompt, temperature=0)


if __name__ == "__main__":
    from retrieval.bm25_search import load_bm25_index

    bm25 = load_bm25_index()

    # Test 1 — Simple question
    result = run_agent(
        "What notice period must an employer give before terminating a permanent employee?",
        bm25
    )
    print("\n\nFINAL ANSWER:")
    print(result["answer"])
    print(f"\nIterations used: {result['iterations']}")
    print(f"Sources cited: {len(result['sources'])}")

    print("\n" + "="*60 + "\n")

    # Test 2 — Multi-hop question (agent should loop)
    result2 = run_agent(
        "What are the rights of an employee terminated during probation and what compensation are they entitled to under Pakistani law?",
        bm25
    )
    print("\n\nFINAL ANSWER:")
    print(result2["answer"])
    print(f"\nIterations used: {result2['iterations']}")