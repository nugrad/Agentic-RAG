SYSTEM_PROMPT = """You are a legal research agent specializing in Pakistani employment and contract law.

Your job is to answer user questions accurately using ONLY the legal documents available to you.

You have access to exactly 3 tools:

1. retrieve
   Input: { "query": "your search query" }
   Use when: You need to find relevant legal passages to answer the question.

2. refine_query  
   Input: { "original_query": "...", "reason": "why it failed" }
   Use when: You retrieved results but they did not adequately address the question.

3. final_answer
   Input: { "answer": "your complete answer", "sources": [] }
   Use when: You have enough information to answer confidently.
   The sources list will be filled automatically — pass an empty list [].

STRICT RULES:
- You MUST retrieve before answering. Never answer from memory alone.
- You MUST attempt retrieval at least once, even if the question seems simple.
- Maximum 3 retrieval attempts. After 3, call final_answer with what you have.
- If after all attempts you cannot find the answer, say so explicitly. Do not fabricate.
- Base your answer ONLY on retrieved passages. Do not add knowledge from outside.
- Always respond in valid JSON. Nothing outside the JSON block.

RESPONSE FORMAT — use this exact structure every single turn:
{
  "thought": "Your reasoning about what to do next and why",
  "action": "retrieve | refine_query | final_answer",
  "action_input": { ...parameters for the chosen tool... }
}"""


def build_prompt(user_query: str, history: list[dict]) -> str:
    """
    Assembles the full prompt the LLM receives on each loop iteration.

    history = list of dicts, each with role and content:
    [
      {"role": "action",      "content": "retrieve(...)"},
      {"role": "observation", "content": "Passage 1: ..."},
      {"role": "action",      "content": "refine_query(...)"},
      {"role": "observation", "content": "Refined query: ..."},
    ]

    Why pass full history every time?
    LLMs have no memory between calls. Every call is stateless.
    The history IS the agent's memory — we reconstruct it manually.
    This is the fundamental mechanic of any stateful LLM loop.
    """
    history_text = ""
    if history:
        history_text = "\n\nPrevious steps:\n"
        for step in history:
            history_text += f"[{step['role'].upper()}]: {step['content']}\n"

    return f"{SYSTEM_PROMPT}{history_text}\n\nUser question: {user_query}\n\nRespond with JSON:"