import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from retrieval.bm25_search import load_bm25_index
from retrieval.pipeline import retrieve

load_dotenv()


def build_ragas_llm():
    """
    Groq as RAGAS judge LLM.
    llama-3.3-70b is capable enough to judge faithfulness
    and relevancy — same model your agent falls back to.
    temperature=0 for deterministic scoring.
    """
    return LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
            
        )
    )


def build_ragas_embeddings():
    """
    BGE model for RAGAS embeddings — already downloaded on Day 3.
    No extra download, no API calls, no rate limits.
    RAGAS uses embeddings only for answer_relevancy metric
    to measure semantic similarity between question and answer.
    BGE is more than capable for this.
    """
   
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def run_pipeline_for_question(question: str, bm25_index) -> dict:
    from agent.llm import get_gemini_response

    chunks = retrieve(question, bm25_index)
    contexts = [chunk["text"] for chunk in chunks]

    if not contexts:
        return {
            "contexts": [],
            "answer": "No relevant information found in the documents."
        }

    context_text = "\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""You are a legal assistant for Pakistani employment and contract law.

Answer the question below in 2-4 sentences maximum.
Be direct. Answer ONLY what was asked.
Use ONLY information from the provided context.
Do not explain background. Do not add caveats unless they are in the context.
If the context does not answer the question, say exactly: "The provided documents do not contain information about this."

Context:
{context_text}

Question: {question}

Answer:"""

    answer = get_gemini_response(prompt, temperature=0)

    return {
        "contexts": contexts,
        "answer": answer
    }


def run_evaluation(
    test_file: str = "evaluation/test_dataset.json",
    output_file: str = "evaluation/results.json"
):
    print("Loading test dataset...")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

# Deduplicate by question
    seen = set()
    deduped = []

    for item in test_data:
        if item["question"] not in seen:
            deduped.append(item)
            seen.add(item["question"])

    test_data = deduped

    print(f"Found {len(test_data)} unique test questions.")

    print("Loading BM25 index...")
    bm25_index = load_bm25_index("data/chunks.json")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(test_data):
        print(f"\nProcessing {i+1}/{len(test_data)}: {item['question'][:60]}...")

        try:
            result = run_pipeline_for_question(item["question"], bm25_index)

            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(item["ground_truth"])

            print(f"  ✅ Contexts: {len(result['contexts'])} | Answer: {result['answer'][:80]}...")

        except Exception as e:
            print(f"  ❌ Failed: {e}. Skipping.")
            continue

        print("\nBuilding RAGAS dataset...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("Configuring RAGAS with Groq + BGE...")
    ragas_llm = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False
    )

    

    results_df = results.to_pandas()

    summary = {
    "faithfulness":      float(results_df["faithfulness"].mean()),
    "answer_relevancy":  float(results_df["answer_relevancy"].mean()),
    "context_precision": float(results_df["context_precision"].mean()),
    "context_recall":    float(results_df["context_recall"].mean()),
}

    results_dict = results_df.to_dict(orient="records")

    output = {
        "summary": summary,
        "per_question": results_dict
    }

    os.makedirs("evaluation", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness      : {summary['faithfulness']:.4f}")
    print(f"Answer Relevancy  : {summary['answer_relevancy']:.4f}")
    print(f"Context Precision : {summary['context_precision']:.4f}")
    print(f"Context Recall    : {summary['context_recall']:.4f}")
    print("="*50)
    print(f"\nSaved to: {output_file}")

    return summary   # ← This should be indented at the same level as the other statements in the function


if __name__ == "__main__":
    run_evaluation()