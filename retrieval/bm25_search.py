from rank_bm25 import BM25Okapi
import json
import re


def tokenize(text: str) -> list[str]:
    """
    BM25 works on tokens (words), not raw strings.
    We lowercase and split on non-alphanumeric characters.

    Why custom tokenizer instead of .split()?
    Legal text has terms like "section-14", "sub-clause(a)", "30-days".
    Simple split() keeps these as one token — losing "section", "14" separately.
    This tokenizer splits them properly.

    We do NOT remove stopwords here.
    In legal text, "not", "shall", "may", "without" change meaning completely.
    Removing them would destroy precision.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


class BM25Index:
    """
    Wraps rank_bm25 with our chunk structure.
    Built in memory — no persistence needed.
    BM25 is fast to rebuild from chunks.json on every startup.
    No point in persisting a BM25 index.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Returns top_k chunks ranked by BM25 score.
        Each result includes the chunk dict + its bm25_score + its rank.
        """
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get indices sorted by score descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            chunk["bm25_rank"] = rank
            results.append(chunk)

        return results


def load_bm25_index(chunks_file: str = "data/chunks.json") -> BM25Index:
    """
    Loads chunks from disk and builds BM25 index.
    Call this once at startup — store the returned object in memory.
    Do not call inside a per-request function.
    """
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return BM25Index(chunks)


if __name__ == "__main__":
    index = load_bm25_index()
    results = index.search("termination of employment notice period", top_k=5)

    print("\n--- BM25 Top 5 ---")
    for r in results:
        print(f"\nRank {r['bm25_rank'] + 1} | Score: {r['bm25_score']:.4f}")
        print(f"Source: {r['source']}, Page: {r['page']}")
        print(f"Text: {r['text'][:200]}...")