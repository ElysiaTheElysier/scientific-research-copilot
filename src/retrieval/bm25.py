"""BM25 Lexical Retrieval Module for Scientific Papers.

Implements a pure Python BM25Okapi index tailored for scientific and technical text:
- Robust tokenization preserving acronyms (ECS, AWM), camelCase, identifiers, and hyphenated terms.
- Fast in-memory inverted index and vector scoring for 374 chunks.
- Returns ranked chunk payloads with BM25 scores.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"


def tokenize_scientific_text(text: str) -> list[str]:
    """Tokenize text preserving scientific terminology, acronyms, and compound terms."""
    if not text:
        return []

    # Replace common markdown formatting
    clean = re.sub(r"[#*_`\[\]\(\)\{\}<>|]", " ", text)

    # Extract alphanumeric words including compound words with hyphens or underscores
    raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", clean)

    tokens = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        tokens.append(tok_lower)

        # For compound words (e.g., post-graph-rag), also index sub-parts
        if "-" in tok or "_" in tok:
            sub_parts = re.split(r"[-_]", tok)
            for sub in sub_parts:
                if len(sub) > 1:
                    tokens.append(sub.lower())

        # For camelCase words (e.g. Doc2Graph, SelfGraphRAG), extract parts
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+", tok)
        if len(camel_parts) > 1:
            for cp in camel_parts:
                if len(cp) > 1:
                    tokens.append(cp.lower())

    return tokens


class BM25Index:
    """In-memory BM25Okapi index for scientific chunks."""

    def __init__(
        self,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks_path = Path(chunks_path)
        self.k1 = k1
        self.b = b
        self.chunks: list[dict] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: list[int] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._build_index()

    def _build_index(self):
        """Load chunks and build inverted index."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.corpus_size = len(self.chunks)
        if self.corpus_size == 0:
            return

        total_len = 0
        for doc_idx, chunk in enumerate(self.chunks):
            # Index content plus section and paper_id for richer lexical matching
            text_to_index = f"{chunk.get('paper_id', '')} {chunk.get('section', '')} {chunk.get('content', '')}"
            tokens = tokenize_scientific_text(text_to_index)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.doc_freqs[term] += 1
                self.postings[term].append((doc_idx, count))

        self.avg_doc_len = total_len / self.corpus_size

        # Compute Robertson-Spärck Jones IDF with smoothing
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search the BM25 index with a natural language scientific query."""
        query_tokens = tokenize_scientific_text(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.postings:
                continue

            idf_val = self.idf[term]
            for doc_idx, tf in self.postings[term]:
                doc_len = self.doc_lens[doc_idx]
                # BM25 standard formula
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf_val * ((tf * (self.k1 + 1.0)) / denom)
                scores[doc_idx] += term_score

        if not scores:
            # Fallback to returning top chunks if no terms matched
            return []

        # Sort by score descending
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for rank, (doc_idx, score) in enumerate(sorted_indices, 1):
            chunk = dict(self.chunks[doc_idx])
            chunk["bm25_score"] = round(score, 4)
            chunk["bm25_rank"] = rank
            results.append(chunk)

        return results


if __name__ == "__main__":
    import sys

    index = BM25Index()
    query = sys.argv[1] if len(sys.argv) > 1 else "Evidence Confidence Score ECS Homo-RAG"
    res = index.search(query, limit=5)
    print(f"BM25 Search Results for '{query}':")
    for r in res:
        print(f"Rank {r['bm25_rank']} (Score: {r['bm25_score']}): Chunk {r['chunk_id']} | Paper: {r['paper_id']} | Section: {r['section']}")
        print(f"  Content: {r['content'][:120]}...")
