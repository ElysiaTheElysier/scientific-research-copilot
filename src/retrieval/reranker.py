"""Cross-Encoder Reranking Module for Passage and Evidence Re-ordering.

Uses a specialized passage ranking cross-encoder (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
to compute joint cross-attention query-document scores, significantly improving top-K precision.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Optional
import torch
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Reranks candidate chunks using a neural CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        batch_size: int = 16,
    ) -> list[dict]:
        """Rerank candidates using cross-encoder relevance scores.

        Args:
            query: Question or search query.
            candidates: List of retrieved chunk dictionaries.
            top_k: Number of top reranked chunks to return.
            batch_size: Batch size for cross-encoder inference.

        Returns:
            Top-K candidate chunks sorted by reranker_score descending.
        """
        if not candidates:
            return []

        # Prepare query-passage pairs
        pairs = []
        for c in candidates:
            # Include paper_id and section as context prefix
            passage = f"Paper: {c.get('paper_id', '')} Section: {c.get('section', '')}\n{c.get('content', '')}"
            pairs.append((query, passage))

        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Attach scores
        scored_candidates = []
        for chunk, score in zip(candidates, scores):
            c_copy = dict(chunk)
            c_copy["reranker_score"] = float(score)
            scored_candidates.append(c_copy)

        # Sort descending
        scored_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Re-assign rank
        reranked = scored_candidates[:top_k]
        for rank, c in enumerate(reranked, 1):
            c["reranker_rank"] = rank

        return reranked


if __name__ == "__main__":
    from src.retrieval.hybrid import HybridRetriever

    hybrid = HybridRetriever()
    reranker = Reranker()

    query = "What does ECS stand for in Homo-RAG?"
    candidates = hybrid.search(query, limit=15)
    top_reranked = reranker.rerank(query, candidates, top_k=5)

    print(f"\nReranked Top-5 for '{query}':")
    for r in top_reranked:
        print(f"Rank {r['reranker_rank']} (CrossEncoder Score: {r['reranker_score']:.4f} | RRF Rank: {r.get('rrf_rank')}):")
        print(f"  Chunk {r['chunk_id']} | Paper: {r['paper_id']} | Section: {r['section']}")
        print(f"  Snippet: {r['content'][:140]}...")
