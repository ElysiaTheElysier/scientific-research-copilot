"""Cross-Encoder Reranking Module with Two-Stage Rank Fusion.

Uses a specialized passage ranking cross-encoder (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
combined with First-Stage Reciprocal Rank Fusion (RRF) to prevent reranker false-positive
drift and achieve high MRR while preserving retrieval recall.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Optional
import torch
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_FUSION_K = 60
DEFAULT_RRF_WEIGHT = 0.60


class Reranker:
    """Reranks candidate chunks using neural CrossEncoder and Two-Stage Rank Fusion."""

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
        use_rank_fusion: bool = True,
        rrf_weight: float = DEFAULT_RRF_WEIGHT,
        fusion_k: int = DEFAULT_FUSION_K,
    ) -> list[dict]:
        """Rerank candidates using cross-encoder relevance scores and Two-Stage Rank Fusion.

        Args:
            query: Question or search query.
            candidates: List of retrieved chunk dictionaries from first-stage retrieval.
            top_k: Number of top reranked chunks to return.
            batch_size: Batch size for cross-encoder inference.
            use_rank_fusion: If True, fuses first-stage RRF rank with cross-encoder rank
                             to prevent false-positive drift and preserve high recall.
            rrf_weight: Weight given to first-stage RRF rank (0.60 optimal per grid search).
            fusion_k: Smoothing constant for reciprocal rank fusion.

        Returns:
            Top-K candidate chunks sorted by final relevance score descending.
        """
        if not candidates:
            return []

        # Prepare query-passage pairs cleanly
        pairs = []
        for c in candidates:
            sec = c.get("section", "")
            content = c.get("content", "")
            if sec and sec != "General" and not content.startswith(sec):
                passage = f"{sec}\n{content}"
            else:
                passage = content
            pairs.append((query, passage[:2000]))

        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Attach cross-encoder scores and sort to get reranker rank
        scored = []
        for chunk, score in zip(candidates, scores):
            c_copy = dict(chunk)
            c_copy["reranker_score"] = float(score)
            scored.append(c_copy)

        scored.sort(key=lambda x: x["reranker_score"], reverse=True)
        for rank, c in enumerate(scored, 1):
            c["reranker_rank"] = rank

        # Two-Stage Rank Fusion
        if use_rank_fusion:
            for c in scored:
                r_rrf = c.get("rrf_rank", len(scored))
                r_re = c.get("reranker_rank", len(scored))
                c["final_score"] = rrf_weight * (1.0 / (fusion_k + r_rrf)) + (1.0 - rrf_weight) * (1.0 / (fusion_k + r_re))
            scored.sort(key=lambda x: x["final_score"], reverse=True)
        else:
            for c in scored:
                c["final_score"] = c["reranker_score"]

        # Return top_k with final rank
        final_results = scored[:top_k]
        for rank, c in enumerate(final_results, 1):
            c["final_rank"] = rank

        return final_results


if __name__ == "__main__":
    from src.retrieval.hybrid import HybridRetriever

    hybrid = HybridRetriever()
    reranker = Reranker()

    query = "What does ECS stand for in Homo-RAG?"
    candidates = hybrid.search(query, limit=20, candidate_pool=25)
    top_reranked = reranker.rerank(query, candidates, top_k=5)

    print(f"\nTwo-Stage Reranked Top-5 for '{query}':")
    for r in top_reranked:
        print(f"Rank {r['final_rank']} (FinalScore: {r['final_score']:.6f} | RerankerScore: {r['reranker_score']:.3f} | RRF Rank: {r.get('rrf_rank')}):")
        print(f"  Chunk {r['chunk_id']} | Paper: {r['paper_id']} | Section: {r['section']}")
        print(f"  Snippet: {r['content'][:140]}...")
