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
        sub_queries: Optional[list[str]] = None,
        enforce_quota: bool = True,
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
            sub_queries: Optional list of decomposed sub-queries for entity balance.
            enforce_quota: If True and multiple sub-queries exist, guarantees representation
                           of each entity in final top_k.

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

        # Entity Quota Selection when multiple sub-queries are present
        if sub_queries and len(sub_queries) > 1 and enforce_quota:
            num_sub = len(sub_queries)
            min_per_subquery = max(1, top_k // (num_sub * 2))  # e.g. for top_k=8 and 2 subqueries -> 2 per subquery
            selected_cids: set[int] = set()
            selected_chunks: list[dict] = []

            # 1. Fill minimum quota per sub-query
            for q_idx in range(num_sub):
                matching = [c for c in scored if q_idx in c.get("sub_query_sources", []) and c["chunk_id"] not in selected_cids]
                for c in matching[:min_per_subquery]:
                    selected_cids.add(c["chunk_id"])
                    selected_chunks.append(c)

            # 2. Fill remaining slots up to top_k by highest final_score
            for c in scored:
                if len(selected_chunks) >= top_k:
                    break
                if c["chunk_id"] not in selected_cids:
                    selected_cids.add(c["chunk_id"])
                    selected_chunks.append(c)

            # 3. Sort selected chunks by final_score descending
            selected_chunks.sort(key=lambda x: x["final_score"], reverse=True)
            final_results = selected_chunks
        else:
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
