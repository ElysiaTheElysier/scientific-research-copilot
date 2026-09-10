"""Hybrid Retrieval Module fusing Dense Vector Search and BM25 via Reciprocal Rank Fusion (RRF).

Combines:
- Dense semantic vector search via Qdrant (Qwen3-Embedding-0.6B)
- BM25 lexical keyword search via BM25Index
- Reciprocal Rank Fusion (RRF, Cormack et al.) with standard constant k=60
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Optional
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectordb import VectorStore

RRF_K = 60


class HybridRetriever:
    """Combines Dense Vector and BM25 Lexical retrieval using RRF."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: int = RRF_K,
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        limit: int = 10,
        candidate_pool: int = 25,
    ) -> list[dict]:
        """Execute hybrid search using Reciprocal Rank Fusion.

        Args:
            query: Natural language query.
            limit: Final number of ranked chunks to return.
            candidate_pool: Number of candidate chunks to fetch from each retriever.

        Returns:
            List of chunk dicts sorted by fused RRF score descending.
        """
        # 1. Retrieve Dense Candidates with scores
        query_vector = self.vector_store.model.encode(query).tolist()
        dense_points = self.vector_store.client.query_points(
            collection_name=self.vector_store.client.get_collections().collections[0].name,
            query=query_vector,
            limit=candidate_pool,
            with_payload=True,
        ).points

        dense_ranks: dict[int, int] = {}
        dense_scores: dict[int, float] = {}
        chunk_map: dict[int, dict] = {}

        for rank, pt in enumerate(dense_points, 1):
            cid = pt.payload.get("chunk_id")
            dense_ranks[cid] = rank
            dense_scores[cid] = float(pt.score)
            chunk_map[cid] = dict(pt.payload)

        # 2. Retrieve BM25 Candidates
        bm25_results = self.bm25_index.search(query, limit=candidate_pool)
        bm25_ranks: dict[int, int] = {}
        bm25_scores: dict[int, float] = {}

        for rank, b_chunk in enumerate(bm25_results, 1):
            cid = b_chunk.get("chunk_id")
            bm25_ranks[cid] = rank
            bm25_scores[cid] = float(b_chunk.get("bm25_score", 0.0))
            if cid not in chunk_map:
                chunk_map[cid] = dict(b_chunk)

        # 3. Compute Reciprocal Rank Fusion (RRF) Scores
        all_chunk_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        fused_scores: dict[int, float] = {}

        for cid in all_chunk_ids:
            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[cid])
            fused_scores[cid] = score

        # 4. Sort and format final results
        sorted_cids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        final_results = []
        for rank, (cid, rrf_score) in enumerate(sorted_cids, 1):
            chunk = chunk_map[cid]
            chunk["rrf_score"] = round(rrf_score, 6)
            chunk["rrf_rank"] = rank
            chunk["dense_rank"] = dense_ranks.get(cid)
            chunk["dense_score"] = round(dense_scores.get(cid, 0.0), 4) if cid in dense_scores else None
            chunk["bm25_rank"] = bm25_ranks.get(cid)
            chunk["bm25_score"] = round(bm25_scores.get(cid, 0.0), 4) if cid in bm25_scores else None
            final_results.append(chunk)

        return final_results

    def search_multi_query(
        self,
        sub_queries: list[str],
        limit: int = 25,
        candidate_pool_per_query: int = 20,
    ) -> list[dict]:
        """Execute multi-query hybrid search with balanced round-robin candidate interleaving.

        Args:
            sub_queries: List of decomposed sub-queries (e.g. one for each comparative entity).
            limit: Total candidate pool size to return for downstream reranking.
            candidate_pool_per_query: Number of candidates to fetch per sub-query.

        Returns:
            List of unique chunk dicts balanced across sub-queries, with source tracking.
        """
        if not sub_queries:
            return []
        if len(sub_queries) == 1:
            return self.search(sub_queries[0], limit=limit, candidate_pool=candidate_pool_per_query)

        # 1. Retrieve hybrid candidates independently for each sub-query
        per_query_candidates: list[list[dict]] = []
        for sq in sub_queries:
            results = self.search(sq, limit=candidate_pool_per_query, candidate_pool=candidate_pool_per_query)
            per_query_candidates.append(results)

        # 2. Balanced Round-Robin Interleaving to guarantee entity diversity
        interleaved_cids = []
        seen_cids = set()
        chunk_map: dict[int, dict] = {}
        max_depth = max((len(cands) for cands in per_query_candidates), default=0)

        for depth in range(max_depth):
            for q_idx, cands in enumerate(per_query_candidates):
                if depth < len(cands):
                    chunk = cands[depth]
                    cid = chunk.get("chunk_id")
                    if cid not in seen_cids:
                        seen_cids.add(cid)
                        c_copy = dict(chunk)
                        c_copy["sub_query_sources"] = [q_idx]
                        c_copy["sub_query_ranks"] = {q_idx: depth + 1}
                        chunk_map[cid] = c_copy
                        interleaved_cids.append(cid)
                    else:
                        # Chunk was already retrieved by an earlier sub-query: mark dual relevance
                        if q_idx not in chunk_map[cid]["sub_query_sources"]:
                            chunk_map[cid]["sub_query_sources"].append(q_idx)
                            chunk_map[cid]["sub_query_ranks"][q_idx] = depth + 1
                            # Boost RRF score for multi-query consensus
                            chunk_map[cid]["rrf_score"] = round(
                                chunk_map[cid]["rrf_score"] + chunk.get("rrf_score", 0.0), 6
                            )

                    if len(interleaved_cids) >= limit:
                        break
            if len(interleaved_cids) >= limit:
                break

        # 3. Format final candidates with round-robin rank
        final_candidates = []
        for rank, cid in enumerate(interleaved_cids[:limit], 1):
            chunk = chunk_map[cid]
            chunk["rrf_rank"] = rank
            final_candidates.append(chunk)

        return final_candidates


if __name__ == "__main__":
    import sys

    hybrid = HybridRetriever()
    query = sys.argv[1] if len(sys.argv) > 1 else "What does ECS stand for in Homo-RAG?"
    results = hybrid.search(query, limit=5)
    print(f"Hybrid Search Results for '{query}':")
    for r in results:
        print(f"Rank {r['rrf_rank']} (RRF: {r['rrf_score']:.5f} | DenseRank: {r['dense_rank']} | BM25Rank: {r['bm25_rank']}): Chunk {r['chunk_id']} | {r['paper_id']} | {r['section']}")
        print(f"  {r['content'][:140]}...")
