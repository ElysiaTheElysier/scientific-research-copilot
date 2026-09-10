import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_transform import QueryDecomposer
from src.retrieval.reranker import Reranker
from src.retrieval.vectordb import VectorStore

DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are an expert scientific research assistant.
Answer the user's question accurately, completely, and objectively using ONLY the retrieved context below.

Follow these strict guidelines:
1. Grounding & Abstention: Ground every statement strictly in the provided context. If the context does not contain sufficient information to answer the question or any of its sub-questions, explicitly state: "Based on the provided context, there is insufficient information to answer this question." Do not speculate, extrapolate, or use external knowledge.
2. Technical Precision: Preserve all exact technical terminology, mathematical notation, model names, dataset names, acronyms, and quantitative metrics exactly as written in the text without paraphrase or approximation.
3. Multi-Part & Comparison Questions: Thoroughly address ALL components of multi-part questions. For comparison or contrast questions, explicitly describe each entity or concept being compared and highlight their distinct mechanisms, differences, or trade-offs.
4. Grounded Citations: Explicitly cite the source Paper ID and Section name (e.g., [Paper: 2608.01234 | Section: 3.2 Methodology]) for each factual assertion.
5. Multimodal Evidence: If a figure visual analysis is present in the context, refer directly to the figure findings, trends, and visual data in your explanation.
6. Conciseness & Structure: Be direct, structured, and concise. Do not include internal thinking tags or filler commentary."""

RetrievalMode = Literal["dense", "bm25", "hybrid", "v2", "v3"]


class ScientificRAG:
    """Scientific RAG Copilot with modular retrieval architectures.

    Modes:
    - 'dense': Pure vector search (V1 baseline)
    - 'bm25': Pure lexical keyword search
    - 'hybrid': Reciprocal Rank Fusion (Dense + BM25)
    - 'v2': Two-Stage Hybrid + Cross-Encoder Reranker
    - 'v3': Multi-Query Balanced Retrieval + Dynamic Top-K (Default)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "v3",
        candidate_pool: int = 25,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.candidate_pool = candidate_pool
        self.query_decomposer = QueryDecomposer()

        # Initialize modular retrieval components lazily or upfront
        self.vector_store = VectorStore()
        if retrieval_mode in ("bm25", "hybrid", "v2", "v3"):
            self.bm25_index = BM25Index()
        else:
            self.bm25_index = None

        if retrieval_mode in ("hybrid", "v2", "v3"):
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
            )
        else:
            self.hybrid_retriever = None

        if retrieval_mode in ("v2", "v3"):
            self.reranker = Reranker()
        else:
            self.reranker = None

    def retrieve(self, query: str, limit: int = None, sub_queries: list[str] = None) -> list[dict]:
        """Retrieve and rank chunks according to active retrieval_mode."""
        k = limit or self.top_k

        if self.retrieval_mode == "dense":
            return self.vector_store.search(query, limit=k)

        elif self.retrieval_mode == "bm25":
            return self.bm25_index.search(query, limit=k)

        elif self.retrieval_mode == "hybrid":
            return self.hybrid_retriever.search(query, limit=k, candidate_pool=self.candidate_pool)

        elif self.retrieval_mode == "v2":
            # 1. Hybrid RRF candidate generation
            candidates = self.hybrid_retriever.search(query, limit=self.candidate_pool, candidate_pool=self.candidate_pool)
            # 2. Cross-Encoder reranking
            return self.reranker.rerank(query, candidates, top_k=k)

        elif self.retrieval_mode == "v3":
            if sub_queries is None:
                sub_queries = self.query_decomposer.decompose(query)

            if len(sub_queries) > 1:
                # Comparative / multi-part: search multi-query and enforce entity quota
                dynamic_k = limit or max(self.top_k, 8)
                candidates = self.hybrid_retriever.search_multi_query(
                    sub_queries, limit=self.candidate_pool, candidate_pool_per_query=20
                )
                return self.reranker.rerank(
                    query, candidates, top_k=dynamic_k, sub_queries=sub_queries, enforce_quota=True
                )
            else:
                # Single-focus query: standard two-stage hybrid
                candidates = self.hybrid_retriever.search(query, limit=self.candidate_pool, candidate_pool=self.candidate_pool)
                return self.reranker.rerank(query, candidates, top_k=k)

        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

    def answer(self, query: str) -> dict:
        sub_queries = [query]
        if self.retrieval_mode == "v3":
            sub_queries = self.query_decomposer.decompose(query)
            dynamic_k = 8 if len(sub_queries) > 1 else self.top_k
            retrieved_chunks = self.retrieve(query, limit=dynamic_k, sub_queries=sub_queries)
        else:
            retrieved_chunks = self.retrieve(query, limit=self.top_k)

        context_blocks = []
        citations = []

        for c in retrieved_chunks:
            chunk_type = c.get("chunk_type", "text")
            paper_id = c.get("paper_id", "Unknown")
            section = c.get("section", "General")
            image_id = c.get("image_id")

            header = f"[{chunk_type.upper()}] Paper: {paper_id} | Section: {section}"
            if image_id:
                header += f" | Image: data/processed/images/{image_id}"

            context_blocks.append(f"{header}\n{c.get('content', '')}")
            citations.append({
                "chunk_id": c.get("chunk_id"),
                "chunk_type": chunk_type,
                "paper_id": paper_id,
                "section": section,
                "image_path": f"data/processed/images/{image_id}" if image_id else None,
                "reranker_score": c.get("reranker_score"),
                "rrf_score": c.get("rrf_score"),
                "dense_score": c.get("dense_score"),
                "bm25_score": c.get("bm25_score"),
                "sub_query_sources": c.get("sub_query_sources"),
            })

        combined_context = "\n\n---\n\n".join(context_blocks)
        user_prompt = f"Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        return {
            "query": query,
            "sub_queries": sub_queries,
            "answer": response["message"]["content"],
            "citations": citations,
            "retrieval_mode": self.retrieval_mode,
        }


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Scientific Research Copilot RAG")
    parser.add_argument("query", nargs="*", help="Question to ask (if omitted, starts interactive mode)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive CLI loop")
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid", "v2", "v3"],
        default="v3",
        help="Retrieval mode (dense=V1, bm25, hybrid, v2=two-stage, v3=multi-query balanced; default: v3)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks in generation context (default: 5)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama LLM model to use (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    rag = ScientificRAG(model_name=args.model, retrieval_mode=args.mode, top_k=args.top_k)

    if args.query and not args.interactive:
        query = " ".join(args.query)
        result = rag.answer(query)
        print("\n" + "=" * 60)
        print(f"QUESTION: {result['query']} (Mode: {result['retrieval_mode'].upper()})")
        if len(result.get("sub_queries", [])) > 1:
            print("DECOMPOSED SUB-QUERIES:")
            for i, sq in enumerate(result["sub_queries"], 1):
                print(f"  [{i}] {sq}")
        print("=" * 60)
        print(f"\nANSWER:\n{result['answer']}")
        print("\n" + "-" * 60)
        print("CITATIONS & EVIDENCE USED:")
        for cit in result["citations"]:
            img_info = f" -> {cit['image_path']}" if cit["image_path"] else ""
            extra = f" (Reranker: {cit['reranker_score']:.3f})" if cit.get("reranker_score") is not None else ""
            src = f" [Source Subquery: {cit['sub_query_sources']}]" if cit.get("sub_query_sources") else ""
            print(f"- [{cit['chunk_type']}] Paper: {cit['paper_id']} | Section: {cit['section']}{img_info}{extra}{src}")
    else:
        print(f"=== Scientific Research Copilot (Mode: {args.mode.upper()}) ===")
        print("Type your question or 'exit' to quit.\n")
        while True:
            try:
                query = input("\nAsk Copilot > ").strip()
                if not query or query.lower() in ("exit", "quit", "q"):
                    break
                result = rag.answer(query)
                print("\n" + result["answer"])
                print("\nSources:")
                for cit in result["citations"]:
                    img_info = f" (Figure: {cit['image_path']})" if cit["image_path"] else ""
                    print(f"  • [{cit['chunk_type']}] {cit['paper_id']} - {cit['section']}{img_info}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
