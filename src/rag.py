import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vectordb import VectorStore

DEFAULT_MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """You are an evidence-grounded scientific research assistant.
Answer the user's question using ONLY the retrieved context below.
Cite the relevant Paper ID and Section for your claims.
If a figure visual analysis is provided, refer to the figure and its findings."""

RetrievalMode = Literal["dense", "bm25", "hybrid", "v2"]


class ScientificRAG:
    """Scientific RAG Copilot with modular retrieval architectures.

    Modes:
    - 'dense': Pure vector search (V1 baseline)
    - 'bm25': Pure lexical keyword search
    - 'hybrid': Reciprocal Rank Fusion (Dense + BM25)
    - 'v2': Full V2 pipeline (Dense + BM25 -> RRF -> Cross-Encoder Reranker)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "v2",
        candidate_pool: int = 20,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.candidate_pool = candidate_pool

        # Initialize modular retrieval components lazily or upfront
        self.vector_store = VectorStore()
        if retrieval_mode in ("bm25", "hybrid", "v2"):
            self.bm25_index = BM25Index()
        else:
            self.bm25_index = None

        if retrieval_mode in ("hybrid", "v2"):
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
            )
        else:
            self.hybrid_retriever = None

        if retrieval_mode == "v2":
            self.reranker = Reranker()
        else:
            self.reranker = None

    def retrieve(self, query: str, limit: int = None) -> list[dict]:
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

        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

    def answer(self, query: str) -> dict:
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
        choices=["dense", "bm25", "hybrid", "v2"],
        default="v2",
        help="Retrieval mode (dense=V1, bm25, hybrid, v2=hybrid+reranker; default: v2)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks in generation context (default: 5)")

    args = parser.parse_args()

    rag = ScientificRAG(retrieval_mode=args.mode, top_k=args.top_k)

    if args.query and not args.interactive:
        query = " ".join(args.query)
        result = rag.answer(query)
        print("\n" + "=" * 60)
        print(f"QUESTION: {result['query']} (Mode: {result['retrieval_mode'].upper()})")
        print("=" * 60)
        print(f"\nANSWER:\n{result['answer']}")
        print("\n" + "-" * 60)
        print("CITATIONS & EVIDENCE USED:")
        for cit in result["citations"]:
            img_info = f" -> {cit['image_path']}" if cit["image_path"] else ""
            extra = f" (Reranker: {cit['reranker_score']:.3f})" if cit.get("reranker_score") is not None else ""
            print(f"- [{cit['chunk_type']}] Paper: {cit['paper_id']} | Section: {cit['section']}{img_info}{extra}")
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
