import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from src.retrieval.vectordb import VectorStore

DEFAULT_MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """You are an evidence-grounded scientific research assistant.
Answer the user's question using ONLY the retrieved context below.
Cite the relevant Paper ID and Section for your claims.
If a figure visual analysis is provided, refer to the figure and its findings."""


class ScientificRAG:
    def __init__(self, model_name: str = DEFAULT_MODEL, top_k: int = 5):
        self.store = VectorStore()
        self.model_name = model_name
        self.top_k = top_k

    def answer(self, query: str) -> dict:
        retrieved_chunks = self.store.search(query, limit=self.top_k)

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
        }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rag = ScientificRAG()

    if len(sys.argv) > 1 and sys.argv[1] not in ("-i", "--interactive"):
        query = " ".join(sys.argv[1:])
        result = rag.answer(query)
        print("\n" + "=" * 60)
        print(f"QUESTION: {result['query']}")
        print("=" * 60)
        print(f"\nANSWER:\n{result['answer']}")
        print("\n" + "-" * 60)
        print("CITATIONS & EVIDENCE USED:")
        for cit in result["citations"]:
            img_info = f" -> {cit['image_path']}" if cit["image_path"] else ""
            print(f"- [{cit['chunk_type']}] Paper: {cit['paper_id']} | Section: {cit['section']}{img_info}")
    else:
        print("=== Scientific Research Copilot (Baseline RAG) ===")
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

