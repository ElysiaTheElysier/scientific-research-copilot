"""Query Transformation and Decomposition Module for Scientific RAG.

Provides automated decomposition of comparative, multi-entity, and multi-part queries
into focused, standalone sub-queries. This prevents cross-paper retrieval starvation
and ensures balanced evidence retrieval across all target subjects.
"""

import json
import re
from typing import Optional
import ollama

# Syntactic pattern rules for zero-latency deterministic query decomposition
DECOMPOSITION_PATTERNS = [
    # 1. How do/does A and B differ in/on/regarding C
    (
        re.compile(r"^How\s+do(?:es)?\s+(.+?)\s+and\s+(.+?)\s+differ\s+(?:in|on|regarding)\s+(.+?)\??$", re.IGNORECASE),
        lambda m: (m.group(1), m.group(2), m.group(3)),
    ),
    # 2. How do/does A differ from B in/on/regarding C
    (
        re.compile(r"^How\s+do(?:es)?\s+(.+?)\s+differ\s+from\s+(.+?)\s+(?:in|on|regarding)\s+(.+?)\??$", re.IGNORECASE),
        lambda m: (m.group(1), m.group(2), m.group(3)),
    ),
    # 3. How do/does A compare with/to B in/on/regarding C
    (
        re.compile(r"^How\s+do(?:es)?\s+(.+?)\s+compare\s+(?:with|to)\s+(.+?)\s+(?:in|on|regarding)\s+(.+?)\??$", re.IGNORECASE),
        lambda m: (m.group(1), m.group(2), m.group(3)),
    ),
    # 4. How do/does A and B address/handle/evaluate/approach/treat C
    (
        re.compile(r"^How\s+do(?:es)?\s+(.+?)\s+and\s+(.+?)\s+(?:address|handle|evaluate|approach|treat)\s+(.+?)\??$", re.IGNORECASE),
        lambda m: (m.group(1), m.group(2), m.group(3)),
    ),
    # 5. Compare A and/with/to/vs B in/on/regarding C
    (
        re.compile(r"^Compare\s+(.+?)\s+(?:with|to|and|vs\.?|versus)\s+(.+?)\s+(?:in|on|regarding)\s+(.+?)\??$", re.IGNORECASE),
        lambda m: (m.group(1), m.group(2), m.group(3)),
    ),
    # 6. What are the differences between A and B in/on/regarding C
    (
        re.compile(r"^What\s+are\s+the\s+differences\s+between\s+(.+?)\s+and\s+(.+?)\s+(?:in|on|regarding)\s+(.+?)\??$", re.IGNORECASE),
        lambda m: (m.group(1), m.group(2), m.group(3)),
    ),
]

COMPARATIVE_INDICATORS = {
    "differ", "difference", "differences", "compare", "comparing", "comparison",
    "versus", "vs", "vs.", "contrast", "contrasting",
}


class QueryDecomposer:
    """Decomposes comparative and multi-part questions into focused sub-queries."""

    def __init__(self, fallback_model: str = "qwen2.5:3b"):
        self.fallback_model = fallback_model

    @staticmethod
    def _clean_text(s: str) -> str:
        s = s.strip().strip("'\"`")
        # Strip common syntactic prefixes
        for prefix in ("their approach to ", "its approach to ", "the approach to ", "the impact of "):
            if s.lower().startswith(prefix):
                s = s[len(prefix):]
        return s.strip()

    def is_comparative(self, query: str) -> bool:
        """Check if query contains comparative keywords."""
        words = set(re.findall(r"\b\w+\b", query.lower()))
        return bool(words & COMPARATIVE_INDICATORS)

    def decompose(self, query: str, use_llm_fallback: bool = True) -> list[str]:
        """Decompose a query into sub-queries.

        Returns:
            list[str]: 2 or more focused sub-queries if comparative, or [query] if single-focus.
        """
        clean_q = query.strip()

        # 1. Fast Pattern Matching (Zero-Latency Deterministic Extraction)
        for pattern, extractor in DECOMPOSITION_PATTERNS:
            match = pattern.match(clean_q)
            if match:
                ent_a, ent_b, aspect = extractor(match)
                clean_a = self._clean_text(ent_a)
                clean_b = self._clean_text(ent_b)
                clean_aspect = self._clean_text(aspect)

                sub_q1 = f"{clean_a} {clean_aspect}".strip()
                sub_q2 = f"{clean_b} {clean_aspect}".strip()
                return [sub_q1, sub_q2]

        # If no regex matched and query does not contain comparative signals, return original
        if not self.is_comparative(clean_q) or not use_llm_fallback:
            return [clean_q]

        # 2. Lightweight LLM Decomposition Fallback
        try:
            prompt = (
                f"Decompose the following scientific research question into 2 standalone search queries, "
                f"one focusing on each entity or method being compared.\n\n"
                f"Question: {clean_q}\n\n"
                f"Respond with ONLY a valid JSON object matching this schema:\n"
                f'{{"sub_queries": ["<query for entity 1>", "<query for entity 2>"]}}'
            )
            response = ollama.chat(
                model=self.fallback_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            content = response["message"]["content"].strip()
            # Extract JSON from potential code fences
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                sub_queries = data.get("sub_queries", [])
                if isinstance(sub_queries, list) and len(sub_queries) >= 2:
                    return [s.strip() for s in sub_queries if isinstance(s, str) and s.strip()]
        except Exception:
            pass

        return [clean_q]


if __name__ == "__main__":
    decomposer = QueryDecomposer()
    test_queries = [
        "How do post-graph-rag and SelfGraphRAG differ in their approach to handling entity deduplication and graph extraction quality?",
        "How do retrieval-stage defenses differ from rerank-stage defenses in mitigating adversarial attacks on RAG pipelines?",
        "How do 'Why RAGs Hallucinate' and 'Assessing the Downstream Utility of Evidence-Aware Retrieval' address the limitations of conventional retrieval evaluation metrics?",
        "How does STeReO compare with conventional pipeline-cascade approaches (ASR + Text RAG) in retrieving spoken audio content?",
        "How does the impact of misleading context differ from the impact of distraction context on model accuracy in company factual QA?",
        "What vector database and embedding model are used in post-graph-rag?",
    ]

    print("=== Testing QueryDecomposer ===")
    for q in test_queries:
        subs = decomposer.decompose(q)
        print(f"\nOriginal: {q}")
        if len(subs) > 1:
            print("Decomposed into:")
            for i, s in enumerate(subs, 1):
                print(f"  [{i}] {s}")
        else:
            print("  -> Single query (no decomposition)")

