# V3 Baseline Specification & Benchmark Results

**Frozen At**: 2026-09-10  
**Git Tag**: `v3-multi-query`  
**Evaluation Results**: `evaluation/results/v3_results.json`, `evaluation/results/v3_summary.csv`  
**Experiment Log**: `evaluation/V3_EXPERIMENT_LOG.md`  
**Comparison Report**: `evaluation/V2_1_V3_COMPARISON.md`  

---

## 1. System Architecture & Pipeline Flow

The V3 system introduces **Query Decomposition, Balanced Round-Robin Multi-Query Retrieval, Entity Quota Reranking, and Dynamic Context Budgeting**:

```
[User Query]
       │
       ▼
[Query Decomposition Layer]
  • Module: src/retrieval/query_transform.py (QueryDecomposer)
  • Logic: Fast syntactic regex extraction + local LLM fallback
  • Output: [Sub-Query A, Sub-Query B] for comparative/multipart, or [Query] for single-focus
       │
       ├───────────────────────────────────────────┐
       ▼                                           ▼
[Sub-Query 1 Hybrid Retrieval]              [Sub-Query 2 Hybrid Retrieval]
  • Dense (Qwen3-Embedding-0.6B)              • Dense (Qwen3-Embedding-0.6B)
  • BM25 Okapi                                • BM25 Okapi
  • RRF (k=60) ➔ Top-20 candidates            • RRF (k=60) ➔ Top-20 candidates
       │                                           │
       └─────────────────────┬─────────────────────┘
                             ▼
         [Balanced Round-Robin Interleaving]
           • Module: src/retrieval/hybrid.py (search_multi_query)
           • Interleaves candidates across sub-queries (depth 0, 1, 2...)
           • Tracks sub_query_sources and boosts multi-query consensus
           • Output: Top-25 balanced candidate pool
                             │
                             ▼
         [Two-Stage Cross-Encoder Reranker with Entity Quota]
           • Model: cross-encoder/ms-marco-MiniLM-L-6-v2
           • Module: src/retrieval/reranker.py
           • Two-Stage Rank Fusion: 0.60 * RRF(rrf_rank) + 0.40 * RRF(reranker_rank)
           • Enforces minimum quota: at least 2 chunks per sub-query in Top-8
                             │
                             ▼
         [Dynamic Context Budgeting]
           • Module: src/rag.py
           • Single-focus queries: Top-5 context
           • Comparative / multi-part queries: Top-8 context
                             │
                             ▼
         [6-Point Evidence-Grounded Generator]
           • Model: qwen2.5:7b via Ollama (localhost:11434)
           • Strict grounding, exact technical terminology, and grounded citations
                             │
                             ▼
         [Final Answer + Grounded Citations]
```

---

## 2. Component Specifications

| Component | Setting / Parameter | Implementation File |
|---|---|---|
| **Query Decomposer** | Syntactic regex rules + `qwen2.5:3b` fallback | `src/retrieval/query_transform.py` |
| **Dense Embeddings** | `Qwen/Qwen3-Embedding-0.6B` (1024d) | `src/retrieval/vectordb.py` |
| **Vector DB** | Qdrant (Cosine, disk `data/qdrant`, collection `scientific_papers`) | `src/retrieval/vectordb.py` |
| **BM25 Parameters** | $k_1 = 1.5, b = 0.75$, tokenized via `\b\w+\b` | `src/retrieval/bm25.py` |
| **Candidate Pool** | Top-25 balanced candidates | `src/retrieval/hybrid.py` |
| **Reranker Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `src/retrieval/reranker.py` |
| **Two-Stage Fusion**| $w_{\text{rrf}}=0.60, w_{\text{reranker}}=0.40, k=60$ | `src/retrieval/reranker.py` |
| **Entity Quota** | $\min(2, \lfloor K / 4 \rfloor)$ chunks per sub-query | `src/retrieval/reranker.py` |
| **Context Budget** | Dynamic: $K=5$ (single query), $K=8$ (multi-query) | `src/rag.py` |
| **Generator Model** | `qwen2.5:7b` via Ollama (`Q4_K_M`, 4.7 GB) | `src/rag.py` |

---

## 3. Benchmark Performance Metrics (40 Questions)

| Metric | V1 Baseline | V2.0 Baseline | V2.1 Baseline | V3 Final |
|---|:---:|:---:|:---:|:---:|
| **Hit@1** | 36.1% | 55.6% | 55.6% | **58.3%** |
| **Hit@5** | 77.8% | 83.3% | 83.3% | **83.3%** |
| **Hit@8** | N/A | N/A | N/A | **86.1%** |
| **Recall@1** | 0.1755 | 0.3319 | 0.3319 | **0.3389** |
| **Recall@5** | 0.5037 | 0.6472 | 0.6472 | **0.6542** |
| **Recall@8** | N/A | N/A | N/A | **0.6736** |
| **MRR** | 0.5331 | 0.6898 | 0.6898 | **0.6938** |
| **Mean Correctness (0–2)** | 1.06 / 2.0 | 0.9167 / 2.0 | 1.1389 / 2.0 | **1.2222 / 2.0** |
| **Fully Correct %** | 47.2% | 27.8% | 41.7% | **58.3%** |
| **Partially Correct %** | 11.1% | 36.1% | 30.6% | **5.6%** |
| **Incorrect %** | 41.7% | 36.1% | 27.8% | **36.1%** |
| **Mean Faithfulness (0–2)** | 0.86 / 2.0 | 1.0833 / 2.0 | 1.4167 / 2.0 | **1.4167 / 2.0** |
| **Fully Grounded %** | 38.9% | 52.8% | 66.7% | **61.1%** |
| **Mean Keyword Overlap** | 0.3800 | 0.4584 | 0.5292 | **0.5354 (BEST)** |
| **Citation Correctness** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Abstention Accuracy** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Hallucination Rate** | 0.0% | 0.0% | 0.0% | **0.0%** |
| **P50 Total Latency** | 2.80s | 2.62s | 13.27s | **16.45s** |

---

## 4. Category-Level Performance Breakdown

- **Direct Factual** (6 questions): Correctness = **1.67 / 2.0**, Fully Correct = **83.3%**, Faithfulness = **2.00 / 2.0**, Keyword Overlap = **0.67**.
- **Technical Concept** (6 questions): Correctness = **1.83 / 2.0**, Fully Correct = **83.3%**, Incorrect = **0.0%**, Faithfulness = **2.00 / 2.0**, Keyword Overlap = **0.58**.
- **Multi-Hop Reasoning** (5 questions): Correctness = **1.60 / 2.0**, Fully Correct = **80.0%**, Faithfulness = **2.00 / 2.0**, Keyword Overlap = **0.41**.
- **Numerical / Experimental** (4 questions): Correctness = **1.50 / 2.0**, Fully Correct = **75.0%**, Keyword Overlap = **0.74**.
- **Exact Terminology** (5 questions): Correctness = **0.80 / 2.0**, Fully Correct = **40.0%**, Keyword Overlap = **0.46**.
- **Figure Related** (5 questions): Correctness = **1.00 / 2.0**, Fully Correct = **40.0%**, Keyword Overlap = **0.56**.
- **Comparison** (5 questions): Recall@8 = **0.5533** (+19.0% absolute increase), Keyword Overlap = **0.33**.
- **Unanswerable** (4 questions): **100.0% Abstention**, 0% Hallucination.

---

## 5. How to Reproduce V3 Results

```powershell
# 1. Checkout V3 tag
git checkout v3-multi-query

# 2. Run full evaluation benchmark
.venv\Scripts\python.exe evaluation/run_v3_experiments.py --model qwen2.5:7b

# 3. Interactive CLI query
.venv\Scripts\python.exe src/rag.py "How do post-graph-rag and SelfGraphRAG differ in their approach to handling entity deduplication and graph extraction quality?"
```
