# V2.1 Baseline Specification & Benchmark Results

**Frozen At**: 2026-09-10  
**Git Tag**: `v2.1-generator-improvement`  
**Git Commit**: `3bd0def`  
**Evaluation Results**: `evaluation/results/v2_1_results.json`, `evaluation/results/v2_1_summary.csv`  
**Experiment Log**: `evaluation/V2_1_EXPERIMENT_LOG.md`  
**Comparison Report**: `evaluation/V2_0_V2_1_COMPARISON.md`  

---

## 1. System Architecture & Pipeline Flow

The V2.1 system builds upon the frozen V2.0 two-stage hybrid retrieval stack and introduces a high-capacity generator (`qwen2.5:7b`) with a 6-point evidence-grounded prompt:

```
[User Query]
       │
       ├───────────────────────────────────────────┐
       ▼                                           ▼
[Dense Query Encoding]                      [BM25 Tokenization]
  • Model: Qwen/Qwen3-Embedding-0.6B          • Regex: \b\w+\b (lower)
  • Dimensions: 1024                          • Params: k1=1.5, b=0.75
       │                                           │
       ▼                                           ▼
[Qdrant ANN Search]                         [BM25 Inverted Index]
  • Top-25 candidates                         • Top-25 candidates
       │                                           │
       └─────────────────────┬─────────────────────┘
                             ▼
               [Stage 1: Reciprocal Rank Fusion]
                 • Score: 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)
                 • Output: Top-25 fused candidates
                             │
                             ▼
               [Stage 2: Cross-Encoder Reranking]
                 • Model: cross-encoder/ms-marco-MiniLM-L-6-v2
                 • Pairs: (query, candidate_chunk_text)
                 • Softmax logits across top-25 pool
                             │
                             ▼
               [Two-Stage Rank Fusion]
                 • Score = 0.60 * RRF_Score(rank_rrf) + 0.40 * RRF_Score(rank_reranker)
                 • Output: Top-5 evidence chunks
                             │
                             ▼
               [Top-5 Context Selection]
                 • Format: "[{CHUNK_TYPE}] Paper: {paper_id} | Section: {section}\n{content}"
                             │
                             ▼
               [6-Point Evidence-Grounded Generator]
                 • Model: qwen2.5:7b via Ollama (localhost:11434)
                 • Enforces: Grounding, terminology preservation, complete coverage, 
                             standardized citations, multimodal grounding, zero CoT overhead
                             │
                             ▼
               [Final Answer + Grounded Citations]
```

---

## 2. Ingestion & Retrieval Components (Frozen from V2.0)

| Component | Setting / Parameter | Implementation File |
|---|---|---|
| **Corpus Source** | 10 scientific papers in PDF format (`data/raw/2608.*.pdf`) | `data/raw/` |
| **Total Chunks** | 374 chunks (319 text, 55 multimodal figure chunks) | `data/processed/chunks.json` |
| **Dense Embeddings** | `Qwen/Qwen3-Embedding-0.6B` (1024d) | `src/retrieval/vectordb.py` |
| **Vector DB** | Qdrant (Cosine similarity, local disk storage at `data/qdrant`) | `src/retrieval/vectordb.py` |
| **Dense Candidate Pool** | Top-25 (`limit=25`) | `src/retrieval/hybrid.py` |
| **BM25 Parameters** | $k_1 = 1.5, b = 0.75$, tokenized via `\b\w+\b` | `src/retrieval/bm25.py` |
| **BM25 Candidate Pool** | Top-25 | `src/retrieval/hybrid.py` |
| **Stage 1 Fusion** | Reciprocal Rank Fusion ($k=60$) | `src/retrieval/hybrid.py` |
| **Cross-Encoder Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `src/retrieval/reranker.py` |
| **Stage 2 Fusion** | Two-Stage Rank Fusion ($w_{\text{rrf}}=0.60, w_{\text{reranker}}=0.40, k=60$) | `src/retrieval/reranker.py` |
| **Context Selection** | Top-5 chunks ($K=5$) | `src/rag.py` |

---

## 3. V2.1 Generation Layer & Prompt Specification

### Generator Model Configuration
- **Model**: `qwen2.5:7b` via Ollama (`localhost:11434`)
- **Quantization**: `Q4_K_M` (4.7 GB model size)
- **VRAM Consumption**: ~6.5 GB during inference
- **Fallback / Low-latency Model**: `qwen2.5:3b` selectable via `--model qwen2.5:3b`

### 6-Point Evidence-Grounded System Prompt
```text
You are an expert scientific research assistant. Your task is to provide precise, rigorous, and evidence-grounded answers based strictly on the provided scientific literature context.

Strictly adhere to the following rules:
1. Evidence Grounding & Abstention:
   - Rely ONLY on clear facts directly mentioned in the context. Do NOT extrapolate, speculate, or bring in outside knowledge.
   - If the context does not contain sufficient information to answer the question completely, you MUST state clearly: "The provided context does not contain sufficient information to answer this question." Do not attempt to guess.

2. Exact Technical Terminology & Numerical Fidelity:
   - Preserve exact technical terms, acronyms, mathematical formulas, and benchmark scores from the context.
   - Do not paraphrase or approximate specific technical definitions or numerical metrics.

3. Complete Coverage of Multi-Part & Comparative Questions:
   - When a question asks to compare two or more entities, models, or datasets, you MUST address each entity explicitly.
   - If information is present for one entity but missing for another, state the facts for the available entity and explicitly note the lack of context for the other.

4. Citation Format:
   - For every factual assertion, cite the source using the exact format: [Paper: <paper_id> | Section: <section_name>].

5. Visual & Figure Grounding:
   - When context includes figure captions, diagrams, or visual summaries, cite the figure and explain its architectural flow or empirical finding directly.

6. Output Format & Efficiency:
   - Output your answer directly without conversational filler (such as "Certainly!", "Sure!").
   - Do NOT output chain-of-thought reasoning tokens or meta-commentary.
```

---

## 4. Evaluation Dataset & Frozen Retrieval Metrics

- **Dataset**: `evaluation/questions.json` (40 questions, 36 answerable, 4 unanswerable canaries).
- **Retrieval Metrics (Frozen across all runs)**:
  - **Hit@1**: 55.56%
  - **Hit@3**: 77.78%
  - **Hit@5**: 83.33%
  - **Hit@10**: 91.67%
  - **Recall@1**: 0.3319
  - **Recall@3**: 0.5444
  - **Recall@5**: 0.6472
  - **Recall@10**: 0.7417
  - **MRR**: 0.6898

---

## 5. Official 2×2 Factorial Ablation Results

Under identical cached Top-5 context, the 4 configurations yielded the following benchmark metrics:

| Metric | V2.0 Baseline<br>*(3B + Old Prompt)* | Ablation A<br>*(3B + Improved Prompt)* | Ablation B<br>*(7B + Old Prompt)* | V2.1 Full System<br>*(7B + Improved Prompt)* | Delta<br>*(V2.1 vs V2.0)* |
|---|:---:|:---:|:---:|:---:|:---:|
| **Mean Correctness (0–2)** | 0.9167 | **1.1667** | 1.0278 | **1.1389** | **+0.2222** (+24.2%) |
| Fully Correct % | 27.8% | **41.7%** | 30.6% | **41.7%** | **+13.9%** (+50.0% rel.) |
| Partially Correct % | 36.1% | 33.3% | 41.7% | 30.6% | -5.5% |
| Incorrect % | 36.1% | **25.0%** | 27.8% | **27.8%** | **-8.3%** (-23.0% rel.) |
| **Mean Faithfulness (0–2)** | 1.0833 | **1.4167** | 0.9722 | **1.4167** | **+0.3334** (+30.8%) |
| Fully Grounded % | 52.8% | **69.4%** | 44.4% | **66.7%** | **+13.9%** (+26.3% rel.) |
| **Mean Keyword Overlap** | 0.4584 | 0.4840 | 0.5043 | **0.5292 (BEST)** | **+0.0708** (+15.4%) |
| **Citation Correctness** | 100.0% | 100.0% | 100.0% | **100.0%** | **0.0%** (Perfect) |
| **Citation Completeness** | 94.4% | 94.4% | 94.4% | **94.4%** | **0.0%** |
| **Abstention Accuracy** | 100.0% | 100.0% | 100.0% | **100.0%** | **0.0%** (Perfect) |
| **Hallucination Rate** | 0.0% | 0.0% | 0.0% | **0.0%** | **0.0%** (Zero) |
| **Mean Generation Latency** | 2.95s | 4.43s | 10.59s | **13.41s** | +10.46s |
| **P50 Total Latency** | 2.62s | 4.26s | 10.98s | **13.27s** | +10.65s |
| **P95 Total Latency** | 7.16s | 9.59s | 16.62s | **27.37s** | +20.21s |

---

## 6. Category-by-Category Performance (V2.1 Full System)

| Category | Count | Correctness | Faithfulness | Keyword Overlap | Incorrect % | Key Result |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Direct Factual** | 6 | **1.50 / 2.0** | **2.00 / 2.0** | **0.68** | **0.0%** | 100% grounded answers, zero factual errors. |
| **Technical Concept** | 6 | **1.67 / 2.0** | **1.67 / 2.0** | **0.57** | **0.0%** | Deep architectural mechanisms accurately explained. |
| **Exact Terminology** | 5 | **1.20 / 2.0** | **1.40 / 2.0** | **0.52** | 20.0% | Acronyms and specific phrases quoted verbatim. |
| **Numerical / Experimental** | 4 | **1.75 / 2.0** | **0.50 / 2.0** | **0.74** | **0.0%** | Highest lexical precision; tabular figures retained. |
| **Multi-Hop Reasoning** | 5 | **1.00 / 2.0** | **2.00 / 2.0** | **0.45** | 20.0% | 100% grounded across synthesis steps. |
| **Figure Related** | 5 | **0.80 / 2.0** | **1.40 / 2.0** | **0.49** | 20.0% | Multimodal OCR and caption analysis grounded. |
| **Comparison** | 5 | **0.00 / 2.0** | **0.60 / 2.0** | **0.27** | 100.0% | **Retrieval bottleneck**: Recall@5 = 0.3633. |
| **Unanswerable** | 4 | **N/A** | **N/A** | **N/A** | **0.0%** | 100% abstention on out-of-corpus canaries. |

---

## 7. How to Reproduce V2.1 Results

```powershell
# 1. Checkout V2.1 tag
git checkout v2.1-generator-improvement

# 2. Run the full benchmark for V2.1 Full System
.venv\Scripts\python.exe evaluation/run_v2_1_experiments.py --experiments v2_1_7b_improved_prompt

# 3. Interactive CLI query
.venv\Scripts\python.exe src/rag.py "What vector database and embedding model are used in post-graph-rag?"
```

