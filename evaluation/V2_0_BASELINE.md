# V2.0 Baseline Specification & Benchmark Results

**Frozen At**: 2026-09-10  
**Git Tag**: `v2.0-baseline`  
**Git Commit**: `f1def46` (committed as part of Two-Stage Rank Fusion implementation)  
**Evaluation Results**: `evaluation/results/v2_experiment_results.json`  
**Comparison Report**: `evaluation/V1_V2_COMPARISON.md`  

---

## 1. System Architecture & Pipeline Flow

The V2.0 system implements a multi-stage hybrid retrieval and cross-encoder reranking pipeline with two-stage rank fusion:

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
                 • Preserves Lexical/Dense Consensus while surfacing specific evidence
                             │
                             ▼
               [Top-5 Context Selection]
                 • Top-5 chunks formatted with headers
                 • Format: "[{CHUNK_TYPE}] Paper: {paper_id} | Section: {section}\n{content}"
                             │
                             ▼
               [Local LLM Generation]
                 • Model: qwen2.5:3b via Ollama (localhost:11434)
                 • Persona: Evidence-grounded scientific research assistant
                             │
                             ▼
               [Final Answer + Grounded Citations]
```

---

## 2. Ingestion & Chunking Specification (Frozen from V1)

- **Corpus Source**: 10 scientific papers in PDF format (`data/raw/2608.*.pdf`).
- **Parsing Engine**: `Docling` Markdown extraction with bounding box detection.
- **Multimodal Visual Captions**: Captioned via `llava:13b` using contextual prompt (paper title, section name, caption, and in-text context).
- **Chunking Logic**:
  - Split on Markdown level-2 headers (`## Section Name`).
  - Noise sections (`references`, `bibliography`) filtered out.
  - Sections $\le 1000$ words kept whole.
  - Sections $> 1000$ words split with `RecursiveCharacterTextSplitter` (size: 3000 chars, overlap: 300 chars).
- **Total Corpus Size**: **374 chunks** (319 text chunks, 55 multimodal figure chunks).
- **Processed File**: `data/processed/chunks.json` (789 KB).

---

## 3. Retrieval & Reranker Configuration Details

| Component | Setting / Parameter | Implementation File |
|---|---|---|
| **Dense Embeddings** | `Qwen/Qwen3-Embedding-0.6B` (1024d) | `src/retrieval/vectordb.py` |
| **Vector DB** | Qdrant (Cosine, local disk `data/qdrant`) | `src/retrieval/vectordb.py` |
| **Vector Collection** | `scientific_papers` | `src/retrieval/vectordb.py` |
| **Dense Candidate Pool** | Top-25 (`limit=25`) | `src/retrieval/hybrid.py` |
| **BM25 Index** | Pure Python `BM25Index` | `src/retrieval/bm25.py` |
| **BM25 Parameters** | $k_1 = 1.5, b = 0.75$ | `src/retrieval/bm25.py` |
| **BM25 Tokenization** | `re.findall(r'\b\w+\b', text.lower())` | `src/retrieval/bm25.py` |
| **BM25 Candidate Pool** | Top-25 | `src/retrieval/hybrid.py` |
| **RRF Constant ($k$)** | $60$ | `src/retrieval/hybrid.py` |
| **Cross-Encoder Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `src/retrieval/reranker.py` |
| **Reranker Batch Size** | 16 | `src/retrieval/reranker.py` |
| **Reranker Candidate Pool**| Top-25 from RRF | `src/retrieval/reranker.py` |
| **Fusion Algorithm** | Two-Stage Rank Fusion ($w_{\text{rrf}}=0.60, w_{\text{reranker}}=0.40, k=60$) | `src/retrieval/reranker.py` |
| **Final Context Budget** | Top-5 chunks ($K=5$) | `src/rag.py` |
| **Context Window Format**| `[{CHUNK_TYPE}] Paper: {paper_id} \| Section: {section} [\| Image: {path}]\n{content}` | `src/rag.py` |

---

## 4. Generator & Prompt Specification (Frozen from V1)

- **LLM Model**: `qwen2.5:3b` via Ollama (localhost:11434).
- **System Prompt**:
  ```text
  You are an expert scientific research assistant.
  Answer questions accurately and concisely using ONLY the provided context.
  Always cite the source paper ID and section name when making assertions.
  If the context does not contain enough information to answer, state clearly: "Based on the provided context, I cannot answer this question."
  Do NOT speculate, make assumptions, or bring in outside knowledge not grounded in the context.
  ```
- **User Prompt Structure**:
  ```text
  Context:
  {formatted_context_blocks}

  Question: {query}

  Answer:
  ```

---

## 5. Evaluation Dataset & Methodology

- **Dataset**: `evaluation/questions.json` (Frozen, 40 questions).
- **Question Distribution**:
  - Direct Factual: 6
  - Technical Concept: 6
  - Exact Terminology: 5
  - Comparison: 5
  - Multi-hop Reasoning: 5
  - Figure Related: 5
  - Numerical / Experimental: 4
  - Unanswerable (Refusal Canaries): 4
- **Scoring Calibrations**:
  - Dual-signal verification: Exact Ground-Truth Keyword & Entity check combined with LLM Judge evaluation.
  - Normalized rubric for Correctness (0 = Incorrect, 1 = Partially Correct, 2 = Fully Correct).
  - Normalized rubric for Faithfulness (0 = Hallucinated/Unsupported, 1 = Partially Grounded, 2 = Fully Grounded).

---

## 6. Official V2.0 Performance Results

### Aggregate Metrics Comparison

| Metric | V1 Baseline (Dense) | V2.0 Hybrid + Two-Stage Reranker | Delta (V2.0 vs V1) |
|---|---|---|---|
| **Hit@1** | 36.1% | **55.6%** | **+19.4%** |
| **Hit@3** | 63.9% | **77.8%** | **+13.9%** |
| **Hit@5** | 77.8% | **83.3%** | **+5.5%** |
| **Hit@10** | 88.9% | **91.7%** | **+2.8%** |
| **Recall@1** | 0.1755 | **0.3319** | **+0.1564** |
| **Recall@3** | 0.3454 | **0.5444** | **+0.1990** |
| **Recall@5** | 0.5037 | **0.6472** | **+0.1435** |
| **Recall@10** | 0.6931 | **0.7417** | **+0.0486** |
| **MRR** | 0.5331 | **0.6898** | **+0.1567** |
| **Mean Correctness** | 1.06 / 2.0 | **1.11 / 2.0** | **+0.06** |
| Fully Correct % | 47.2% | **36.1%** | -11.1% |
| Partially Correct % | 11.1% | **38.9%** | +27.8% |
| Incorrect % | 41.7% | **25.0%** | **-16.7%** |
| **Mean Faithfulness** | 0.86 / 2.0 | **1.33 / 2.0** | **+0.47** |
| Fully Grounded % | 38.9% | **52.8%** | **+13.9%** |
| **Citation Correctness** | 100.0% | **100.0%** | 0.0% |
| **Citation Completeness** | 94.4% | **94.4%** | 0.0% |
| **Abstention Accuracy** | 100.0% | **100.0%** | 0.0% |
| **Hallucination Rate** | 0.0% | **0.0%** | 0.0% |
| **Mean Latency** | 3.39s | **4.45s** | +1.06s |
| **P50 Latency** | 2.80s | **3.90s** | +1.10s |
| **P95 Latency** | 7.02s | **8.59s** | +1.57s |

---

## 7. How to Reproduce V2.0 Results

```powershell
# Checkout V2.0 tag
git checkout v2.0-baseline

# Run full evaluation benchmark across all 40 questions
.venv\Scripts\python.exe evaluation/run_experiments.py
```
