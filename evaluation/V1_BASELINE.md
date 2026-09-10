# V1 Baseline Specification & Benchmark Results

**Frozen At**: 2026-09-10  
**Git Tag**: `v1-baseline`  
**Git Commit**: `a2cf7131fefb7e5df10c82b526160f80fa956f82`  
**Evaluation Results**: `evaluation/results/baseline_results.json`  
**Evaluation Report**: `evaluation/BASELINE_REPORT.md`  

---

## 1. System Architecture & Pipeline Flow

The V1 system implements a pure dense vector Retrieval-Augmented Generation pipeline:

```
[User Question]
       │
       ▼
[Dense Query Encoding]
  • Model: Qwen/Qwen3-Embedding-0.6B (PyTorch / CUDA)
  • Dimensions: 1024
  • Max Sequence Length: 4096 tokens
       │
       ▼
[Vector Database ANN Search]
  • Engine: Qdrant (Local disk at data/qdrant)
  • Collection: scientific_papers
  • Metric: Cosine Distance
  • Top-K: 5 (retrieved without reranking or filtering)
       │
       ▼
[Context Block Formatting]
  • Format: "[{CHUNK_TYPE}] Paper: {paper_id} | Section: {section} [ | Image: {path}]\n{content}"
  • Delimiter: "\n\n---\n\n"
       │
       ▼
[LLM Prompt Assembly]
  • System Prompt: Evidence-grounded scientific persona (cite Paper ID & Section)
  • User Prompt: "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
       │
       ▼
[Local LLM Generation]
  • Model: qwen2.5:3b via Ollama (localhost:11434)
  • Parameters: Default temperature, context window 32,768 tokens
       │
       ▼
[Final Output & Citations]
```

---

## 2. Ingestion & Chunking Approach

- **Input Papers**: 10 scientific papers in PDF format (`data/raw/2608.*.pdf`).
- **Parsing**: Single-pass conversion using `Docling` to generate Markdown and extract picture bounding boxes.
- **Multimodal Captioning**: Figures captioned with `llava:13b` via Ollama using academic prompt grounded with paper title, section name, caption, and in-text citation mentions.
- **Chunking Strategy**:
  - Section-aware splitting on Markdown level-2 headers (`## Section Name`).
  - Noise sections (`references`, `bibliography`) filtered out.
  - Short sections ($\le 1000$ words) kept intact to preserve context.
  - Long sections ($> 1000$ words) sub-split with `RecursiveCharacterTextSplitter` (chunk size: 3000 chars, overlap: 300 chars).
  - Figure chunks formatted with title, caption, manuscript context, and detailed visual breakdown.
- **Corpus Statistics**:
  - Total chunks: **374**
  - Text chunks: **319** (chunk_id 0–318)
  - Multimodal figure chunks: **55** (chunk_id 319–373)
  - Processed file: `data/processed/chunks.json` (789 KB)

---

## 3. Retrieval & Generation Configuration

| Parameter | Value | Source File |
|---|---|---|
| Retrieval Method | Pure Dense (Approximate Nearest Neighbors) | `src/retrieval/vectordb.py` |
| Embedding Model | `Qwen/Qwen3-Embedding-0.6B` | `src/retrieval/vectordb.py` |
| Vector Dimension | 1024 | `src/retrieval/vectordb.py` |
| Distance Metric | Cosine (`Distance.COSINE`) | `src/retrieval/vectordb.py` |
| Vector DB Path | `data/qdrant` | `src/retrieval/vectordb.py` |
| Collection Name | `scientific_papers` | `src/retrieval/vectordb.py` |
| Retrieval Top-K | 5 | `src/rag.py` |
| Query Filters | None | `src/retrieval/vectordb.py` |
| Lexical Search | None | N/A |
| Reranker | None | N/A |
| Generator Model | `qwen2.5:3b` | `src/rag.py` |
| LLM Provider | Ollama (local) | `src/rag.py` |
| System Prompt | Evidence-grounded research assistant | `src/rag.py` |

---

## 4. Evaluation Dataset (`evaluation/questions.json`)

- **Total Questions**: 40
- **Answerable Questions**: 36 (90%)
- **Unanswerable Questions**: 4 (10%)
- **Paper Coverage**: 10 / 10 papers
- **Difficulty Breakdown**:
  - Easy: 10 (25%)
  - Medium: 20 (50%)
  - Hard: 10 (25%)
- **Question Categories**:
  - `direct_factual`: 6
  - `technical_concept`: 6
  - `exact_terminology`: 5
  - `comparison`: 5
  - `multi_hop_reasoning`: 5
  - `figure_related`: 5
  - `numerical_experimental`: 4
  - `unanswerable`: 4
- **Ground-Truth Chunks**: 49 unique chunk IDs mapped directly to `data/processed/chunks.json`.

---

## 5. V1 Baseline Performance Summary

### Aggregate Metrics

| Metric | V1 Score | Definition / Target |
|---|---|---|
| **Hit@1** | 36.1% | First retrieved chunk is ground-truth |
| **Hit@3** | 63.9% | At least 1 ground-truth chunk in top-3 |
| **Hit@5** | **77.8%** | Default RAG context window hit rate |
| **Hit@10** | **88.9%** | Upper retrieval ceiling |
| **Recall@1** | 0.1755 | Fraction of relevant chunks at rank 1 |
| **Recall@3** | 0.3454 | Fraction of relevant chunks in top-3 |
| **Recall@5** | **0.5037** | Fraction of relevant chunks in top-5 |
| **Recall@10**| **0.6931** | Fraction of relevant chunks in top-10 |
| **MRR** | **0.5331** | Mean Reciprocal Rank of first relevant chunk |
| **Mean Correctness** | **1.06 / 2.0** | Rubric: 0=Incorrect, 1=Partial, 2=Fully Correct |
| Fully Correct % | 47.2% | 17 of 36 answerable questions |
| Partially Correct %| 11.1% | 4 of 36 answerable questions |
| Incorrect % | 41.7% | 15 of 36 answerable questions |
| **Mean Faithfulness**| **0.86 / 2.0** | Grounding rubric: 0=unsupported, 2=fully grounded |
| Fully Grounded % | 38.9% | 14 of 36 answerable questions |
| **Citation Correctness**| **100.0%** | $\ge 1$ relevant paper in top-5 context |
| **Citation Completeness**| **94.4%** | ALL relevant papers in top-5 context |
| **Abstention Accuracy**| **100.0%** | Correctly refused all 4 unanswerable questions |
| **Hallucination Rate** | **0.0%** | 0 unanswerable questions hallucinated |
| **Mean Latency** | **3.4s** | Total end-to-end latency per query |
| P50 / P95 Latency | 2.8s / 7.0s | P95 on complex figure queries |

### Category Performance Highlights
- **Figure-related**: 100% Hit@5, 1.4/2.0 correctness — Multimodal visual chunking performs exceptionally well.
- **Numerical / Experimental**: 100% Hit@5, 1.5/2.0 correctness — Exact tables and stats are well captured.
- **Exact Terminology**: **40.0% Hit@5**, **0.1650 MRR** — Primary retrieval bottleneck in V1.
- **Multi-hop / Comparison**: Recall@5 is only **0.2367 to 0.3233** — Insufficient chunks retrieved for cross-section synthesis.

---

## 6. How to Reproduce V1 Results

```powershell
# Checkout V1 tag
git checkout v1-baseline

# Run evaluation runner
.venv\Scripts\python.exe evaluation/run_baseline.py
```
