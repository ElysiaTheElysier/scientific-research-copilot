# Scientific Research Copilot — Baseline Evaluation Dataset

## 1. Dataset Purpose

This evaluation dataset is a manually curated, rigorously grounded benchmark designed specifically to evaluate the performance, retrieval quality, and generation faithfulness of the **baseline Scientific Research Copilot RAG pipeline**.

Before implementing enhancements such as hybrid search (BM25 + Dense), cross-encoder rerankers, query rewriting, graph expansion, or automated evaluation frameworks (RAGAS), it is critical to measure the raw baseline system:
- **Embedding Model**: `Qwen/Qwen3-Embedding-0.6B` (1024-dimensional dense vectors)
- **Vector Database**: Local persistent Qdrant collection (`scientific_papers`, Cosine distance)
- **Chunk Corpus**: 374 chunks (319 text chunks + 55 multimodal figure chunks) across 10 recent scientific papers in arXiv cs.AI/cs.IR/q-bio
- **Baseline Generator**: `qwen2.5:3b` via Ollama

This dataset tests realistic failure modes of scientific RAG systems, including terminology confusion, multi-hop reasoning disconnects, cross-document discrepancies, figure understanding, sensitivity to missing context, and refusal to hallucinate when evidence is absent.

---

## 2. Corpus Overview (10 Source Papers)

The dataset references exclusively the 10 scientific papers ingested in `data/raw/` and processed into `data/processed/chunks.json`:

| Paper ID | Short Name | Full Title | Text Chunks | Figure Chunks |
|---|---|---|---|---|
| `2608.24921v1` | **post-graph-rag** | *post-graph-rag: A PostgreSQL-Native Graph RAG Engine with Extraction-Time Quality Gates and a Temporal Relation Model* | 0–34 | 319–324 |
| `2608.24977v2` | **RAG Attacks Survey** | *Retrieved But Not Reliable: A Survey on Attacks and Defenses in Retrieval-Augmented Generation* | 35–81 | 325–331 |
| `2608.25123v1` | **SelfGraphRAG** | *SelfGraphRAG: Bridging the Supervision Gap in Graph-Based RAG with Synthetic QA Generation* | 82–108 | 332–333 |
| `2608.25466v1` | **Homo-RAG** | *Homo-RAG: Homology-Guided Retrieval-Augmented Generation for Cross-Species Gene Function Prediction* | 109–140 | 334–342 |
| `2608.25618v1` | **AWM** | *AWM: Answerable Working Memory for Long-Document VQA Agents* | 141–178 | 343–346 |
| `2608.25717v1` | **Geo-bias in RAG** | *When RAG Fails to Equalize: Geo-bias in Factual Question Answering over Public Companies* | 179–223 | 347–356 |
| `2608.25986v1` | **CEMMKG** | *Multi-Granularity Context-Enhanced RAG over Multimodal Knowledge Graphs* | 224–255 | 357–360 |
| `2608.26194v1` | **STeReO** | *A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers* | 256–270 | 361–362 |
| `2608.26379v1` | **Evidence-Aware Utility** | *Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG* | 271–291 | 363–366 |
| `2608.26385v1` | **Why RAGs Hallucinate** | *Why RAGs Hallucinate: Penalty-Aware Evaluation of Retrieval-Augmented Generation Systems with Knowledge-Gap Canaries* | 292–318 | 367 |

---

## 3. Question Categories and Taxonomy

The dataset contains exactly **40 questions** partitioned across 8 specialized evaluation categories and 3 difficulty levels:

### 3.1 Category Breakdown
1. **Direct factual questions (6 questions: `q001`–`q006`)**:
   Tests the baseline retriever's ability to fetch explicit facts, databases, model backbones, and pipeline stages clearly stated in the text.
2. **Technical concept questions (6 questions: `q007`–`q012`)**:
   Tests the system's ability to retrieve and synthesize deep architectural principles, security threat models, and algorithmic designs (e.g. memory-only answerability, validity-composition problem, multi-granularity local/global context).
3. **Exact terminology / acronym questions (5 questions: `q013`–`q017`)**:
   Tests retrieval sensitivity to specific technical acronyms and domain terms (e.g. `MMC`, `STeReO`, `ECS`, `Inductive vs Deductive`, `Verbalize`).
4. **Comparison questions (5 questions: `q018`–`q022`)**:
   Tests multi-entity comparison both within a single paper (e.g. misleading vs distraction context, retrieval vs rerank defenses) and across different papers (e.g. post-graph-rag vs SelfGraphRAG entity handling; penalty-aware scoring vs answer-support utility).
5. **Multi-hop reasoning questions (5 questions: `q023`–`q027`)**:
   Requires piecing together evidence scattered across non-contiguous chunks or multiple papers (e.g. Homo-RAG 3-hop biological retrieval chain; AWM-GRPO joint advantage mechanics; ungrounded parametric leakage synthesis).
6. **Figure-related questions (5 questions: `q028`–`q032`)**:
   Specifically tests the retrieval and understanding of multimodal figure chunks, architectural diagrams, system charts, and metric curves generated during document ingestion.
7. **Numerical / experimental-result questions (4 questions: `q033`–`q036`)**:
   Tests precision retrieval of exact benchmark metrics, percentages, odds ratios, and ablation results from tables and text.
8. **Unanswerable / insufficient-evidence questions (4 questions: `q037`–`q040`)**:
   Tests the system's ability to detect knowledge gaps and abstain gracefully without hallucinating or leaking parametric pretraining artifacts.

### 3.2 Difficulty Distribution
- **Easy**: 10 questions (25.0%) — Direct single-hop lookups and clear definitions.
- **Medium**: 20 questions (50.0%) — Concepts requiring multi-sentence synthesis or specific structural details.
- **Hard**: 10 questions (25.0%) — Multi-hop reasoning, cross-paper comparisons, detailed quantitative figure breakdowns, and subtle unanswerable queries.

---

## 4. Ground-Truth Methodology

Every question in this dataset was curated against the ground-truth text chunks in `data/processed/chunks.json`:

1. **No Hallucinated Chunk IDs**:
   Every entry in `relevant_chunk_ids` corresponds to an exact, integer `chunk_id` present in `data/processed/chunks.json`.
2. **No Fabricated Evidence for Unanswerable Questions**:
   Questions tagged with `answerable: false` strictly have `relevant_chunk_ids: []`, `paper_ids: []`, `paper_titles: []`, `relevant_section: []`, `evidence_type: "none"`, and `expected_answer: null`.
3. **Multimodal Anchoring**:
   Every figure question references an existing image in `data/processed/images/` and the corresponding figure chunk in `chunks.json`.
4. **Cross-Paper Linkage**:
   For comparison and multi-hop questions spanning multiple papers (such as `q018`, `q020`, and `q027`), all participating `paper_ids` and their respective `relevant_chunk_ids` are explicitly recorded.

---

## 5. Schema Specification (`evaluation/questions.json`)

Each item in `evaluation/questions.json` adheres to the following structure:

```json
{
  "id": "q001",
  "question": "Which database management system and vector indexing extension does post-graph-rag use...",
  "question_type": "direct_factual",
  "answerable": true,
  "difficulty": "easy",
  "paper_ids": ["2608.24921v1"],
  "paper_titles": [
    "post-graph-rag: A PostgreSQL-Native Graph RAG Engine with Extraction-Time Quality Gates and a Temporal Relation Model"
  ],
  "relevant_section": ["Chandan Rajah", "1 Introduction"],
  "relevant_chunk_ids": [0, 1],
  "figure_ids": [],
  "figure_captions": [],
  "figure_image_path": [],
  "evidence_type": "text",
  "expected_answer": "post-graph-rag utilizes PostgreSQL as its unified database engine...",
  "evaluation_notes": "Look for explicit mentions of PostgreSQL and pgvector with HNSW indexing..."
}
```

For unanswerable questions:
```json
{
  "id": "q037",
  "question": "What learning rate schedule and number of training epochs are recommended...",
  "question_type": "unanswerable",
  "answerable": false,
  "difficulty": "easy",
  "paper_ids": [],
  "paper_titles": [],
  "relevant_section": [],
  "relevant_chunk_ids": [],
  "figure_ids": [],
  "figure_captions": [],
  "figure_image_path": [],
  "evidence_type": "none",
  "expected_answer": null,
  "evaluation_notes": "The corpus does not provide evidence for this question. The system should explicitly state that the available papers do not provide sufficient evidence."
}
```

---

## 6. How to Use This Dataset for RAG Evaluation

This dataset provides the necessary ground-truth annotations to benchmark the retrieval and generation stages independently.

### 6.1 Retrieval Metrics (on Answerable Questions, N = 36)

Let $Q$ be the set of answerable questions. For each question $q \in Q$:
- Let $R_k(q)$ be the set of top-$k$ retrieved chunk IDs returned by the vector store.
- Let $G(q)$ be the ground-truth set of `relevant_chunk_ids`.

#### 1. Hit@K
Measures whether at least one ground-truth chunk is retrieved within the top $k$ candidates:
$$\text{Hit@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( |R_k(q) \cap G(q)| > 0 \right)$$

#### 2. Recall@K
Measures the proportion of all relevant evidence chunks captured in the top $k$:
$$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|R_k(q) \cap G(q)|}{|G(q)|}$$

#### 3. Mean Reciprocal Rank (MRR)
Evaluates where the first relevant chunk appears in the ranked list:
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$
*(where $\text{rank}_1(q)$ is the 1-based rank of the first chunk in $R_k(q)$ that belongs to $G(q)$, or $\infty$ if none is found).*

---

### 6.2 Generation Metrics

#### 1. Citation Accuracy (Precision & Recall)
When the generator outputs citations (e.g. `[chunk 0]` or `[Paper 2608.24921v1]`):
- **Citation Precision**: Fraction of cited chunks that actually belong to $G(q)$.
- **Citation Recall**: Fraction of ground-truth chunks in $G(q)$ that were cited.

#### 2. Answer Faithfulness & Correctness
- **Faithfulness (Grounding)**: Checks whether all statements in the model's generated answer are supported by the retrieved passages, avoiding hallucinated claims.
- **Answer Correctness / Semantic Similarity**: Semantic overlap between the generated answer and `expected_answer` (using LLM-as-a-judge or embedding cosine similarity).

---

### 6.3 Abstention & Hallucination Resistance (on Unanswerable Questions, N = 4)

For unanswerable questions (`q037`–`q040`), the correct behavior is for the system to abstain and declare that the available corpus lacks sufficient evidence.

- **Abstention Rate**:
  $$\text{Abstention Rate} = \frac{\text{Number of unanswerable queries where system declined}}{\text{Total unanswerable queries (4)}}$$
- **Canary Violation / Hallucination Rate**:
  $$\text{Violation Rate} = \frac{\text{Number of unanswerable queries where system attempted an answer}}{\text{Total unanswerable queries (4)}}$$

Any generated factual assertion for unanswerable questions is considered an evaluation failure (parametric leakage / hallucination).

---

## 7. Execution and Verification

To verify the integrity and schema validity of this dataset against the ingested corpus at any time:

```bash
python scratch/generate_and_validate.py
```

This automated validator confirms:
1. All question IDs are unique.
2. Every referenced `chunk_id` exists in `data/processed/chunks.json`.
3. Every `paper_id` matches an ingested paper and markdown document in `data/processed/`.
4. Every `figure_id` exists in `data/processed/images/`.
5. Answerable questions have non-empty ground-truth references and expected answers.
6. Unanswerable questions contain no fabricated evidence.
7. Category, difficulty, and paper distributions match specifications.
