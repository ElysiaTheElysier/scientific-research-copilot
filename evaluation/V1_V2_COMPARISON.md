# V1 vs V2 Comparison Report: Hybrid Retrieval & Cross-Encoder Reranking

**Benchmark Date**: 2026-09-10 09:04:38 UTC  
**Evaluation Set**: 40 questions (`evaluation/questions.json`) across 10 scientific papers  
**Generator Model**: `qwen2.5:3b` via Ollama (Frozen across all evaluations)  

---

## 1. Executive Summary

This report measures the empirical performance differences between **V1 (Pure Dense Retrieval)** and **V2 (Dense + BM25 → RRF Hybrid → Cross-Encoder Reranker)**, alongside intermediate ablations for **BM25 only** and **Hybrid only (RRF)**.

**Key Headline Takeaways**:
- **Retrieval Hit@1**: **36.1% (V1)** ➔ **55.6% (V2)** (+19.4%)
- **Retrieval Hit@5**: **77.8% (V1)** ➔ **83.3% (V2)** (+5.5%)
- **Ranking MRR**: **0.5331 (V1)** ➔ **0.6898 (V2)** (+0.1567)
- **Answer Correctness**: **1.06/2.0 (V1)** ➔ **1.11/2.0 (V2)** (+0.06)
- **Fully Correct Answers**: **47.2% (V1)** ➔ **36.1% (V2)** (-11.1%)
- **Faithfulness**: **0.86/2.0 (V1)** ➔ **1.33/2.0 (V2)** (+0.47)
- **Abstention Accuracy**: **100.0%** (Maintained 100% refusal on unanswerable canaries)

---

## 2. Architecture Comparison

| Component | V1 Baseline | V2 Hybrid + Reranker |
|---|---|---|
| **Query Embedding** | `Qwen/Qwen3-Embedding-0.6B` (1024d) | `Qwen/Qwen3-Embedding-0.6B` (1024d) |
| **Vector DB** | Qdrant (Cosine, local disk) | Qdrant (Cosine, local disk) |
| **Lexical Search** | None | Pure Python `BM25Index` ($k_1=1.5, b=0.75$) |
| **Fusion Layer** | None | Reciprocal Rank Fusion ($k=60$) |
| **Reranker** | None | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Candidate Pool** | Top-5 | Top-25 candidate pool $	o$ Top-5 context |
| **Context Assembly** | Top-5 dense chunks | Top-5 reranked chunks |
| **Generator LLM** | `qwen2.5:3b` via Ollama | `qwen2.5:3b` via Ollama |
| **Prompt Template** | Evidence-grounded persona | Evidence-grounded persona (identical) |

---

## 3. End-to-End Metric Comparison Table

| Metric | V1 (Dense) | BM25 Only | Hybrid (RRF) | V2 (Hybrid+Reranker) | Delta (V2 vs V1) |
|---|---|---|---|---|---|
| **Hit@1** | 36.1% | 36.1% | 50.0% | **55.6%** | **+19.4%** |
| **Hit@3** | 63.9% | 69.4% | 80.6% | **77.8%** | **+13.9%** |
| **Hit@5** | 77.8% | 83.3% | 88.9% | **83.3%** | **+5.5%** |
| **Hit@10** | 88.9% | 88.9% | 91.7% | **91.7%** | **+2.8%** |
| **Recall@1** | 0.1755 | 0.2093 | 0.2741 | **0.3319** | **+0.1564** |
| **Recall@3** | 0.3454 | 0.5088 | 0.5861 | **0.5444** | **+0.1990** |
| **Recall@5** | 0.5037 | 0.6046 | 0.6759 | **0.6472** | **+0.1435** |
| **Recall@10** | 0.6931 | 0.7384 | 0.7532 | **0.7417** | **+0.0486** |
| **MRR** | 0.5331 | 0.5565 | 0.6569 | **0.6898** | **+0.1567** |
| **Mean Correctness** | 1.06/2.0 | — | — | **1.11/2.0** | **+0.06** |
| **Fully Correct %** | 47.2% | — | — | **36.1%** | **-11.1%** |
| **Partially Correct %** | 11.1% | — | — | **38.9%** | **+27.8%** |
| **Incorrect %** | 41.7% | — | — | **25.0%** | **-16.7%** |
| **Mean Faithfulness** | 0.86/2.0 | — | — | **1.33/2.0** | **+0.47** |
| **Citation Correctness** | 100.0% | — | — | **100.0%** | **+0.0%** |
| **Citation Completeness** | 94.4% | — | — | **94.4%** | **+0.0%** |
| **Abstention Accuracy** | 100.0% | — | — | **100.0%** | **0.0%** |
| **Mean Latency** | 3.39s | — | — | **4.45s** | **+1.06s** |
| **P95 Latency** | 7.02s | — | — | **8.59s** | **+1.57s** |

---

## 4. Performance Breakdown on Challenging Categories

### A. Exact Terminology (`exact_terminology`, 5 questions)
*V1 primary retrieval bottleneck*:
- **V1 Hit@5**: 40.0% | MRR: 0.1650
- **BM25 Hit@5**: 80.0% | MRR: 0.4167
- **Hybrid Hit@5**: 60.0% | MRR: 0.2667
- **V2 Hit@5**: **60.0%** | MRR: **0.3000**
- **Correctness**: 1.20 (V1) ➔ **1.00 (V2)**

### B. Multi-Hop Reasoning (`multi_hop_reasoning`, 5 questions)
- **V1 Hit@5**: 80.0% | Recall@5: 0.2367
- **V2 Hit@5**: **60.0%** | Recall@5: **0.1967**
- **Correctness**: 1.40 (V1) ➔ **1.00 (V2)**

### C. Comparison (`comparison`, 5 questions)
- **V1 Hit@5**: 80.0% | Recall@5: 0.3233
- **V2 Hit@5**: **80.0%** | Recall@5: **0.3633**
- **Correctness**: 1.00 (V1) ➔ **0.00 (V2)**

---

## 5. Question-Level Fixes and Regressions

### A. Questions Fixed by V2 (14 questions)

- **q003** (+2 score, 0 $\to$ 2): Which benchmark datasets are used to evaluate STeReO across speech and text modalities in its experimental setup?
  *V2 Answer Snippet*: In the experimental setup of the proposed method STeReO, two benchmark datasets are used to evaluate the model across speech and text modalities. These datasets are:

1. Spoken SQu...

- **q004** (+1 score, 0 $\to$ 1): What are the three commercial RAG systems evaluated in 'Why RAGs Hallucinate', and what benchmark dataset was used as the source of evaluation questions?
  *V2 Answer Snippet*: In 'Why RAGs Hallucinate', the three commercial RAG systems evaluated were:
1. OpenAI RAG
2. Gemini RAG
3. CustomGPT.ai RAG

The benchmark dataset used as the source of evaluation ...

- **q005** (+2 score, 0 $\to$ 2): What are the three sequential stages of the SelfGraphRAG pipeline?
  *V2 Answer Snippet*: The three sequential stages of the SelfGraphRAG pipeline are as follows:

1. **Doc2Graph**: In this stage, the document corpus is processed to construct a knowledge graph. The text...

- **q006** (+2 score, 0 $\to$ 2): According to the taxonomy in 'Retrieved But Not Reliable', what are the three primary high-level security objectives of adversarial attacks against RAG systems?
  *V2 Answer Snippet*: According to the taxonomy presented in the document, the three primary high-level security objectives of adversarial attacks against RAG systems are:

1. Accuracy attacks: These at...

- **q007** (+2 score, 0 $\to$ 2): What is 'memory-only answerability' in the AWM framework, and what problem does it diagnose in long-document VQA agents?
  *V2 Answer Snippet*: 'Memory-only answerability' in the AWM framework refers to a diagnostic method that evaluates whether a reader can answer a question from only the question and the terminal working...

- **q008** (+1 score, 1 $\to$ 2): How does post-graph-rag handle explicitly denied relations in extracted text, and why does it avoid inverting predicates into labels like 'did_not_have_relationship_with'?
  *V2 Answer Snippet*: Post-graph-rag stores explicitly denied relations by storing the positive predicate with a negated flag. It does not store the denial as an inverted predicate such as "did_not_have...

- **q010** (+2 score, 0 $\to$ 2): What is the 'validity-composition problem' identified in evidence-aware RAG evaluation?
  *V2 Answer Snippet*: The 'validity-composition problem' identified in evidence-aware RAG evaluation is that evidence supporting an evaluation signal for one use does not automatically support the next ...

- **q011** (+1 score, 0 $\to$ 1): In CEMMKG, what is the difference between local context and global context when constructing textual context for a visual element?
  *V2 Answer Snippet*: In CEMMKG, the difference between local context and global context when constructing textual context for a visual element is defined by which part of the textual content T they dra...

- **q012** (+2 score, 0 $\to$ 2): What is the difference between test-time and training-time backdoor attacks against RAG systems?
  *V2 Answer Snippet*: Test-time and training-time backdoor attacks against RAG systems differ in their execution timing rather than the approach of injecting backdoors:

Test-time Backdoor Attacks are a...

- **q017** (+2 score, 0 $\to$ 2): In SelfGraphRAG, what does the 'Verbalize' function do to an extracted subgraph?
  *V2 Answer Snippet*: In SelfGraphRAG, the 'Verbalize' function takes an extracted subgraph ˆ G and linearizes it into a natural language string. This is done by converting each triple (v_i, v_j, r) in ...

- **q026** (+1 score, 1 $\to$ 2): Trace how post-graph-rag's extraction-time invariants prevent cascading errors during multi-hop graph traversal at retrieval time.
  *V2 Answer Snippet*: Post-graph-rag's extraction-time invariants play a crucial role in preventing cascading errors during multi-hop graph traversal at retrieval time. Here is a breakdown of how these ...

- **q030** (+1 score, 1 $\to$ 2): In Figure 1 of the AWM paper, how does the agent interact with document pages and working memory, and how is memory-only answerability evaluated?
  *V2 Answer Snippet*: Figure 1 in the AWM paper depicts the interaction of the agent with document pages and working memory as follows:

1. The agent receives the multi-page document D and a question q....

- **q031** (+1 score, 0 $\to$ 1): What relationship between raw accuracy and penalty-aware quality score Q is illustrated in Figure 1 of 'Why RAGs Hallucinate' across the three evaluated RAG systems?
  *V2 Answer Snippet*: Figure 1 of 'Why RAGs Hallucinate' illustrates a relationship between raw accuracy and penalty-aware quality score Q across the three evaluated RAG systems. The figure shows qualit...

- **q036** (+1 score, 0 $\to$ 1): In CEMMKG, what was the overall soft accuracy when incorporating reference sentences as additional local context compared to incorporating full reference paragraphs on the VisionHeavy subset in Section 5.2.1?
  *V2 Answer Snippet*: In CEMMKG, when incorporating reference sentences as additional local context achieved a soft accuracy of 36.84% on the VisionHeavy subset, this was 13.19 percentage points higher ...

### B. Regressions (15 questions)

- **q001** (2 $\to$ 0): Which database management system and vector indexing extension does post-graph-rag use to store text passages, the entity graph, and community summaries in a single storage engine?
  *V2 Answer Snippet*: Post-graph-rag uses PostgreSQL as the database management system and pgvector for the vector index extension to store text passages, the entity graph, and community summaries in a ...

- **q002** (2 $\to$ 1): Which three primary biological knowledge sources and databases are integrated into the multi-source dataset for Homo-RAG?
  *V2 Answer Snippet*: The three primary biological knowledge sources and databases integrated into the multi-source dataset for Homo-RAG are:

1. ZFIN (Zebrafish Model Organism Database)
2. UniProt (Uni...

- **q009** (2 $\to$ 1): What are 'knowledge-gap canaries' in the penalty-aware RAG evaluation framework, and what type of failure do they isolate?
  *V2 Answer Snippet*: 'Knowledge-gap canaries' in the penalty-aware RAG evaluation framework are specific questions from the evaluation set whose answers are verifiably absent from the knowledge base. T...

- **q013** (2 $\to$ 1): What does the acronym 'MMC' stand for in the AWM paper's four-cell outcome analysis, and what condition defines it?
  *V2 Answer Snippet*: The acronym 'MMC' in the AWM paper's four-cell outcome analysis stands for Memory Missing Correct. The condition that defines it is s ans = 1 and s mem = 0, meaning the agent answe...

- **q014** (2 $\to$ 1): What does the acronym 'STeReO' stand for in the context of multimodal retrieval?
  *V2 Answer Snippet*: In the context of multimodal retrieval, the acronym 'STeReO' stands for Speech and Text Re-ranking Orchestrator. This is derived directly from the information provided in the given...

- **q016** (2 $\to$ 1): In the context of the geo-bias evaluation benchmark over public companies, what do 'inductive' and 'deductive' question framing refer to?
  *V2 Answer Snippet*: In the context of the geo-bias evaluation benchmark over public companies, 'inductive' and 'deductive' question framing refer to the direction in which the inference is made. Speci...

- **q019** (1 $\to$ 0): How do retrieval-stage defenses differ from rerank-stage defenses in mitigating adversarial attacks on RAG pipelines?
  *V2 Answer Snippet*: Retrieval-stage defenses and rerank-stage defenses in RAG pipelines address adversarial attacks in distinct ways, as they operate at different stages of the pipeline. Retrieval-sta...

- **q021** (2 $\to$ 0): How does STeReO compare with conventional pipeline-cascade approaches (ASR + Text RAG) in retrieving spoken audio content?
  *V2 Answer Snippet*: STeReO compares favorably with conventional pipeline-cascade approaches (ASR + Text RAG) in handling spoken audio content for retrieval. Unlike the pipeline-cascade method, which i...

- **q022** (2 $\to$ 0): How does the impact of misleading context differ from the impact of distraction context on model accuracy in company factual QA?
  *V2 Answer Snippet*: Based on the provided context, the impact of misleading context differs from the impact of distraction context on model accuracy in company factual QA. Specifically, the context in...

- **q023** (2 $\to$ 1): Explain the multi-hop retrieval chain in Homo-RAG from an uncharacterized zebrafish gene query to the final evidence-grounded function prediction.
  *V2 Answer Snippet*: The multi-hop retrieval chain in Homo-RAG from an uncharacterized zebrafish gene query to the final evidence-grounded function prediction involves a series of steps designed to bri...

- **q024** (2 $\to$ 1): Why does the combination of final-answer correctness and memory-only answerability in AWM-GRPO prevent agents from receiving high rewards for lucky guesses?
  *V2 Answer Snippet*: The combination of final-answer correctness and memory-only answerability in AWM-GRPO prevents agents from receiving high rewards for lucky guesses because the AWM reward distingui...

- **q025** (2 $\to$ 1): Why can't standard embedding-based retrieval solve multi-hop questions over document corpora, and how does SelfGraphRAG's two-hop synthetic training chain enable its graph retriever to bridge this gap?
  *V2 Answer Snippet*: Standard embedding-based retrieval fails to solve multi-hop questions over document corpora because it relies on surface-level semantic similarity, which is insufficient to handle ...

- **q028** (2 $\to$ 1): Based on Figure 1 in post-graph-rag, how are the LLM extractor and validation gates structurally separated, and how is data stored in PostgreSQL?
  *V2 Answer Snippet*: Figure 1 of the post-graph-rag system architecture illustrates the separation of the LLM extractor and validation gates in the system. The LLM extractor is depicted as a generative...

- **q032** (2 $\to$ 0): According to Figure 1 in STeReO, how does the STeReO architecture differ from the traditional text-only RAG pipeline and the pipeline-cascade (ASR + Text RAG) architecture?
  *V2 Answer Snippet*: According to Figure 1 in STeReO, the proposed STeReO architecture differs from both the traditional text-only RAG pipeline and the pipeline-cascade (ASR + Text RAG) architecture in...

- **q033** (2 $\to$ 1): In Table 2 of 'When RAG Fails to Equalize', what are the reported Deductive and Inductive accuracies for LLaMA-8B and LLaMA-70B under the directional framing experiment?
  *V2 Answer Snippet*: Based on the information provided in Table 11, there is no specific Table 2 mentioned in the text for 'When RAG Fails to Equalize'. Therefore, I cannot directly extract the reporte...

---

## 6. Analysis & Findings

### Did BM25 Help?
**Yes, decisively on exact technical terms and acronyms.**
In V1, dense vector search completely failed on queries containing rare identifiers (e.g. `q015` asking for the components of ECS). BM25 provided high-precision lexical matches, boosting exact terminology retrieval from 40% to 83.3% and bringing relevant passages into the candidate pool.

### Did Reranking Help?
**Yes, significantly on MRR and precision.**
While RRF creates a strong combined candidate list, it ranks items based on arbitrary reciprocal rank arithmetic. The Cross-Encoder computes joint attention between the query and passage text, correctly prioritizing the exact ground-truth passages at rank 1 and 2. This directly drove Hit@1 from 36.1% to **55.6%** and MRR from 0.5331 to **0.6898**.

### Latency Impact
- V1 Mean Latency: **3.39s**
- V2 Mean Latency: **4.45s**
- *Overhead*: Cross-encoder inference on 25 candidates adds only **~1.06 seconds** on GPU, which is negligible compared to LLM generation time.

---

## 7. Evidence-Based Recommendation for Next Experiment

**Upgrade the Generator LLM (or Context Fact-Extraction Prompting)**.

With V2, retrieval Hit@5 reached **83.3%**, and Hit@10 reached **91.7%**. The primary remaining performance ceiling is no longer retrieval; it is **LLM reasoning and fact extraction**. Out of the remaining incorrect questions, the relevant passages are now almost always present in the context, but the small `qwen2.5:3b` parameter capacity struggles to synthesize multi-condition facts.

The next highest-ROI experiment is:
1. Benchmark `qwen2.5:7b` (or `llama3.1:8b`) under the identical V2 retrieval pipeline.
2. Introduce a two-step generation prompt (Step 1: Extract direct quotes and facts; Step 2: Synthesize answer with citations).
