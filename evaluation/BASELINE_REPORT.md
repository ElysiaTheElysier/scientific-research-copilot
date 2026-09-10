# Baseline RAG Evaluation Report

**Generated**: 2026-09-10T08:21:31.169645+00:00
**Git Commit**: `a2cf7131fefb7e5df10c82b526160f80fa956f82`
**Branch**: `master`

---

## 1. Executive Summary

This report evaluates the current baseline Scientific Research Copilot RAG system against a curated
evaluation dataset of **40 questions** (36 answerable, 4 unanswerable)
covering **10 scientific papers** with **374 indexed chunks** (319 text + 55 figure).

**Key Findings**:
- **Retrieval**: Hit@5 = 77.8%, MRR = 0.5331
- **Answer Quality**: 47.2% fully correct, 1.06/2.0 mean correctness
- **Faithfulness**: 0.86/2.0 mean faithfulness
- **Abstention**: 100.0% accuracy on unanswerable questions
- **Latency**: 3.4s average end-to-end

---

## 2. Repository Architecture

```
scientific-research-copilot/
├── src/
│   ├── ingestion/          # PDF→Markdown→Chunks pipeline
│   │   ├── convert.py      # Docling PDF conversion
│   │   ├── caption.py      # LLaVA figure captioning
│   │   ├── chunk.py        # Section-aware chunking
│   │   ├── context_extractor.py
│   │   └── pipeline.py     # Unified pipeline runner
│   ├── retrieval/
│   │   └── vectordb.py     # Qdrant + Qwen3-Embedding-0.6B
│   └── rag.py              # Baseline RAG (qwen2.5:3b)
├── data/
│   ├── raw/                # 10 PDF papers
│   ├── processed/          # Markdown, chunks.json, images
│   └── qdrant/             # Local vector DB
├── evaluation/             # Evaluation dataset & results
└── test/                   # Unit tests
```

---

## 3. Current RAG Pipeline

| Stage | Component | Detail |
|---|---|---|
| **Embedding** | `Qwen/Qwen3-Embedding-0.6B` | 1024-dim, max 4096 tokens, CUDA |
| **Vector DB** | Qdrant (local disk) | Collection: `scientific_papers`, Cosine distance |
| **Retrieval** | Dense-only (ANN) | Default top-K = 5, no filters, no reranking |
| **LLM** | `qwen2.5:3b` via Ollama | Default temperature, Q4_K_M quantization |
| **Prompt** | System + Context + Question | Evidence-grounded persona, cite Paper ID & Section |
| **Chunking** | Section-aware + multimodal | 3000 char / 300 overlap, sections ≤1000 words kept intact |
| **Corpus** | 374 chunks | 319 text + 55 figure chunks from 10 papers |

**Pipeline Flow**:
```
User Question → Qwen3-Embedding (1024d) → Qdrant Cosine ANN (top-5)
→ Context blocks with metadata headers → System prompt + context
→ qwen2.5:3b via Ollama → Answer with citations
```

---

## 4. Evaluation Dataset

| Property | Value |
|---|---|
| Total questions | 40 |
| Answerable | 36 |
| Unanswerable | 4 |
| Question types | 8 categories |
| Difficulty | Easy: 10, Medium: 20, Hard: 10 |
| Papers covered | 10/10 |
| Cross-paper questions | 3 (q018, q020, q027) |
| Ground-truth chunks referenced | 49 unique chunk IDs |
| Figure-related questions | 5 |

### Dataset Warnings

| Issue | Details |
|---|---|
| Section name quirk | q001, q026, q028 have `"Chandan Rajah"` as `relevant_section` — this is the author name parsed as a section header by the chunking pipeline for paper 2608.24921v1. Chunk 0's content is actually the abstract. |

---

## 5. Retrieval Metrics

*Evaluated on 36 answerable questions only.*

| Metric | Score |
|---|---|
| **Hit@1** | 36.1% |
| **Hit@3** | 63.9% |
| **Hit@5** | 77.8% |
| **Hit@10** | 88.9% |
| **Recall@1** | 0.1755 |
| **Recall@3** | 0.3454 |
| **Recall@5** | 0.5037 |
| **Recall@10** | 0.6931 |
| **MRR** | 0.5331 |

---

## 6. Answer Quality

*Evaluated on 36 answerable questions.*

| Metric | Score |
|---|---|
| **Mean Correctness** | 1.06 / 2.0 |
| **Fully Correct (2)** | 47.2% |
| **Partially Correct (1)** | 11.1% |
| **Incorrect (0)** | 41.7% |
| **Mean Keyword Overlap** | 0.4280 |

---

## 7. Faithfulness / Groundedness

| Metric | Score |
|---|---|
| **Mean Faithfulness** | 0.86 / 2.0 |
| **Fully Grounded (2)** | 38.9% |

---

## 8. Citation Quality

| Metric | Score |
|---|---|
| **Citation Correctness** | 100.0% |
| **Citation Completeness** | 94.4% |

*Citation correctness: ≥1 ground-truth paper appears in retrieved top-5.*
*Citation completeness: ALL ground-truth papers appear in retrieved top-5.*

---

## 9. Abstention / Unanswerable Performance

| Metric | Score |
|---|---|
| **Abstention Accuracy** | 100.0% |
| **Hallucination Rate** | 0.0% |
| **Total Unanswerable** | 4 |
| **Correct Abstentions** | 4 |

### Individual Unanswerable Results

**q037**: What learning rate schedule and number of training epochs are recommended by the authors of post-gra...
- Status: ✅ Correctly abstained
- Answer excerpt: The retrieved context does not contain any specific recommendations for the learning rate schedule and number of training epochs for fine-tuning the underlying LLM extractor in post-graph-rag. The con...
- Judge reasoning: The provided corpus does not contain any specific recommendations for the learning rate schedule and number of training epochs for fine-tuning the underlying LLM extractor in post-graph-rag.

**q038**: What was the measured inference latency in milliseconds for running Homo-RAG on an NVIDIA H100 GPU c...
- Status: ✅ Correctly abstained
- Answer excerpt: Based on the provided context, there is no specific information about the inference latency for Homo-RAG running on an NVIDIA H100 GPU cluster when processing 10,000 gene queries. The context contains...
- Judge reasoning: The provided context does not contain specific information about Homo-RAG's inference latency for processing 10,000 gene queries on an NVIDIA H100 GPU cluster.

**q039**: According to 'Why RAGs Hallucinate', what was the canary violation rate and penalty-aware score Q of...
- Status: ✅ Correctly abstained
- Answer excerpt: According to the provided context, the canary violation rate for Anthropic Claude 3.5 Sonnet was 16.7%, and the penalty-aware score Q for this system was +0.767....
- Judge reasoning: The provided answer fabricates data not found in the context of 'Why RAGs Hallucinate'.

**q040**: Which quantum computing algorithms or quantum annealing hardware are proposed in the 10 papers to op...
- Status: ✅ Correctly abstained
- Answer excerpt: Based on the provided context, there is no mention of quantum computing algorithms or quantum annealing hardware related to optimizing graph traversals in Graph RAG. The context focuses on traditional...
- Judge reasoning: The provided corpus does not contain any information about quantum computing algorithms or quantum annealing hardware related to optimizing graph traversals in Graph RAG.

---

## 10. Performance by Question Type

| Type | Count | Hit@5 | Recall@5 | MRR | Correctness | Faithfulness |
|---|---|---|---|---|---|---|
| comparison | 5 | 80.0% | 0.3233 | 0.4533 | 1 | 1 |
| direct_factual | 6 | 83.3% | 0.7500 | 0.5000 | 0.6667 | 0.3333 |
| exact_terminology | 5 | 40.0% | 0.1667 | 0.1650 | 1.2 | 1 |
| figure_related | 5 | 100.0% | 0.6000 | 0.7500 | 1.4 | 1 |
| multi_hop_reasoning | 5 | 80.0% | 0.2367 | 0.8000 | 1.4 | 1.2 |
| numerical_experimental | 4 | 100.0% | 1.0000 | 0.8750 | 1.5 | 1.5 |
| technical_concept | 6 | 66.7% | 0.5000 | 0.3083 | 0.5 | 0.3333 |
| unanswerable | 4 | 0.0% | 0.0000 | 0.0000 | - | - |

---

## 11. Performance by Difficulty

| Difficulty | Count | Hit@5 | Recall@5 | MRR | Correctness | Faithfulness |
|---|---|---|---|---|---|---|
| easy | 10 | 66.7% | 0.5370 | 0.3694 | 0.8889 | 0.5556 |
| hard | 10 | 90.0% | 0.4133 | 0.7333 | 1.1 | 1 |
| medium | 20 | 76.5% | 0.5392 | 0.5020 | 1.1176 | 0.9412 |

---

## 12. Performance by Paper

| Paper ID | Count | Hit@5 | Recall@5 | MRR | Correctness |
|---|---|---|---|---|---|
| 2608.24921v1 | 6 | 83.3% | 0.4139 | 0.6667 | 1.3333 |
| 2608.24977v2 | 3 | 66.7% | 0.5556 | 0.2278 | 0.3333 |
| 2608.25123v1 | 4 | 75.0% | 0.3750 | 0.4271 | 0.5 |
| 2608.25466v1 | 3 | 66.7% | 0.4444 | 0.6667 | 1.3333 |
| 2608.25618v1 | 5 | 60.0% | 0.5333 | 0.3500 | 1.4 |
| 2608.25717v1 | 4 | 75.0% | 0.4250 | 0.5250 | 1.5 |
| 2608.25986v1 | 2 | 100.0% | 1.0000 | 0.7500 | 0 |
| 2608.26194v1 | 4 | 100.0% | 0.5833 | 0.6750 | 1.5 |
| 2608.26379v1 | 2 | 100.0% | 0.3500 | 0.4167 | 0 |
| 2608.26385v1 | 6 | 83.3% | 0.4000 | 0.6250 | 0.6667 |
| [unanswerable] | 4 | 0.0% | 0.0000 | 0.0000 | - |

---

## 13. Error Analysis

**Total failures identified**: 19

### Failure Type Distribution
- **generation_failure**: 11
- **grounding_failure**: 4
- **retrieval_failure**: 2
- **ranking_failure**: 2

### Representative Failures

#### q002 — grounding_failure
- **What happened**: Answer contains mostly unsupported claims.
- **Generated answer**: The three primary biological knowledge sources and databases integrated into the multi-source dataset for Homo-RAG are ZFIN (Zebrafish Information Network), UniProt, and PubMed.

These resources provi...
- **Expected answer**: Homo-RAG integrates three biological resources: ZFIN (Zebrafish Information Network) for zebrafish gene information and human orthology mappings, UniProt for curated protein functional annotations of ...
- **Reasoning**: The generated answer does not match the expected answer as it does not list all three biological resources (ZFIN, UniProt, and PubMed) integrated into Homo-RAG. The context provided does not contain any information about Homo-RAG or its integrated resources.

#### q003 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: STeReO is evaluated across the Spoken SQuAD and MS MARCO datasets in its experimental setup....
- **Expected answer**: STeReO is evaluated on Spoken SQuAD (comprising text queries and TTS-generated audio passages, ~2.8K audio passages) and MS MARCO (comprising text-only web passages, ~9.1K text passages)....
- **Reasoning**: The generated answer is incorrect and does not match the expected answer. The expected answer states that STeReO is evaluated on Spoken SQuAD (comprising text queries and TTS-generated audio passages, ~2.8K audio passages) and MS MARCO (comprising text-only web passages, ~9.1K text passages). The generated answer is vague and does not specify these details. Moreover, the context provided does not contain any information relevant to the evaluation of STeReO or the datasets mentioned.

#### q004 — retrieval_failure
- **What happened**: Ground-truth chunks were not retrieved in top-10.
- **Generated answer**: The three commercial RAG systems evaluated in 'Why RAGs Hallucinate' were not explicitly named in the provided context. However, based on the context, it can be inferred that three commercial RAG syst...
- **Expected answer**: The three commercial RAG systems evaluated are CustomGPT.ai, OpenAI RAG (OpenAI vector store / assistants), and Gemini RAG (Google File Search store). The evaluation benchmark used was SimpleQA-Verifi...
- **Reasoning**: The generated answer does not match the expected answer, as it incorrectly states that the three commercial RAG systems were not explicitly named in the provided context when the expected answer clearly states that they are CustomGPT.ai, OpenAI RAG, and Gemini RAG. Additionally, the generated answer incorrectly states that the benchmark dataset was used for evaluation, while the expected answer indicates that the RAG systems were evaluated using the benchmark dataset, which is 1,000 SimpleQA-Verified questions. Furthermore, the generated answer is unsupported by the provided context, as there is no information about specific RAG systems or benchmark datasets in the given context.

#### q005 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: The three sequential stages of the SelfGraphRAG pipeline are:

1. **Doc2Graph**: This stage involves constructing a knowledge graph (G) from an unlabeled document corpus (D) using a chunking function....
- **Expected answer**: SelfGraphRAG operates in three sequential stages: (1) Doc2Graph, which constructs the knowledge graph G from corpus D; (2) SynthGen, which generates synthetic QA training pairs T directly from graph s...
- **Reasoning**: The generated answer is incorrect and does not match the expected answer. It contains incorrect information about the stages and processes, such as the mention of a chunking function, entity and relation extraction steps, and the use of LLMs for question generation. Additionally, the context provided does not contain any information that could support the generated answer.

#### q006 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: According to the taxonomy in 'Retrieved But Not Reliable', the three primary high-level security objectives of adversarial attacks against RAG systems are accuracy, privacy, and fairness....
- **Expected answer**: Adversarial attacks on RAG are organized into three primary objectives: (1) Accuracy attacks (inducing incorrect or harmful outputs), (2) Privacy attacks (inferring or extracting confidential corpus c...
- **Reasoning**: The GENERATED ANSWER is incorrect and does not match the expected answer. It also lacks faithfulness as it cites a 'Retrieved But Not Reliable' which is not present in the provided context.

#### q007 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: 'Memory-only answerability' in the AWM framework is a diagnostic that evaluates whether an agent's terminal working memory alone can support the final answer provided. It does this by having a frozen ...
- **Expected answer**: Memory-only answerability is an evaluation diagnostic where a frozen reader model receives solely the question and the agent-written terminal working memory (without page images or trajectory context)...
- **Reasoning**: The generated answer is mostly unsupported. It introduces new facts not aligned with the expected answer or the provided context, such as the 42.5% failure rate and the claim about AWM-GRPO. The generated answer also diverges from the expected answer by not directly addressing the concept of memory-only answerability and instead provides additional context about the GRPO framework and evaluation metrics, which is not grounded in the provided context.

#### q008 — grounding_failure
- **What happened**: Answer contains mostly unsupported claims.
- **Generated answer**: Post-graph-rag handles explicitly denied relations in extracted text by rendering them explicitly as (A) -[NOT worked_with]-&gt; (B) . This design approach is chosen to avoid the naïve rendering of a ...
- **Expected answer**: post-graph-rag stores explicitly denied relations using the original positive predicate accompanied by a boolean negation flag. It avoids inverting predicates into labels like 'did_not_have_relationsh...
- **Reasoning**: The generated answer does not fully match the expected answer and does not use the context provided.

#### q010 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: The 'validity-composition problem' identified in evidence-aware RAG evaluation is that evidence supporting an evaluation signal for one use does not automatically support the next inference or decisio...
- **Expected answer**: The validity-composition problem is the finding that evidence supporting an evaluation metric for one purpose does not automatically validate subsequent inferences or downstream decisions built from i...
- **Reasoning**: The generated answer does not match the expected answer and is not grounded in the provided context. The generated answer introduces new concepts and arguments not aligned with the validity-composition problem described in the expected answer.

#### q011 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: In CEMMKG, the difference between local context and global context when constructing textual context for a visual element lies in their scope and density of information.

Local context captures fine-g...
- **Expected answer**: Local context captures specific textual information localized around a visual element, consisting of surrounding text adjacent to the image and reference-based text (sentences, paragraphs, or paragrap...
- **Reasoning**: The generated answer does not match the expected answer or the retrieved context. The answer introduces concepts not present in the expected answer, such as 'CEMMKG' and a detailed explanation of local and global context, which are not supported by the provided context.

#### q012 — ranking_failure
- **What happened**: Ground-truth chunks exist in top-10 but not in top-5 used for generation.
- **Generated answer**: The context provided does not contain information specifically addressing the difference between test-time and training-time backdoor attacks against RAG systems. Therefore, I cannot directly answer t...
- **Expected answer**: In test-time backdoor attacks, the attacker cannot modify model parameters; they inject malicious documents into the corpus that act as trigger-aligned context, retrieved only when a query contains a ...
- **Reasoning**: The generated answer does not match the expected answer and is unrelated to the provided context. The context does not contain information about the specific difference between test-time and training-time backdoor attacks against RAG systems, and the generated answer instead provides a general explanation of backdoor attacks in RAG systems without addressing the question.

#### q015 — retrieval_failure
- **What happened**: Ground-truth chunks were not retrieved in top-10.
- **Generated answer**: The acronym 'ECS' in Homo-RAG stands for Evidence Confidence Score. The five weighted components that comprise it are Precision, Recall, F1, NDCG@10, and MRR....
- **Expected answer**: 'ECS' stands for 'Evidence Confidence Score'. According to Table 1, it is composed of five weighted features: (1) semantic score (50%, dense vector similarity from S-PubMedBert-MS-MARCO); (2) gene mat...
- **Reasoning**: The GENERATED ANSWER is completely incorrect and unrelated to the expected answer and the provided context. The GENERATED ANSWER does not mention 'ECS' or any of its components, instead it introduced a new acronym 'Homo-RAG' and provided a list of five metrics (Precision, Recall, F1, NDCG@10, MRR) that are not related to the context or the expected answer.

#### q017 — ranking_failure
- **What happened**: Ground-truth chunks exist in top-10 but not in top-5 used for generation.
- **Generated answer**: Based on the provided context, there is no information about a 'Verbalize' function or what it does to an extracted subgraph in SelfGraphRAG. The context focuses on the process of constructing a knowl...
- **Expected answer**: The 'Verbalize' function deterministically linearizes a subgraph into a natural language string by converting each triple (v_i, v_j, r) into the fixed sentence template 'v_i [r] v_j' and concatenating...
- **Reasoning**: The GENERATED ANSWER does not mention the 'Verbalize' function or its operations, which contradicts the EXPECTED ANSWER. The context does not contain any information about the 'Verbalize' function.

#### q018 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: Post-graph-rag and SelfGraphRAG differ in their approaches to handling entity deduplication and graph extraction quality. Post-graph-rag, as described in the context, extracts a knowledge graph from t...
- **Expected answer**: SelfGraphRAG reuses GraphRAG's Doc2Graph extraction without modification, taking the raw union of chunk-level entities without deduplication or coreference resolution, leaving disconnected nodes when ...
- **Reasoning**: The generated answer does not match the expected answer and is mostly unsupported by the retrieved context. The generated answer introduces new concepts and details not present in the expected answer, such as predicate normalisation, controlled-vocabulary snapping, alias-based entity resolution, and explicit negation. Additionally, it incorrectly states that SelfGraphRAG does not explicitly mention entity deduplication and graph extraction quality as its focus, which is contradicted by the expected answer.

#### q020 — generation_failure
- **What happened**: Evidence was retrieved but the answer is incorrect.
- **Generated answer**: 'Why RAGs Hallucinate' and 'Assessing the Downstream Utility of Evidence-Aware Retrieval' both address the limitations of conventional retrieval evaluation metrics by highlighting that answer support-...
- **Expected answer**: 'Why RAGs Hallucinate' argues that conventional volume scoring (+1 for correct, 0 for incorrect) rewards hallucination by encouraging models to guess blindly; it introduces asymmetric penalty-aware sc...
- **Reasoning**: The generated answer does not directly compare the content of 'Why RAGs Hallucinate' and 'Assessing the Downstream Utility of Evidence-Aware Retrieval', nor does it accurately reflect the content of the expected answer.

#### q026 — grounding_failure
- **What happened**: Answer contains mostly unsupported claims.
- **Generated answer**: To prevent cascading errors during multi-hop graph traversal at retrieval time, post-graph-rag uses a set of extraction-time invariants that decide what is allowed to become graph structure at all. Th...
- **Expected answer**: In unconstrained extraction, generative LLMs produce vague predicates ('relates_to'), pronominal vertices ('his father'), bare numbers ('$18.4 billion'), and filing-specific transient names. During mu...
- **Reasoning**: The GENERATED ANSWER focuses on extraction-time invariants rather than the issues described in the RETRIEVED CONTEXT related to multi-hop retrieval and graph traversal failures. The answer does not directly address the problems mentioned in the expected answer.

---

## 14. Main Bottlenecks

3. **Generation Quality** (Mean correctness=1.06/2.0): The 3B model struggles with complex scientific reasoning. Consider a larger LLM or improved prompting.

4. **Faithfulness** (Mean=0.86/2.0): Significant grounding issues — model generates unsupported claims.

---

## 15. Recommended Next Experiments

Based on the measured failure modes:

1. **Hybrid Retrieval (BM25 + Dense)**: Add sparse keyword matching alongside dense retrieval to improve Hit@K for terminology-heavy queries.
2. **Cross-Encoder Reranking**: Add a reranker after retrieval to improve ranking quality and MRR.
3. **Larger LLM**: Upgrade from `qwen2.5:3b` to `qwen2.5:7b` or `llama3:8b` for better scientific reasoning and instruction following.
4. **Improved Prompting**: Add explicit abstention instructions ("If the context does not contain sufficient evidence, say so") and structured citation format.
5. **RAGAS / Automated Evaluation**: Implement RAGAS metrics (context relevance, answer relevance, faithfulness) for continuous evaluation.

---

## 16. Limitations

1. **LLM-as-Judge Bias**: The judge model (`qwen2.5:3b`) is the same as the generator model. Scores may be inflated.
2. **Small Evaluation Set**: 40 questions may not capture all failure modes.
3. **No Human Evaluation**: All metrics are automated — no human annotation.
4. **Single Retrieval Method**: Only dense retrieval evaluated (no BM25 baseline comparison).
5. **Dataset Quirks**: Some chunk section names reflect parser artifacts (e.g., author names as section headers).

---

## 17. Reproducibility Information

| Property | Value |
|---|---|
| Git Commit | `a2cf7131fefb7e5df10c82b526160f80fa956f82` |
| Git Branch | `master` |
| Python Version | `3.13.2` |
| Embedding Model | `Qwen/Qwen3-Embedding-0.6B` |
| LLM Model | `qwen2.5:3b` |
| Vector DB | Qdrant local at `data/qdrant` |
| Collection | `scientific_papers` |
| Retrieval K (eval) | 10 (generation uses top-5) |
| Eval Dataset | `evaluation/questions.json` (40 questions) |
| Timestamp | 2026-09-10T08:21:31.169645+00:00 |
| Command | `cd c:\Ki_OJT\scientific-research-copilot && .venv\Scripts\python.exe evaluation/run_baseline.py` |
| Torch | `2.6.0+cu124` |
| sentence-transformers | `6.0.1` |
| qdrant-client | `N/A` |
| ollama | `N/A` |
