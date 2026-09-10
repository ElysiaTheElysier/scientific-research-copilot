"""Baseline RAG Evaluation Runner.

Executes all evaluation questions against the current RAG system,
computes retrieval/generation/faithfulness/citation/abstention metrics,
and generates a comprehensive BASELINE_REPORT.md.

Usage:
    cd c:\\Ki_OJT\\scientific-research-copilot
    .venv\\Scripts\\python.exe evaluation/run_baseline.py

Does NOT modify any source code in src/.
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ollama
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants (matching src/retrieval/vectordb.py and src/rag.py exactly)
# ---------------------------------------------------------------------------
COLLECTION_NAME = "scientific_papers"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DB_PATH = str(PROJECT_ROOT / "data" / "qdrant")
LLM_MODEL = "qwen2.5:3b"
DEFAULT_TOP_K = 5
EVAL_TOP_K = 10  # Retrieve top-10 for metrics at K=1,3,5,10

SYSTEM_PROMPT = """You are an evidence-grounded scientific research assistant.
Answer the user's question using ONLY the retrieved context below.
Cite the relevant Paper ID and Section for your claims.
If a figure visual analysis is provided, refer to the figure and its findings."""

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "baseline_results.json"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "BASELINE_REPORT.md"
CSV_PATH = RESULTS_DIR / "baseline_summary.csv"

# ---------------------------------------------------------------------------
# Stop words for keyword overlap scoring
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "yet", "if", "that", "which", "who", "whom",
    "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "about", "up", "also", "just", "because", "while",
}

# Abstention detection phrases
ABSTENTION_PHRASES = [
    "cannot be answered", "cannot answer", "not enough information",
    "no information", "not mentioned", "not discussed", "not addressed",
    "insufficient evidence", "insufficient information", "not found",
    "does not contain", "do not contain", "doesn't contain",
    "no relevant information", "cannot determine", "unable to answer",
    "unable to determine", "not available", "not provided",
    "i don't have enough", "i cannot find", "the context does not",
    "the provided context", "based on the provided context, there is no",
    "no evidence", "not covered", "outside the scope",
    "i'm unable to", "i am unable to",
]


def log(msg: str):
    """Print with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ===================================================================
# PHASE 1: RAG EXECUTION
# ===================================================================

def build_context_prompt(chunks: list[dict], query: str) -> tuple[str, list[dict]]:
    """Reconstruct the exact prompt from src/rag.py using the top-K chunks.

    Returns (user_prompt, citations).
    """
    context_blocks = []
    citations = []

    for c in chunks:
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
    return user_prompt, citations


def run_rag_on_question(
    question: dict,
    qdrant_client: QdrantClient,
    embed_model: SentenceTransformer,
) -> dict:
    """Execute the RAG pipeline for one question, capturing scores."""
    query = question["question"]
    qid = question["id"]

    # 1) Embed query
    t0 = time.perf_counter()
    query_vector = embed_model.encode(query).tolist()
    embed_time = time.perf_counter() - t0

    # 2) Search Qdrant with limit=EVAL_TOP_K (to compute metrics at K=1..10)
    t1 = time.perf_counter()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=EVAL_TOP_K,
        with_payload=True,
    )
    retrieval_time = time.perf_counter() - t1

    # Extract scored results
    retrieved = []
    for rank, point in enumerate(results.points, 1):
        retrieved.append({
            "rank": rank,
            "chunk_id": point.payload.get("chunk_id"),
            "score": float(point.score),
            "paper_id": point.payload.get("paper_id", "Unknown"),
            "section": point.payload.get("section", "General"),
            "chunk_type": point.payload.get("chunk_type", "text"),
            "word_count": point.payload.get("word_count", 0),
            "payload": point.payload,  # keep full payload for prompt construction
        })

    # 3) Build prompt using top-5 (matching DEFAULT_TOP_K=5)
    top_k_payloads = [r["payload"] for r in retrieved[:DEFAULT_TOP_K]]
    user_prompt, citations = build_context_prompt(top_k_payloads, query)

    # 4) Call LLM
    t2 = time.perf_counter()
    error_msg = None
    answer_text = ""
    token_info = {}
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer_text = response["message"]["content"]
        token_info = {
            "prompt_eval_count": response.get("prompt_eval_count", 0),
            "eval_count": response.get("eval_count", 0),
            "total_duration_ns": response.get("total_duration", 0),
        }
    except Exception as e:
        error_msg = str(e)
        answer_text = f"[ERROR] {e}"

    generation_time = time.perf_counter() - t2
    total_time = time.perf_counter() - t0

    # Strip full payloads from stored results (too large)
    retrieved_clean = []
    for r in retrieved:
        rc = {k: v for k, v in r.items() if k != "payload"}
        retrieved_clean.append(rc)

    return {
        "question_id": qid,
        "question": query,
        "question_type": question.get("question_type"),
        "difficulty": question.get("difficulty"),
        "answerable": question.get("answerable"),
        "ground_truth_chunk_ids": question.get("relevant_chunk_ids", []),
        "ground_truth_paper_ids": question.get("paper_ids", []),
        "expected_answer": question.get("expected_answer"),
        "retrieved": retrieved_clean,
        "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved_clean],
        "retrieved_paper_ids": list(dict.fromkeys(
            r["paper_id"] for r in retrieved_clean
        )),
        "citations": citations,
        "generated_answer": answer_text,
        "error": error_msg,
        "latency": {
            "embed_s": round(embed_time, 3),
            "retrieval_s": round(retrieval_time, 3),
            "generation_s": round(generation_time, 3),
            "total_s": round(total_time, 3),
        },
        "tokens": token_info,
    }


# ===================================================================
# PHASE 2: EVALUATION (LLM Judge + Keyword + Rule-based)
# ===================================================================

JUDGE_PROMPT = """You are a strict evaluation judge for a scientific RAG system.

## Task
Compare the GENERATED ANSWER against the EXPECTED ANSWER and the RETRIEVED CONTEXT.
Evaluate on two dimensions.

## Scoring Rubric

### Correctness (does the answer match the expected answer?):
- 0 = Incorrect or contradictory — key facts are wrong or missing
- 1 = Partially correct — some correct facts but incomplete or has minor errors
- 2 = Fully correct — all key facts from expected answer are present and accurate

### Faithfulness (is the answer grounded in the retrieved context?):
- 0 = Mostly unsupported — claims are fabricated or not in the context
- 1 = Partially grounded — some claims supported, some unsupported
- 2 = Fully grounded — all claims are supported by retrieved context

## Input

EXPECTED ANSWER:
{expected_answer}

RETRIEVED CONTEXT (top chunks provided to the model):
{context_summary}

GENERATED ANSWER:
{generated_answer}

## Output
Respond ONLY with valid JSON (no markdown fences):
{{"correctness": <0|1|2>, "faithfulness": <0|1|2>, "reasoning": "<brief explanation>"}}"""

ABSTENTION_JUDGE_PROMPT = """You are a strict evaluation judge for a scientific RAG system.

## Task
This question is UNANSWERABLE from the corpus. The model should REFUSE to answer or clearly state it cannot find relevant information.

Evaluate whether the model correctly abstained.

## Scoring
- "correct_abstention": true if the model refuses, says it cannot answer, or states insufficient evidence
- "correct_abstention": false if the model fabricates an answer or claims to have evidence

QUESTION:
{question}

GENERATED ANSWER:
{generated_answer}

## Output
Respond ONLY with valid JSON (no markdown fences):
{{"correct_abstention": <true|false>, "reasoning": "<brief explanation>"}}"""


def keyword_overlap(expected: str, generated: str) -> float:
    """Compute keyword recall: fraction of expected answer's keywords in generated answer."""
    if not expected or not generated:
        return 0.0

    def tokenize(text):
        words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
        return {w for w in words if w not in STOP_WORDS and len(w) > 2}

    expected_kw = tokenize(expected)
    generated_kw = tokenize(generated)

    if not expected_kw:
        return 1.0

    overlap = expected_kw & generated_kw
    return round(len(overlap) / len(expected_kw), 4)


def check_abstention_rule_based(answer: str) -> bool:
    """Rule-based check for abstention phrases."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def evaluate_single(result: dict) -> dict:
    """Run LLM judge + keyword overlap + rule-based checks on one result."""
    qid = result["question_id"]
    is_answerable = result["answerable"]
    scores = {}

    if not is_answerable:
        # --- UNANSWERABLE: check abstention ---
        rule_abstains = check_abstention_rule_based(result["generated_answer"])

        # LLM judge for abstention
        llm_abstains = rule_abstains  # default fallback
        judge_reasoning = "rule-based only"
        try:
            prompt = ABSTENTION_JUDGE_PROMPT.format(
                question=result["question"],
                generated_answer=result["generated_answer"],
            )
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp["message"]["content"].strip()
            # Try to parse JSON
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            parsed = json.loads(raw)
            llm_abstains = parsed.get("correct_abstention", rule_abstains)
            judge_reasoning = parsed.get("reasoning", "")
        except Exception as e:
            judge_reasoning = f"LLM judge parse error: {e}"

        scores = {
            "correctness": None,
            "faithfulness": None,
            "keyword_overlap": None,
            "abstention_rule_based": rule_abstains,
            "abstention_llm_judge": llm_abstains,
            "correct_abstention": rule_abstains or llm_abstains,
            "judge_reasoning": judge_reasoning,
        }
    else:
        # --- ANSWERABLE: correctness + faithfulness ---
        expected = result.get("expected_answer") or ""
        generated = result.get("generated_answer") or ""

        # Keyword overlap
        kw_score = keyword_overlap(expected, generated)

        # Build context summary for judge (abbreviated)
        context_lines = []
        for r in result.get("retrieved", [])[:DEFAULT_TOP_K]:
            context_lines.append(
                f"[Chunk {r['chunk_id']}] Paper: {r['paper_id']} | Section: {r['section']} | Type: {r['chunk_type']}"
            )
        context_summary = "\n".join(context_lines) if context_lines else "No context retrieved."

        # LLM judge
        correctness = 1  # default fallback
        faithfulness = 1
        judge_reasoning = "fallback"

        try:
            prompt = JUDGE_PROMPT.format(
                expected_answer=expected,
                context_summary=context_summary,
                generated_answer=generated,
            )
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp["message"]["content"].strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            # Find JSON object in response
            json_match = re.search(r"\{[^{}]*\}", raw)
            if json_match:
                parsed = json.loads(json_match.group())
                correctness = int(parsed.get("correctness", 1))
                faithfulness = int(parsed.get("faithfulness", 1))
                judge_reasoning = parsed.get("reasoning", "")
            else:
                judge_reasoning = f"No JSON found in judge response: {raw[:200]}"
        except Exception as e:
            judge_reasoning = f"LLM judge error: {e}"

        # Clamp scores
        correctness = max(0, min(2, correctness))
        faithfulness = max(0, min(2, faithfulness))

        scores = {
            "correctness": correctness,
            "faithfulness": faithfulness,
            "keyword_overlap": kw_score,
            "abstention_rule_based": None,
            "abstention_llm_judge": None,
            "correct_abstention": None,
            "judge_reasoning": judge_reasoning,
        }

    return scores


# ===================================================================
# PHASE 3: METRICS COMPUTATION
# ===================================================================

def compute_retrieval_metrics(results: list[dict]) -> dict:
    """Compute Hit@K, Recall@K, MRR for answerable questions."""
    answerable = [r for r in results if r["answerable"]]
    if not answerable:
        return {}

    K_values = [1, 3, 5, 10]
    metrics = {}

    # Per-question retrieval evaluation
    hits = {k: [] for k in K_values}
    recalls = {k: [] for k in K_values}
    reciprocal_ranks = []

    for r in answerable:
        gt_ids = set(r["ground_truth_chunk_ids"])
        retrieved_ids = r["retrieved_chunk_ids"]

        if not gt_ids:
            continue

        # Hit@K and Recall@K
        for k in K_values:
            top_k_ids = set(retrieved_ids[:k])
            hit = 1 if (gt_ids & top_k_ids) else 0
            recall = len(gt_ids & top_k_ids) / len(gt_ids) if gt_ids else 0
            hits[k].append(hit)
            recalls[k].append(recall)

        # MRR: find rank of first relevant chunk
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, 1):
            if cid in gt_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    for k in K_values:
        metrics[f"hit_at_{k}"] = round(statistics.mean(hits[k]), 4) if hits[k] else 0.0
        metrics[f"recall_at_{k}"] = round(statistics.mean(recalls[k]), 4) if recalls[k] else 0.0

    metrics["mrr"] = round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0

    return metrics


def compute_answer_metrics(results: list[dict]) -> dict:
    """Compute answer quality metrics for answerable questions."""
    answerable = [r for r in results if r["answerable"] and r.get("evaluation")]

    if not answerable:
        return {}

    correctness_scores = [r["evaluation"]["correctness"] for r in answerable
                          if r["evaluation"]["correctness"] is not None]
    faithfulness_scores = [r["evaluation"]["faithfulness"] for r in answerable
                           if r["evaluation"]["faithfulness"] is not None]
    kw_scores = [r["evaluation"]["keyword_overlap"] for r in answerable
                 if r["evaluation"]["keyword_overlap"] is not None]

    n = len(correctness_scores)
    metrics = {}

    if correctness_scores:
        metrics["mean_correctness"] = round(statistics.mean(correctness_scores), 4)
        metrics["fully_correct_pct"] = round(
            sum(1 for s in correctness_scores if s == 2) / n * 100, 1
        )
        metrics["partially_correct_pct"] = round(
            sum(1 for s in correctness_scores if s == 1) / n * 100, 1
        )
        metrics["incorrect_pct"] = round(
            sum(1 for s in correctness_scores if s == 0) / n * 100, 1
        )

    if faithfulness_scores:
        metrics["mean_faithfulness"] = round(statistics.mean(faithfulness_scores), 4)
        metrics["fully_grounded_pct"] = round(
            sum(1 for s in faithfulness_scores if s == 2) / len(faithfulness_scores) * 100, 1
        )

    if kw_scores:
        metrics["mean_keyword_overlap"] = round(statistics.mean(kw_scores), 4)

    return metrics


def compute_citation_metrics(results: list[dict]) -> dict:
    """Evaluate citation quality: do retrieved papers match ground-truth papers?"""
    answerable = [r for r in results if r["answerable"]]
    if not answerable:
        return {}

    correct_citations = 0
    total = 0
    complete_citations = 0

    for r in answerable:
        gt_papers = set(r["ground_truth_paper_ids"])
        if not gt_papers:
            continue
        total += 1
        # Top-5 retrieved papers (what was used for generation context)
        retrieved_papers = set(
            r["retrieved"][i]["paper_id"]
            for i in range(min(DEFAULT_TOP_K, len(r["retrieved"])))
        )

        # Citation correctness: at least one GT paper in retrieved
        if gt_papers & retrieved_papers:
            correct_citations += 1

        # Citation completeness: all GT papers in retrieved
        if gt_papers <= retrieved_papers:
            complete_citations += 1

    return {
        "citation_correctness_rate": round(correct_citations / total, 4) if total else 0.0,
        "citation_completeness_rate": round(complete_citations / total, 4) if total else 0.0,
        "total_evaluated": total,
    }


def compute_abstention_metrics(results: list[dict]) -> dict:
    """Evaluate abstention accuracy on unanswerable questions."""
    unanswerable = [r for r in results if not r["answerable"] and r.get("evaluation")]

    if not unanswerable:
        return {}

    correct = sum(1 for r in unanswerable if r["evaluation"].get("correct_abstention"))
    total = len(unanswerable)

    return {
        "abstention_accuracy": round(correct / total, 4) if total else 0.0,
        "hallucination_rate": round((total - correct) / total, 4) if total else 0.0,
        "total_unanswerable": total,
        "correct_abstentions": correct,
    }


def compute_latency_metrics(results: list[dict]) -> dict:
    """Compute latency statistics."""
    latencies = [r["latency"]["total_s"] for r in results if r.get("latency")]
    gen_latencies = [r["latency"]["generation_s"] for r in results if r.get("latency")]

    if not latencies:
        return {}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    prompt_tokens = [r["tokens"].get("prompt_eval_count", 0) for r in results
                     if r.get("tokens")]
    gen_tokens = [r["tokens"].get("eval_count", 0) for r in results
                  if r.get("tokens")]

    return {
        "mean_latency_s": round(statistics.mean(latencies), 2),
        "p50_latency_s": round(sorted_lat[n // 2], 2),
        "p95_latency_s": round(sorted_lat[int(n * 0.95)], 2),
        "min_latency_s": round(sorted_lat[0], 2),
        "max_latency_s": round(sorted_lat[-1], 2),
        "mean_generation_s": round(statistics.mean(gen_latencies), 2),
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens)) if prompt_tokens else 0,
        "mean_gen_tokens": round(statistics.mean(gen_tokens)) if gen_tokens else 0,
        "error_count": sum(1 for r in results if r.get("error")),
        "error_rate": round(sum(1 for r in results if r.get("error")) / len(results), 4),
    }


def compute_breakdown(results: list[dict], group_key: str) -> dict:
    """Compute retrieval + answer metrics grouped by a key (question_type, difficulty, paper)."""
    groups = defaultdict(list)
    for r in results:
        if group_key == "paper":
            for pid in r.get("ground_truth_paper_ids", []):
                groups[pid].append(r)
            if not r.get("ground_truth_paper_ids"):
                groups["[unanswerable]"].append(r)
        else:
            groups[r.get(group_key, "unknown")].append(r)

    breakdown = {}
    for key, group_results in sorted(groups.items()):
        retrieval = compute_retrieval_metrics(group_results)
        answer = compute_answer_metrics(group_results)
        breakdown[key] = {**retrieval, **answer, "count": len(group_results)}

    return breakdown


def classify_errors(results: list[dict]) -> list[dict]:
    """Classify failures into error categories."""
    errors = []

    for r in results:
        if r.get("error"):
            errors.append({
                "question_id": r["question_id"],
                "failure_type": "system_error",
                "what_happened": f"System error: {r['error']}",
                "generated_answer": r["generated_answer"][:200],
                "expected_answer": (r.get("expected_answer") or "")[:200],
            })
            continue

        ev = r.get("evaluation", {})

        if not r["answerable"]:
            # Abstention failures
            if not ev.get("correct_abstention"):
                errors.append({
                    "question_id": r["question_id"],
                    "failure_type": "abstention_failure",
                    "what_happened": "Model answered an unanswerable question instead of refusing.",
                    "generated_answer": r["generated_answer"][:300],
                    "expected_answer": "Should abstain / refuse to answer",
                    "reasoning": ev.get("judge_reasoning", ""),
                })
            continue

        # Answerable questions
        gt_ids = set(r.get("ground_truth_chunk_ids", []))
        top5_ids = set(r["retrieved_chunk_ids"][:DEFAULT_TOP_K])
        top10_ids = set(r["retrieved_chunk_ids"][:EVAL_TOP_K])
        correctness = ev.get("correctness", 1)
        faithfulness = ev.get("faithfulness", 1)

        if correctness == 0:
            if not (gt_ids & top5_ids):
                # Correct evidence wasn't in top-5
                if gt_ids & top10_ids:
                    failure_type = "ranking_failure"
                    msg = "Ground-truth chunks exist in top-10 but not in top-5 used for generation."
                else:
                    failure_type = "retrieval_failure"
                    msg = "Ground-truth chunks were not retrieved in top-10."
            else:
                failure_type = "generation_failure"
                msg = "Evidence was retrieved but the answer is incorrect."
            errors.append({
                "question_id": r["question_id"],
                "failure_type": failure_type,
                "what_happened": msg,
                "relevant_chunks_in_top5": sorted(gt_ids & top5_ids),
                "relevant_chunks_in_top10": sorted(gt_ids & top10_ids),
                "generated_answer": r["generated_answer"][:300],
                "expected_answer": (r.get("expected_answer") or "")[:200],
                "reasoning": ev.get("judge_reasoning", ""),
            })
        elif faithfulness == 0:
            errors.append({
                "question_id": r["question_id"],
                "failure_type": "grounding_failure",
                "what_happened": "Answer contains mostly unsupported claims.",
                "generated_answer": r["generated_answer"][:300],
                "expected_answer": (r.get("expected_answer") or "")[:200],
                "reasoning": ev.get("judge_reasoning", ""),
            })

    return errors


# ===================================================================
# REPORT GENERATION
# ===================================================================

def generate_report(
    results: list[dict],
    retrieval_metrics: dict,
    answer_metrics: dict,
    citation_metrics: dict,
    abstention_metrics: dict,
    latency_metrics: dict,
    type_breakdown: dict,
    difficulty_breakdown: dict,
    paper_breakdown: dict,
    error_analysis: list[dict],
    repro_info: dict,
) -> str:
    """Generate the BASELINE_REPORT.md markdown string."""

    answerable_results = [r for r in results if r["answerable"]]
    unanswerable_results = [r for r in results if not r["answerable"]]

    # --- Section 1: Executive Summary ---
    report = f"""# Baseline RAG Evaluation Report

**Generated**: {repro_info['timestamp']}
**Git Commit**: `{repro_info['git_commit']}`
**Branch**: `{repro_info['git_branch']}`

---

## 1. Executive Summary

This report evaluates the current baseline Scientific Research Copilot RAG system against a curated
evaluation dataset of **{len(results)} questions** ({len(answerable_results)} answerable, {len(unanswerable_results)} unanswerable)
covering **10 scientific papers** with **374 indexed chunks** (319 text + 55 figure).

**Key Findings**:
- **Retrieval**: Hit@5 = {retrieval_metrics.get('hit_at_5', 0):.1%}, MRR = {retrieval_metrics.get('mrr', 0):.4f}
- **Answer Quality**: {answer_metrics.get('fully_correct_pct', 0):.1f}% fully correct, {answer_metrics.get('mean_correctness', 0):.2f}/2.0 mean correctness
- **Faithfulness**: {answer_metrics.get('mean_faithfulness', 0):.2f}/2.0 mean faithfulness
- **Abstention**: {abstention_metrics.get('abstention_accuracy', 0):.1%} accuracy on unanswerable questions
- **Latency**: {latency_metrics.get('mean_latency_s', 0):.1f}s average end-to-end

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
| Total questions | {len(results)} |
| Answerable | {len(answerable_results)} |
| Unanswerable | {len(unanswerable_results)} |
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

*Evaluated on {len(answerable_results)} answerable questions only.*

| Metric | Score |
|---|---|
| **Hit@1** | {retrieval_metrics.get('hit_at_1', 0):.1%} |
| **Hit@3** | {retrieval_metrics.get('hit_at_3', 0):.1%} |
| **Hit@5** | {retrieval_metrics.get('hit_at_5', 0):.1%} |
| **Hit@10** | {retrieval_metrics.get('hit_at_10', 0):.1%} |
| **Recall@1** | {retrieval_metrics.get('recall_at_1', 0):.4f} |
| **Recall@3** | {retrieval_metrics.get('recall_at_3', 0):.4f} |
| **Recall@5** | {retrieval_metrics.get('recall_at_5', 0):.4f} |
| **Recall@10** | {retrieval_metrics.get('recall_at_10', 0):.4f} |
| **MRR** | {retrieval_metrics.get('mrr', 0):.4f} |

---

## 6. Answer Quality

*Evaluated on {len(answerable_results)} answerable questions.*

| Metric | Score |
|---|---|
| **Mean Correctness** | {answer_metrics.get('mean_correctness', 0):.2f} / 2.0 |
| **Fully Correct (2)** | {answer_metrics.get('fully_correct_pct', 0):.1f}% |
| **Partially Correct (1)** | {answer_metrics.get('partially_correct_pct', 0):.1f}% |
| **Incorrect (0)** | {answer_metrics.get('incorrect_pct', 0):.1f}% |
| **Mean Keyword Overlap** | {answer_metrics.get('mean_keyword_overlap', 0):.4f} |

---

## 7. Faithfulness / Groundedness

| Metric | Score |
|---|---|
| **Mean Faithfulness** | {answer_metrics.get('mean_faithfulness', 0):.2f} / 2.0 |
| **Fully Grounded (2)** | {answer_metrics.get('fully_grounded_pct', 0):.1f}% |

---

## 8. Citation Quality

| Metric | Score |
|---|---|
| **Citation Correctness** | {citation_metrics.get('citation_correctness_rate', 0):.1%} |
| **Citation Completeness** | {citation_metrics.get('citation_completeness_rate', 0):.1%} |

*Citation correctness: ≥1 ground-truth paper appears in retrieved top-5.*
*Citation completeness: ALL ground-truth papers appear in retrieved top-5.*

---

## 9. Abstention / Unanswerable Performance

| Metric | Score |
|---|---|
| **Abstention Accuracy** | {abstention_metrics.get('abstention_accuracy', 0):.1%} |
| **Hallucination Rate** | {abstention_metrics.get('hallucination_rate', 0):.1%} |
| **Total Unanswerable** | {abstention_metrics.get('total_unanswerable', 0)} |
| **Correct Abstentions** | {abstention_metrics.get('correct_abstentions', 0)} |

### Individual Unanswerable Results
"""
    for r in unanswerable_results:
        ev = r.get("evaluation", {})
        status = "✅ Correctly abstained" if ev.get("correct_abstention") else "❌ Hallucinated answer"
        report += f"\n**{r['question_id']}**: {r['question'][:100]}...\n"
        report += f"- Status: {status}\n"
        report += f"- Answer excerpt: {r['generated_answer'][:200]}...\n"
        report += f"- Judge reasoning: {ev.get('judge_reasoning', 'N/A')}\n"

    # --- Section 10: Performance by Question Type ---
    report += f"""
---

## 10. Performance by Question Type

| Type | Count | Hit@5 | Recall@5 | MRR | Correctness | Faithfulness |
|---|---|---|---|---|---|---|
"""
    for qtype, m in type_breakdown.items():
        report += (
            f"| {qtype} | {m['count']} | "
            f"{m.get('hit_at_5', 0):.1%} | {m.get('recall_at_5', 0):.4f} | {m.get('mrr', 0):.4f} | "
            f"{m.get('mean_correctness', '-')}"
            f"{'' if m.get('mean_correctness') is None else ''} | "
            f"{m.get('mean_faithfulness', '-')}"
            f"{'' if m.get('mean_faithfulness') is None else ''} |\n"
        )

    report += f"""
---

## 11. Performance by Difficulty

| Difficulty | Count | Hit@5 | Recall@5 | MRR | Correctness | Faithfulness |
|---|---|---|---|---|---|---|
"""
    for diff, m in difficulty_breakdown.items():
        report += (
            f"| {diff} | {m['count']} | "
            f"{m.get('hit_at_5', 0):.1%} | {m.get('recall_at_5', 0):.4f} | {m.get('mrr', 0):.4f} | "
            f"{m.get('mean_correctness', '-')} | "
            f"{m.get('mean_faithfulness', '-')} |\n"
        )

    # --- Section 12: Performance by Paper ---
    report += f"""
---

## 12. Performance by Paper

| Paper ID | Count | Hit@5 | Recall@5 | MRR | Correctness |
|---|---|---|---|---|---|
"""
    for paper, m in paper_breakdown.items():
        report += (
            f"| {paper} | {m['count']} | "
            f"{m.get('hit_at_5', 0):.1%} | {m.get('recall_at_5', 0):.4f} | {m.get('mrr', 0):.4f} | "
            f"{m.get('mean_correctness', '-')} |\n"
        )

    # --- Section 13: Error Analysis ---
    report += f"""
---

## 13. Error Analysis

**Total failures identified**: {len(error_analysis)}

### Failure Type Distribution
"""
    failure_counts = defaultdict(int)
    for e in error_analysis:
        failure_counts[e["failure_type"]] += 1
    for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        report += f"- **{ftype}**: {count}\n"

    report += "\n### Representative Failures\n"
    for e in error_analysis[:15]:  # Show up to 15
        report += f"""
#### {e['question_id']} — {e['failure_type']}
- **What happened**: {e['what_happened']}
- **Generated answer**: {e.get('generated_answer', 'N/A')[:200]}...
- **Expected answer**: {e.get('expected_answer', 'N/A')[:200]}...
- **Reasoning**: {e.get('reasoning', 'N/A')}
"""

    # --- Section 14: Main Bottlenecks ---
    report += """
---

## 14. Main Bottlenecks

"""
    # Determine bottlenecks from metrics
    bottlenecks = []
    hit5 = retrieval_metrics.get('hit_at_5', 0)
    if hit5 < 0.7:
        bottlenecks.append(f"1. **Retrieval** (Hit@5={hit5:.1%}): Dense-only retrieval misses relevant chunks for many questions. Consider hybrid search (BM25 + dense) or query expansion.")
    mrr = retrieval_metrics.get('mrr', 0)
    if mrr < 0.5:
        bottlenecks.append(f"2. **Ranking** (MRR={mrr:.4f}): Even when relevant chunks are retrieved, they often appear at lower ranks. Consider reranking with a cross-encoder.")
    correctness = answer_metrics.get('mean_correctness', 0)
    if correctness < 1.5:
        bottlenecks.append(f"3. **Generation Quality** (Mean correctness={correctness:.2f}/2.0): The 3B model struggles with complex scientific reasoning. Consider a larger LLM or improved prompting.")
    faithfulness_val = answer_metrics.get('mean_faithfulness', 0)
    if faithfulness_val < 1.5:
        bottlenecks.append(f"4. **Faithfulness** (Mean={faithfulness_val:.2f}/2.0): Significant grounding issues — model generates unsupported claims.")
    abstention_acc = abstention_metrics.get('abstention_accuracy', 0)
    if abstention_acc < 0.75:
        bottlenecks.append(f"5. **Abstention** ({abstention_acc:.1%}): Model fails to refuse unanswerable questions. Add answerability detection.")

    if not bottlenecks:
        bottlenecks.append("No critical bottlenecks identified — system performs well across dimensions.")

    for b in bottlenecks:
        report += f"{b}\n\n"

    # --- Section 15: Recommended Next Experiments ---
    report += f"""---

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
| Git Commit | `{repro_info['git_commit']}` |
| Git Branch | `{repro_info['git_branch']}` |
| Python Version | `{repro_info['python_version']}` |
| Embedding Model | `{repro_info['embedding_model']}` |
| LLM Model | `{repro_info['llm_model']}` |
| Vector DB | Qdrant local at `data/qdrant` |
| Collection | `{repro_info['collection_name']}` |
| Retrieval K (eval) | {EVAL_TOP_K} (generation uses top-{DEFAULT_TOP_K}) |
| Eval Dataset | `evaluation/questions.json` ({len(results)} questions) |
| Timestamp | {repro_info['timestamp']} |
| Command | `cd c:\\Ki_OJT\\scientific-research-copilot && .venv\\Scripts\\python.exe evaluation/run_baseline.py` |
| Torch | `{repro_info.get('torch_version', 'N/A')}` |
| sentence-transformers | `{repro_info.get('st_version', 'N/A')}` |
| qdrant-client | `{repro_info.get('qdrant_version', 'N/A')}` |
| ollama | `{repro_info.get('ollama_version', 'N/A')}` |
"""

    return report


def generate_csv_summary(results: list[dict], csv_path: Path):
    """Generate a CSV summary of per-question results."""
    import csv
    headers = [
        "question_id", "question_type", "difficulty", "answerable",
        "hit_at_5", "recall_at_5", "mrr_rank",
        "correctness", "faithfulness", "keyword_overlap",
        "correct_abstention", "latency_s", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            gt_ids = set(r.get("ground_truth_chunk_ids", []))
            top5 = set(r["retrieved_chunk_ids"][:5])
            top5_list = r["retrieved_chunk_ids"][:5]

            hit5 = 1 if (gt_ids & top5) else 0
            recall5 = len(gt_ids & top5) / len(gt_ids) if gt_ids else ""
            rr_rank = ""
            for rank, cid in enumerate(r["retrieved_chunk_ids"], 1):
                if cid in gt_ids:
                    rr_rank = rank
                    break

            ev = r.get("evaluation", {})
            writer.writerow({
                "question_id": r["question_id"],
                "question_type": r["question_type"],
                "difficulty": r["difficulty"],
                "answerable": r["answerable"],
                "hit_at_5": hit5 if r["answerable"] else "",
                "recall_at_5": f"{recall5:.4f}" if isinstance(recall5, float) else "",
                "mrr_rank": rr_rank,
                "correctness": ev.get("correctness", ""),
                "faithfulness": ev.get("faithfulness", ""),
                "keyword_overlap": ev.get("keyword_overlap", ""),
                "correct_abstention": ev.get("correct_abstention", ""),
                "latency_s": r["latency"]["total_s"],
                "error": r.get("error", ""),
            })


# ===================================================================
# MAIN
# ===================================================================

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    log("=" * 60)
    log("BASELINE RAG EVALUATION")
    log("=" * 60)

    # --- Reproducibility info ---
    repro_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "collection_name": COLLECTION_NAME,
        "torch_version": torch.__version__,
    }
    try:
        import sentence_transformers
        repro_info["st_version"] = sentence_transformers.__version__
    except Exception:
        repro_info["st_version"] = "N/A"
    try:
        import qdrant_client
        repro_info["qdrant_version"] = qdrant_client.__version__
    except Exception:
        repro_info["qdrant_version"] = "N/A"
    try:
        repro_info["ollama_version"] = ollama.__version__
    except Exception:
        repro_info["ollama_version"] = "N/A"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
        branch = "unknown"
    repro_info["git_commit"] = commit
    repro_info["git_branch"] = branch

    # --- Load questions ---
    log(f"Loading questions from {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log(f"Loaded {len(questions)} questions")

    # --- Initialize models ---
    log("Initializing embedding model and Qdrant client...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  Device: {device}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embed_model.max_seq_length = 4096
    qdrant = QdrantClient(path=DB_PATH)
    log("Models initialized")

    # --- Verify Qdrant collection ---
    if not qdrant.collection_exists(COLLECTION_NAME):
        log(f"ERROR: Collection '{COLLECTION_NAME}' not found in Qdrant!")
        sys.exit(1)
    coll_info = qdrant.get_collection(COLLECTION_NAME)
    log(f"Collection '{COLLECTION_NAME}': {coll_info.points_count} points")

    # --- Create results directory ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ===== PHASE 1: RAG Execution =====
    log("")
    log("=" * 60)
    log("PHASE 1: RAG Execution")
    log("=" * 60)
    results = []
    for i, q in enumerate(questions, 1):
        log(f"  [{i:02d}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        result = run_rag_on_question(q, qdrant, embed_model)
        results.append(result)
        log(f"         → {result['latency']['total_s']:.1f}s | "
            f"Top chunk: {result['retrieved_chunk_ids'][0] if result['retrieved_chunk_ids'] else 'N/A'} | "
            f"Error: {result.get('error', 'None')}")

    # Close Qdrant before Phase 2 (release lock)
    qdrant.close()
    log(f"\nPhase 1 complete: {len(results)} questions processed")

    # ===== PHASE 2: LLM Judge Evaluation =====
    log("")
    log("=" * 60)
    log("PHASE 2: LLM Judge Evaluation")
    log("=" * 60)
    for i, r in enumerate(results, 1):
        log(f"  [{i:02d}/{len(results)}] Evaluating {r['question_id']}...")
        scores = evaluate_single(r)
        r["evaluation"] = scores
        if r["answerable"]:
            log(f"         → Correctness: {scores['correctness']}, "
                f"Faithfulness: {scores['faithfulness']}, "
                f"Keyword: {scores['keyword_overlap']:.3f}")
        else:
            log(f"         → Abstention: {scores['correct_abstention']}")

    log(f"\nPhase 2 complete")

    # ===== PHASE 3: Compute Metrics & Generate Report =====
    log("")
    log("=" * 60)
    log("PHASE 3: Metrics & Report Generation")
    log("=" * 60)

    retrieval_metrics = compute_retrieval_metrics(results)
    answer_metrics = compute_answer_metrics(results)
    citation_metrics = compute_citation_metrics(results)
    abstention_metrics = compute_abstention_metrics(results)
    latency_metrics = compute_latency_metrics(results)

    log("Retrieval metrics:")
    for k, v in retrieval_metrics.items():
        log(f"  {k}: {v}")

    log("Answer metrics:")
    for k, v in answer_metrics.items():
        log(f"  {k}: {v}")

    # Breakdowns
    type_breakdown = compute_breakdown(results, "question_type")
    difficulty_breakdown = compute_breakdown(results, "difficulty")
    paper_breakdown = compute_breakdown(results, "paper")

    # Error analysis
    error_analysis = classify_errors(results)
    log(f"Error analysis: {len(error_analysis)} failures identified")

    # Save raw results
    log(f"Saving results to {RESULTS_PATH}")
    # Remove any non-serializable objects
    results_clean = json.loads(json.dumps(results, default=str))
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": repro_info,
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "citation_metrics": citation_metrics,
            "abstention_metrics": abstention_metrics,
            "latency_metrics": latency_metrics,
            "type_breakdown": type_breakdown,
            "difficulty_breakdown": difficulty_breakdown,
            "paper_breakdown": paper_breakdown,
            "error_analysis": error_analysis,
            "results": results_clean,
        }, f, indent=2, ensure_ascii=False)

    # Generate CSV
    log(f"Saving CSV summary to {CSV_PATH}")
    generate_csv_summary(results, CSV_PATH)

    # Generate report
    log(f"Generating BASELINE_REPORT.md at {REPORT_PATH}")
    report = generate_report(
        results=results,
        retrieval_metrics=retrieval_metrics,
        answer_metrics=answer_metrics,
        citation_metrics=citation_metrics,
        abstention_metrics=abstention_metrics,
        latency_metrics=latency_metrics,
        type_breakdown=type_breakdown,
        difficulty_breakdown=difficulty_breakdown,
        paper_breakdown=paper_breakdown,
        error_analysis=error_analysis,
        repro_info=repro_info,
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # --- Final Summary ---
    log("")
    log("=" * 60)
    log("EVALUATION COMPLETE")
    log("=" * 60)
    log(f"Results:  {RESULTS_PATH}")
    log(f"Report:   {REPORT_PATH}")
    log(f"CSV:      {CSV_PATH}")
    log("")
    log("Key Metrics:")
    log(f"  Hit@1:  {retrieval_metrics.get('hit_at_1', 0):.1%}")
    log(f"  Hit@5:  {retrieval_metrics.get('hit_at_5', 0):.1%}")
    log(f"  Hit@10: {retrieval_metrics.get('hit_at_10', 0):.1%}")
    log(f"  MRR:    {retrieval_metrics.get('mrr', 0):.4f}")
    log(f"  Correctness: {answer_metrics.get('mean_correctness', 0):.2f}/2.0")
    log(f"  Faithfulness: {answer_metrics.get('mean_faithfulness', 0):.2f}/2.0")
    log(f"  Abstention:  {abstention_metrics.get('abstention_accuracy', 0):.1%}")
    log(f"  Avg Latency: {latency_metrics.get('mean_latency_s', 0):.1f}s")


if __name__ == "__main__":
    main()

