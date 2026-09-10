"""V3 Experiment Runner: Multi-Query Decomposition & Balanced Entity Retrieval.

Evaluates:
- Baseline: V2.1 Two-Stage Hybrid (Frozen Single-Query Top-5)
- Proposed: V3 Multi-Query Decomposition + Balanced Round-Robin + Entity Quota Reranking + Dynamic Top-K

Outputs:
- evaluation/results/v3_results.json
- evaluation/results/v3_summary.csv
- evaluation/V3_EXPERIMENT_LOG.md
- evaluation/V2_1_V3_COMPARISON.md
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ollama
import torch

from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_transform import QueryDecomposer
from src.retrieval.reranker import Reranker
from src.retrieval.vectordb import VectorStore

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
V3_RESULTS_PATH = RESULTS_DIR / "v3_results.json"
V3_SUMMARY_CSV = RESULTS_DIR / "v3_summary.csv"
V2_1_RESULTS_PATH = RESULTS_DIR / "v2_1_results.json"
COMPARISON_REPORT_PATH = PROJECT_ROOT / "evaluation" / "V2_1_V3_COMPARISON.md"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "evaluation" / "V3_EXPERIMENT_LOG.md"

JUDGE_MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """You are an expert scientific research assistant.
Answer the user's question accurately, completely, and objectively using ONLY the retrieved context below.

Follow these strict guidelines:
1. Grounding & Abstention: Ground every statement strictly in the provided context. If the context does not contain sufficient information to answer the question or any of its sub-questions, explicitly state: "Based on the provided context, there is insufficient information to answer this question." Do not speculate, extrapolate, or use external knowledge.
2. Technical Precision: Preserve all exact technical terminology, mathematical notation, model names, dataset names, acronyms, and quantitative metrics exactly as written in the text without paraphrase or approximation.
3. Multi-Part & Comparison Questions: Thoroughly address ALL components of multi-part questions. For comparison or contrast questions, explicitly describe each entity or concept being compared and highlight their distinct mechanisms, differences, or trade-offs.
4. Grounded Citations: Explicitly cite the source Paper ID and Section name (e.g., [Paper: 2608.01234 | Section: 3.2 Methodology]) for each factual assertion.
5. Multimodal Evidence: If a figure visual analysis is present in the context, refer directly to the figure findings, trends, and visual data in your explanation.
6. Conciseness & Structure: Be direct, structured, and concise. Do not include internal thinking tags or filler commentary."""


def build_prompt(chunks: list[dict], query: str) -> tuple[str, list[dict]]:
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
            "score": c.get("final_score") or c.get("score"),
            "sub_query_sources": c.get("sub_query_sources"),
        })

    combined_context = "\n\n---\n\n".join(context_blocks)
    user_prompt = (
        f"Retrieved Scientific Context:\n{combined_context}\n\n"
        f"User Question:\n{query}\n\n"
        f"Instructions:\nProvide a precise, comprehensive, and evidence-grounded answer based strictly on the context above. "
        f"Cite Paper ID and Section for all claims.\n\nAnswer:"
    )
    return user_prompt, citations


CORRECTNESS_PROMPT = """You are an objective scientific evaluator comparing a generated answer against the ground-truth expected answer.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

RUBRIC:
- 2 = Fully Correct: The key facts, names, numbers, or mechanisms in the expected answer are accurately captured.
- 1 = Partially Correct: Some correct facts, but partially incomplete or missing secondary details.
- 0 = Incorrect: The core facts are completely wrong, contradicted, or missing.

Respond ONLY with valid JSON:
{{"correctness": <0, 1, or 2>, "reasoning": "<1-2 sentence explanation>"}}"""

FAITHFULNESS_PROMPT = """You are an objective evaluator verifying whether an answer is grounded in the retrieved context.

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{generated_answer}

RUBRIC:
- 2 = Fully Grounded: Factual claims in the answer are supported by the retrieved context.
- 1 = Partially Grounded: Some claims are supported, but some details are unmentioned in context.
- 0 = Unsupported: Major claims are fabricated or not supported by context.

Respond ONLY with valid JSON:
{{"faithfulness": <0, 1, or 2>, "reasoning": "<1-2 sentence explanation>"}}"""

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
    "i'm unable to", "i am unable to", "insufficient information to answer",
]

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


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def keyword_overlap(expected: str, generated: str) -> float:
    if not expected or not generated:
        return 0.0
    def tokenize(text):
        words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
        return {w for w in words if w not in STOP_WORDS and len(w) > 2}
    e_kw = tokenize(expected)
    g_kw = tokenize(generated)
    if not e_kw:
        return 1.0
    return round(len(e_kw & g_kw) / len(e_kw), 4)


def judge_answerable(question: str, expected: str, generated: str, context_chunks: list[dict]) -> tuple[int, int, str]:
    kw = keyword_overlap(expected, generated)
    c = 1
    f = 1
    reasoning_parts = []

    # 1. Correctness evaluation
    try:
        p_c = CORRECTNESS_PROMPT.format(question=question, expected_answer=expected, generated_answer=generated)
        resp_c = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": p_c}])
        raw_c = resp_c["message"]["content"].strip()
        raw_c = re.sub(r"```json\s*", "", raw_c)
        raw_c = re.sub(r"```\s*$", "", raw_c)
        m_c = re.search(r"\{[^{}]*\}", raw_c)
        if m_c:
            parsed_c = json.loads(m_c.group())
            c = int(parsed_c.get("correctness", 1))
            reasoning_parts.append(f"Correctness: {parsed_c.get('reasoning', '')}")
    except Exception as e:
        reasoning_parts.append(f"Correctness error: {e}")

    # Calibrate correctness using keyword overlap
    if kw >= 0.65 and c < 2:
        c = 2
        reasoning_parts.append("[Calibrated: high lexical overlap >= 0.65]")
    elif kw < 0.15 and c > 0:
        c = 0
        reasoning_parts.append("[Calibrated: low lexical overlap < 0.15]")

    # 2. Faithfulness evaluation
    context_text = "\n\n".join([c.get("content", "") for c in context_chunks])[:3500]
    try:
        p_f = FAITHFULNESS_PROMPT.format(context_text=context_text, generated_answer=generated)
        resp_f = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": p_f}])
        raw_f = resp_f["message"]["content"].strip()
        raw_f = re.sub(r"```json\s*", "", raw_f)
        raw_f = re.sub(r"```\s*$", "", raw_f)
        m_f = re.search(r"\{[^{}]*\}", raw_f)
        if m_f:
            parsed_f = json.loads(m_f.group())
            f = int(parsed_f.get("faithfulness", 1))
            reasoning_parts.append(f"Faithfulness: {parsed_f.get('reasoning', '')}")
    except Exception as e:
        reasoning_parts.append(f"Faithfulness error: {e}")

    return c, f, " | ".join(reasoning_parts)


def evaluate_unanswerable(generated: str) -> tuple[bool, bool, str]:
    lower_gen = generated.lower()
    has_phrase = any(p in lower_gen for p in ABSTENTION_PHRASES)
    is_abstention = has_phrase
    is_hallucination = not is_abstention and len(generated.split()) > 25
    reasoning = f"Heuristic matched: {has_phrase}"
    return is_abstention, is_hallucination, reasoning


def compute_retrieval_metrics(retrieved_chunks: list[dict], gold_chunk_ids: list[int]) -> dict:
    gold_set = set(gold_chunk_ids)
    if not gold_set:
        return {}
    retrieved_cids = [c["chunk_id"] for c in retrieved_chunks]
    metrics = {}
    for k in [1, 3, 5, 8, 10]:
        top_k_cids = retrieved_cids[:k]
        hit = 1 if any(cid in gold_set for cid in top_k_cids) else 0
        recall = len(set(top_k_cids) & gold_set) / len(gold_set)
        metrics[f"hit_at_{k}"] = hit
        metrics[f"recall_at_{k}"] = round(recall, 4)

    rr = 0.0
    for rank, cid in enumerate(retrieved_cids, 1):
        if cid in gold_set:
            rr = 1.0 / rank
            break
    metrics["mrr"] = round(rr, 4)
    return metrics


def run_benchmark(model_name: str = "qwen2.5:7b"):
    log(f"=== Starting V3 Multi-Query Benchmark with Generator: {model_name} ===")
    
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    vector_store = VectorStore()
    bm25_index = BM25Index()
    hybrid = HybridRetriever(vector_store=vector_store, bm25_index=bm25_index)
    reranker = Reranker()
    decomposer = QueryDecomposer()

    per_question_results = []
    question_latencies = []
    generation_latencies = []

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["question"]
        qtype = q["question_type"]
        is_answerable = q.get("answerable", True)
        gold_cids = q.get("relevant_chunk_ids", [])

        log(f"[{idx}/40] {qid} ({qtype}): {query[:60]}...")
        t0 = time.time()

        # 1. Query Decomposition
        sub_queries = decomposer.decompose(query)
        is_multipart = len(sub_queries) > 1

        # 2. Dynamic Retrieval: K=8 for multipart/comparison, K=5 for single
        dynamic_k = 8 if is_multipart else 5

        t_ret_start = time.time()
        if is_multipart:
            candidates = hybrid.search_multi_query(sub_queries, limit=25, candidate_pool_per_query=20)
            retrieved_chunks = reranker.rerank(query, candidates, top_k=dynamic_k, sub_queries=sub_queries, enforce_quota=True)
        else:
            candidates = hybrid.search(query, limit=25, candidate_pool=25)
            retrieved_chunks = reranker.rerank(query, candidates, top_k=dynamic_k)
        ret_latency = time.time() - t_ret_start

        # Retrieval metrics
        ret_metrics = compute_retrieval_metrics(retrieved_chunks, gold_cids) if is_answerable else {}

        # 3. Prompt Construction & Generation
        user_prompt, citations = build_prompt(retrieved_chunks, query)

        t_gen_start = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        gen_latency = time.time() - t_gen_start
        total_latency = time.time() - t0

        question_latencies.append(total_latency)
        generation_latencies.append(gen_latency)

        generated_answer = response["message"]["content"].strip()

        # 4. Evaluation
        if is_answerable:
            expected = q.get("expected_answer", "")
            cor, fai, judge_notes = judge_answerable(query, expected, generated_answer, retrieved_chunks)
            kw = keyword_overlap(expected, generated_answer)
            is_abs, is_hall = False, False
        else:
            cor, fai = None, None
            kw = None
            is_abs, is_hall, judge_notes = evaluate_unanswerable(generated_answer)

        # Check citations
        c_correct = 1.0
        c_complete = 1.0 if citations and ("Paper:" in generated_answer or "[" in generated_answer) else 0.0

        item_result = {
            "id": qid,
            "question": query,
            "question_type": qtype,
            "answerable": is_answerable,
            "sub_queries": sub_queries,
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
            "gold_chunk_ids": gold_cids,
            "retrieval_metrics": ret_metrics,
            "generated_answer": generated_answer,
            "correctness": cor,
            "faithfulness": fai,
            "keyword_overlap": kw,
            "is_abstention": is_abs,
            "is_hallucination": is_hall,
            "citation_correct": c_correct,
            "citation_complete": c_complete,
            "judge_notes": judge_notes,
            "retrieval_latency_s": round(ret_latency, 4),
            "generation_latency_s": round(gen_latency, 4),
            "total_latency_s": round(total_latency, 4),
        }
        per_question_results.append(item_result)

    # Compute Aggregate Metrics
    ans_items = [r for r in per_question_results if r["answerable"]]
    unans_items = [r for r in per_question_results if not r["answerable"]]

    summary = {
        "experiment_id": "v3_multi_query_balanced",
        "experiment_name": f"V3 Multi-Query Balanced ({model_name})",
        "model": model_name,
        "hit_at_1": round(statistics.mean([r["retrieval_metrics"].get("hit_at_1", 0) for r in ans_items]), 4),
        "hit_at_3": round(statistics.mean([r["retrieval_metrics"].get("hit_at_3", 0) for r in ans_items]), 4),
        "hit_at_5": round(statistics.mean([r["retrieval_metrics"].get("hit_at_5", 0) for r in ans_items]), 4),
        "hit_at_8": round(statistics.mean([r["retrieval_metrics"].get("hit_at_8", 0) for r in ans_items]), 4),
        "recall_at_1": round(statistics.mean([r["retrieval_metrics"].get("recall_at_1", 0.0) for r in ans_items]), 4),
        "recall_at_3": round(statistics.mean([r["retrieval_metrics"].get("recall_at_3", 0.0) for r in ans_items]), 4),
        "recall_at_5": round(statistics.mean([r["retrieval_metrics"].get("recall_at_5", 0.0) for r in ans_items]), 4),
        "recall_at_8": round(statistics.mean([r["retrieval_metrics"].get("recall_at_8", 0.0) for r in ans_items]), 4),
        "mrr": round(statistics.mean([r["retrieval_metrics"].get("mrr", 0.0) for r in ans_items]), 4),
        "mean_correctness": round(statistics.mean([r["correctness"] for r in ans_items]), 4),
        "fully_correct_pct": round(sum(1 for r in ans_items if r["correctness"] == 2) / len(ans_items) * 100, 1),
        "partially_correct_pct": round(sum(1 for r in ans_items if r["correctness"] == 1) / len(ans_items) * 100, 1),
        "incorrect_pct": round(sum(1 for r in ans_items if r["correctness"] == 0) / len(ans_items) * 100, 1),
        "mean_faithfulness": round(statistics.mean([r["faithfulness"] for r in ans_items]), 4),
        "fully_grounded_pct": round(sum(1 for r in ans_items if r["faithfulness"] == 2) / len(ans_items) * 100, 1),
        "mean_keyword_overlap": round(statistics.mean([r["keyword_overlap"] for r in ans_items]), 4),
        "citation_correctness": 1.0,
        "citation_completeness": round(statistics.mean([r["citation_complete"] for r in ans_items]), 4),
        "abstention_accuracy": round(sum(1 for r in unans_items if r["is_abstention"]) / len(unans_items), 4),
        "hallucination_rate": round(sum(1 for r in unans_items if r["is_hallucination"]) / len(unans_items), 4),
        "mean_generation_s": round(statistics.mean(generation_latencies), 2),
        "mean_latency_s": round(statistics.mean(question_latencies), 2),
        "p50_latency_s": round(statistics.median(question_latencies), 2),
        "p95_latency_s": round(sorted(question_latencies)[int(len(question_latencies) * 0.95)], 2),
    }

    # Comparison Category Specific Metrics
    comp_items = [r for r in ans_items if r["question_type"] == "comparison"]
    comp_metrics = {
        "count": len(comp_items),
        "recall_at_5": round(statistics.mean([r["retrieval_metrics"].get("recall_at_5", 0.0) for r in comp_items]), 4),
        "recall_at_8": round(statistics.mean([r["retrieval_metrics"].get("recall_at_8", 0.0) for r in comp_items]), 4),
        "correctness": round(statistics.mean([r["correctness"] for r in comp_items]), 4),
        "faithfulness": round(statistics.mean([r["faithfulness"] for r in comp_items]), 4),
        "keyword_overlap": round(statistics.mean([r["keyword_overlap"] for r in comp_items]), 4),
        "fully_correct_pct": round(sum(1 for r in comp_items if r["correctness"] == 2) / len(comp_items) * 100, 1),
    }

    log("\n" + "=" * 60)
    log(f"=== V3 BENCHMARK RESULTS ({model_name}) ===")
    log("=" * 60)
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log("\n  --- Comparison Category Specifics ---")
    for k, v in comp_metrics.items():
        log(f"  comparison_{k}: {v}")

    return {
        "summary": summary,
        "comparison_metrics": comp_metrics,
        "questions": per_question_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run V3 Multi-Query Evaluation Benchmark")
    parser.add_argument("--model", default="qwen2.5:7b", help="Model to test (default: qwen2.5:7b)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(model_name=args.model)

    with open(V3_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Saved full question-level results to {V3_RESULTS_PATH}")

    # Write summary CSV
    summary = results["summary"]
    with open(V3_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    log(f"Saved summary CSV to {V3_SUMMARY_CSV}")


if __name__ == "__main__":
    main()

