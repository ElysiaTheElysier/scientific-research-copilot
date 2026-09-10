"""V2.1 Experiment Runner: Generation Layer Optimization & Model Scaling.

Evaluates generation improvements with frozen V2.0 retrieval:
1. V2.0 Reproduced: qwen2.5:3b + V2.0 Prompt (Verification)
2. Ablation A: qwen2.5:3b + V2.1 Improved Prompt (Prompt Engineering Isolation)
3. Ablation B: qwen2.5:7b + V2.0 Prompt (Model Scaling Isolation)
4. Full V2.1: qwen2.5:7b + V2.1 Improved Prompt (Full Proposed System)

Outputs:
- evaluation/results/v2_1_results.json
- evaluation/results/v2_1_summary.csv
- evaluation/V2_1_EXPERIMENT_LOG.md
- evaluation/V2_0_V2_1_COMPARISON.md
"""

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
from src.retrieval.reranker import Reranker
from src.retrieval.vectordb import VectorStore

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
V2_1_RESULTS_PATH = RESULTS_DIR / "v2_1_results.json"
V2_1_SUMMARY_CSV = RESULTS_DIR / "v2_1_summary.csv"
COMPARISON_REPORT_PATH = PROJECT_ROOT / "evaluation" / "V2_0_V2_1_COMPARISON.md"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "evaluation" / "V2_1_EXPERIMENT_LOG.md"

TOP_K = 5
EVAL_TOP_K = 10
CANDIDATE_POOL = 25

JUDGE_MODEL = "qwen2.5:3b"

# V2.0 Prompt Specification
V2_0_SYSTEM_PROMPT = """You are an evidence-grounded scientific research assistant.
Answer the user's question using ONLY the retrieved context below.
Cite the relevant Paper ID and Section for your claims.
If a figure visual analysis is provided, refer to the figure and its findings."""

# V2.1 Improved Evidence-Grounded Prompt Specification
V2_1_SYSTEM_PROMPT = """You are an expert scientific research assistant.
Answer the user's question accurately, completely, and objectively using ONLY the retrieved context below.

Follow these strict guidelines:
1. Grounding & Abstention: Ground every statement strictly in the provided context. If the context does not contain sufficient information to answer the question or any of its sub-questions, explicitly state: "Based on the provided context, there is insufficient information to answer this question." Do not speculate, extrapolate, or use external knowledge.
2. Technical Precision: Preserve all exact technical terminology, mathematical notation, model names, dataset names, acronyms, and quantitative metrics exactly as written in the text without paraphrase or approximation.
3. Multi-Part & Comparison Questions: Thoroughly address ALL components of multi-part questions. For comparison or contrast questions, explicitly describe each entity or concept being compared and highlight their distinct mechanisms, differences, or trade-offs.
4. Grounded Citations: Explicitly cite the source Paper ID and Section name (e.g., [Paper: 2608.01234 | Section: 3.2 Methodology]) for each factual assertion.
5. Multimodal Evidence: If a figure visual analysis is present in the context, refer directly to the figure findings, trends, and visual data in your explanation.
6. Conciseness & Structure: Be direct, structured, and concise. Do not include internal thinking tags or filler commentary."""


def build_v2_0_prompt(chunks: list[dict], query: str) -> tuple[str, list[dict]]:
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
            "score": c.get("score"),
        })

    combined_context = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"
    return user_prompt, citations


def build_v2_1_prompt(chunks: list[dict], query: str) -> tuple[str, list[dict]]:
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
            "score": c.get("score"),
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
            c = max(0, min(2, int(parsed_c.get("correctness", 1))))
            reasoning_parts.append(f"Correctness: {parsed_c.get('reasoning', '')}")
    except Exception as e:
        reasoning_parts.append(f"Correctness error: {e}")

    # Dual-signal verification
    if c == 0 and kw >= 0.65:
        c = 2
        reasoning_parts.append(f"[Calibrated to 2 due to strong keyword match: {kw:.2f}]")
    elif c == 0 and kw >= 0.40:
        c = 1
        reasoning_parts.append(f"[Calibrated to 1 due to partial keyword match: {kw:.2f}]")
    elif c == 2 and kw < 0.15:
        c = 1
        reasoning_parts.append(f"[Demoted to 1 due to low keyword match: {kw:.2f}]")

    # 2. Faithfulness evaluation
    context_text = "\n\n".join([
        f"[{c.get('paper_id')} - {c.get('section')}]:\n{c.get('content', '')[:600]}"
        for c in context_chunks[:TOP_K]
    ])
    try:
        p_f = FAITHFULNESS_PROMPT.format(context_text=context_text[:2500], generated_answer=generated)
        resp_f = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": p_f}])
        raw_f = resp_f["message"]["content"].strip()
        raw_f = re.sub(r"```json\s*", "", raw_f)
        raw_f = re.sub(r"```\s*$", "", raw_f)
        m_f = re.search(r"\{[^{}]*\}", raw_f)
        if m_f:
            parsed_f = json.loads(m_f.group())
            f = max(0, min(2, int(parsed_f.get("faithfulness", 1))))
            reasoning_parts.append(f"Faithfulness: {parsed_f.get('reasoning', '')}")
    except Exception as e:
        reasoning_parts.append(f"Faithfulness error: {e}")

    return c, f, " | ".join(reasoning_parts)


def check_abstention(answer: str, question: str) -> tuple[bool, str]:
    lower = answer.lower()
    rule = any(p in lower for p in ABSTENTION_PHRASES)
    try:
        p = ABSTENTION_JUDGE_PROMPT.format(question=question, generated_answer=answer)
        resp = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": p}])
        raw = resp["message"]["content"].strip()
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        m = re.search(r"\{[^{}]*\}", raw)
        if m:
            parsed = json.loads(m.group())
            return parsed.get("correct_abstention", rule), parsed.get("reasoning", "")
    except Exception as e:
        pass
    return rule, "rule-based"


def compute_metrics(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    # Retrieval metrics
    K_vals = [1, 3, 5, 10]
    hits = {k: [] for k in K_vals}
    recalls = {k: [] for k in K_vals}
    rr_list = []

    for r in answerable:
        gt_ids = set(r["ground_truth_chunk_ids"])
        ret_ids = [c["chunk_id"] for c in r["retrieved"]]

        for k in K_vals:
            top_k_ids = set(ret_ids[:k])
            hit = 1 if (gt_ids & top_k_ids) else 0
            rec = len(gt_ids & top_k_ids) / len(gt_ids) if gt_ids else 0
            hits[k].append(hit)
            recalls[k].append(rec)

        rr = 0.0
        for rank, cid in enumerate(ret_ids, 1):
            if cid in gt_ids:
                rr = 1.0 / rank
                break
        rr_list.append(rr)

    ret_metrics = {}
    for k in K_vals:
        ret_metrics[f"hit_at_{k}"] = round(statistics.mean(hits[k]), 4) if hits[k] else 0.0
        ret_metrics[f"recall_at_{k}"] = round(statistics.mean(recalls[k]), 4) if recalls[k] else 0.0
    ret_metrics["mrr"] = round(statistics.mean(rr_list), 4) if rr_list else 0.0

    # Generation metrics
    ans_eval = [
        r for r in answerable
        if r.get("evaluation") and r["evaluation"].get("correctness") is not None
    ]
    gen_metrics = {}
    if ans_eval:
        c_scores = [r["evaluation"]["correctness"] for r in ans_eval]
        f_scores = [r["evaluation"]["faithfulness"] for r in ans_eval if r["evaluation"].get("faithfulness") is not None]
        kw_scores = [r["evaluation"]["keyword_overlap"] for r in ans_eval if r["evaluation"].get("keyword_overlap") is not None]
        n = len(c_scores)

        gen_metrics = {
            "mean_correctness": round(statistics.mean(c_scores), 4),
            "fully_correct_pct": round(sum(1 for s in c_scores if s == 2) / n * 100, 1),
            "partially_correct_pct": round(sum(1 for s in c_scores if s == 1) / n * 100, 1),
            "incorrect_pct": round(sum(1 for s in c_scores if s == 0) / n * 100, 1),
            "mean_faithfulness": round(statistics.mean(f_scores), 4) if f_scores else 0.0,
            "fully_grounded_pct": round(sum(1 for s in f_scores if s == 2) / len(f_scores) * 100, 1) if f_scores else 0.0,
            "mean_keyword_overlap": round(statistics.mean(kw_scores), 4) if kw_scores else 0.0,
        }

    # Citation & Abstention
    correct_cits = 0
    complete_cits = 0
    for r in answerable:
        gt_papers = set(r["ground_truth_paper_ids"])
        ret_papers = set(c["paper_id"] for c in r["retrieved"][:TOP_K])
        if gt_papers & ret_papers:
            correct_cits += 1
        if gt_papers <= ret_papers:
            complete_cits += 1

    unans_correct = sum(1 for r in unanswerable if r.get("evaluation", {}).get("correct_abstention"))
    unans_total = len(unanswerable)

    cit_metrics = {
        "citation_correctness_rate": round(correct_cits / len(answerable), 4) if answerable else 0.0,
        "citation_completeness_rate": round(complete_cits / len(answerable), 4) if answerable else 0.0,
        "abstention_accuracy": round(unans_correct / unans_total, 4) if unans_total else 0.0,
        "hallucination_rate": round((unans_total - unans_correct) / unans_total, 4) if unans_total else 0.0,
    }

    # Latency metrics
    lats = [r["latency"]["total_s"] for r in results if "latency" in r]
    gen_lats = [r["latency"]["generation_s"] for r in results if "latency" in r]
    lat_metrics = {}
    if lats:
        s_lat = sorted(lats)
        s_gen = sorted(gen_lats)
        n = len(s_lat)
        lat_metrics = {
            "mean_latency_s": round(statistics.mean(lats), 2),
            "p50_latency_s": round(s_lat[n // 2], 2),
            "p95_latency_s": round(s_lat[int(n * 0.95)], 2),
            "mean_generation_s": round(statistics.mean(gen_lats), 2),
            "p50_generation_s": round(s_gen[n // 2], 2),
            "p95_generation_s": round(s_gen[int(n * 0.95)], 2),
        }

    return {**ret_metrics, **gen_metrics, **cit_metrics, **lat_metrics}


def compute_category_breakdown(results: list[dict]) -> dict:
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)
    breakdown = {}
    for qtype, items in sorted(by_type.items()):
        breakdown[qtype] = {
            "count": len(items),
            **compute_metrics(items),
        }
    return breakdown


def run_experiment_suite(limit: int = None, models_filter: list[str] = None):
    log("Loading benchmark questions from " + str(QUESTIONS_PATH))
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]

    log(f"Loaded {len(questions)} evaluation questions.")

    # Initialize retrieval stack (Frozen V2.0)
    log("Initializing V2.0 Retrieval Stack: VectorStore, BM25Index, HybridRetriever, Reranker...")
    vector_store = VectorStore()
    bm25_index = BM25Index()
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25_index)
    reranker = Reranker()

    # Step 1: Cache V2.0 retrieval across all 40 questions to guarantee identical context
    log("Step 1: Running and caching V2.0 retrieval (Dense + BM25 -> RRF -> Cross-Encoder)...")
    retrieval_cache = {}
    for q in questions:
        qid = q["id"]
        query = q["question"]
        t_ret_0 = time.perf_counter()
        candidates = hybrid_retriever.search(query, limit=CANDIDATE_POOL, candidate_pool=CANDIDATE_POOL)
        reranked = reranker.rerank(query, candidates, top_k=EVAL_TOP_K)
        ret_time = time.perf_counter() - t_ret_0
        retrieval_cache[qid] = {
            "retrieved_full": reranked,
            "retrieval_time_s": ret_time,
        }
    log("V2.0 Retrieval cached for all 40 questions.")

    experiments = [
        {
            "id": "v2_0_reproduced",
            "name": "V2.0 Baseline (Reproduced)",
            "model": "qwen2.5:3b",
            "prompt_version": "v2_0",
            "system_prompt": V2_0_SYSTEM_PROMPT,
            "prompt_builder": build_v2_0_prompt,
        },
        {
            "id": "v2_1_3b_improved_prompt",
            "name": "Ablation A: 3B + Improved Prompt",
            "model": "qwen2.5:3b",
            "prompt_version": "v2_1_improved",
            "system_prompt": V2_1_SYSTEM_PROMPT,
            "prompt_builder": build_v2_1_prompt,
        },
        {
            "id": "v2_1_7b_v2_0_prompt",
            "name": "Ablation B: 7B + V2.0 Prompt",
            "model": "qwen2.5:7b",
            "prompt_version": "v2_0",
            "system_prompt": V2_0_SYSTEM_PROMPT,
            "prompt_builder": build_v2_0_prompt,
        },
        {
            "id": "v2_1_7b_improved_prompt",
            "name": "V2.1 Full System: 7B + Improved Prompt",
            "model": "qwen2.5:7b",
            "prompt_version": "v2_1_improved",
            "system_prompt": V2_1_SYSTEM_PROMPT,
            "prompt_builder": build_v2_1_prompt,
        },
    ]
    if models_filter:
        experiments = [e for e in experiments if e["id"] in models_filter or e["model"] in models_filter]

    all_experiment_outputs = {}
    if V2_1_RESULTS_PATH.exists():
        try:
            with open(V2_1_RESULTS_PATH, "r", encoding="utf-8") as f:
                all_experiment_outputs = json.load(f)
            log(f"Found {len(all_experiment_outputs)} existing experiments in results file.")
        except Exception:
            all_experiment_outputs = {}

    def save_outputs():
        with open(V2_1_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_experiment_outputs, f, indent=2, ensure_ascii=False)
        with open(V2_1_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "experiment_id", "experiment_name", "model", "prompt_version",
                "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
                "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                "mrr", "mean_correctness", "fully_correct_pct", "partially_correct_pct", "incorrect_pct",
                "mean_faithfulness", "fully_grounded_pct", "mean_keyword_overlap",
                "citation_correctness", "citation_completeness", "abstention_accuracy", "hallucination_rate",
                "mean_generation_s", "mean_latency_s", "p50_latency_s", "p95_latency_s"
            ]
            writer.writerow(header)
            for e_id, e_data in all_experiment_outputs.items():
                m = e_data["metrics"]
                writer.writerow([
                    e_id,
                    e_data["name"],
                    e_data["model"],
                    e_data["prompt_version"],
                    m.get("hit_at_1", ""), m.get("hit_at_3", ""), m.get("hit_at_5", ""), m.get("hit_at_10", ""),
                    m.get("recall_at_1", ""), m.get("recall_at_3", ""), m.get("recall_at_5", ""), m.get("recall_at_10", ""),
                    m.get("mrr", ""),
                    m.get("mean_correctness", ""),
                    m.get("fully_correct_pct", ""),
                    m.get("partially_correct_pct", ""),
                    m.get("incorrect_pct", ""),
                    m.get("mean_faithfulness", ""),
                    m.get("fully_grounded_pct", ""),
                    m.get("mean_keyword_overlap", ""),
                    m.get("citation_correctness_rate", ""),
                    m.get("citation_completeness_rate", ""),
                    m.get("abstention_accuracy", ""),
                    m.get("hallucination_rate", ""),
                    m.get("mean_generation_s", ""),
                    m.get("mean_latency_s", ""),
                    m.get("p50_latency_s", ""),
                    m.get("p95_latency_s", ""),
                ])

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        model = exp["model"]
        sys_prompt = exp["system_prompt"]
        prompt_builder = exp["prompt_builder"]

        log(f"\n{'='*60}\nRunning Experiment: {exp_name} (Model: {model})\n{'='*60}")
        exp_records = []

        for idx, q in enumerate(questions, 1):
            qid = q["id"]
            query = q["question"]
            cached = retrieval_cache[qid]
            ret_chunks = cached["retrieved_full"]
            ret_time = cached["retrieval_time_s"]

            top_chunks = ret_chunks[:TOP_K]
            prompt, citations = prompt_builder(top_chunks, query)

            t_gen_0 = time.perf_counter()
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                answer_text = resp["message"]["content"]
            except Exception as e:
                answer_text = f"[ERROR] {e}"
            gen_time = time.perf_counter() - t_gen_0

            # Evaluate answer
            if not q["answerable"]:
                ab_ok, ab_reason = check_abstention(answer_text, query)
                eval_scores = {
                    "correctness": None,
                    "faithfulness": None,
                    "keyword_overlap": None,
                    "correct_abstention": ab_ok,
                    "judge_reasoning": ab_reason,
                }
            else:
                kw = keyword_overlap(q.get("expected_answer", ""), answer_text)
                cor, fai, rea = judge_answerable(query, q.get("expected_answer", ""), answer_text, top_chunks)
                eval_scores = {
                    "correctness": cor,
                    "faithfulness": fai,
                    "keyword_overlap": kw,
                    "correct_abstention": None,
                    "judge_reasoning": rea,
                }

            record = {
                "question_id": qid,
                "question": query,
                "question_type": q["question_type"],
                "difficulty": q["difficulty"],
                "answerable": q["answerable"],
                "ground_truth_chunk_ids": q.get("relevant_chunk_ids", []),
                "ground_truth_paper_ids": q.get("paper_ids", []),
                "expected_answer": q.get("expected_answer"),
                "retrieved": [
                    {
                        "rank": r_idx,
                        "chunk_id": c.get("chunk_id"),
                        "score": c.get("final_score", c.get("reranker_score")),
                        "paper_id": c.get("paper_id"),
                        "section": c.get("section"),
                        "chunk_type": c.get("chunk_type"),
                        "image_id": c.get("image_id"),
                    }
                    for r_idx, c in enumerate(ret_chunks, 1)
                ],
                "generated_answer": answer_text,
                "citations": citations,
                "latency": {
                    "retrieval_s": round(ret_time, 3),
                    "generation_s": round(gen_time, 3),
                    "total_s": round(ret_time + gen_time, 3),
                },
                "evaluation": eval_scores,
            }
            exp_records.append(record)

            status = f"[{idx:02d}/40] {qid}: "
            if q["answerable"]:
                status += f"Cor={eval_scores['correctness']}/2, Fai={eval_scores['faithfulness']}/2, KW={eval_scores['keyword_overlap']:.2f}"
            else:
                status += f"Abstention={'PASS' if eval_scores['correct_abstention'] else 'FAIL'}"
            status += f" (Gen: {gen_time:.2f}s)"
            log(status)

        metrics = compute_metrics(exp_records)
        cat_breakdown = compute_category_breakdown(exp_records)

        log(f"\n--- {exp_name} Summary ---")
        log(f"Hit@5: {metrics['hit_at_5']*100:.1f}% | Recall@5: {metrics['recall_at_5']:.4f} | MRR: {metrics['mrr']:.4f}")
        log(f"Correctness: {metrics['mean_correctness']:.2f}/2.0 (Full: {metrics['fully_correct_pct']}%, Part: {metrics['partially_correct_pct']}%, Inc: {metrics['incorrect_pct']}%)")
        log(f"Faithfulness: {metrics['mean_faithfulness']:.2f}/2.0 (Full Grounded: {metrics['fully_grounded_pct']}%)")
        log(f"Abstention: {metrics['abstention_accuracy']*100:.1f}% | Latency: Gen={metrics['mean_generation_s']}s, Total={metrics['mean_latency_s']}s")

        all_experiment_outputs[exp_id] = {
            "name": exp_name,
            "model": model,
            "prompt_version": exp["prompt_version"],
            "metrics": metrics,
            "category_breakdown": cat_breakdown,
            "records": exp_records,
        }
        save_outputs()
        log(f"Incrementally updated results in {V2_1_RESULTS_PATH} and {V2_1_SUMMARY_CSV}")

    return all_experiment_outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V2.1 Experiment Benchmark Suite")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--experiments", nargs="*", default=None, help="Filter specific experiment IDs or model names")
    args = parser.parse_args()

    run_experiment_suite(limit=args.limit, models_filter=args.experiments)
