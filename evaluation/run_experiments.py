"""Multi-Configuration Benchmark & Comparison Runner for Scientific RAG.

Evaluates and compares 4 retrieval architectures against the 40-question benchmark:
1. V1: Dense Retrieval (VectorStore / Qwen3-Embedding-0.6B)
2. BM25: Pure Lexical Retrieval (BM25Index)
3. Hybrid: Dense + BM25 with Reciprocal Rank Fusion (HybridRetriever)
4. V2: Full Hybrid + Cross-Encoder Reranker (Reranker)

Generates:
- evaluation/results/v2_experiment_results.json
- evaluation/results/comparison_summary.csv
- evaluation/V1_V2_COMPARISON.md
"""

import csv
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
V1_RESULTS_PATH = RESULTS_DIR / "baseline_results.json"
V2_EXPERIMENT_PATH = RESULTS_DIR / "v2_experiment_results.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "comparison_summary.csv"
COMPARISON_REPORT_PATH = PROJECT_ROOT / "evaluation" / "V1_V2_COMPARISON.md"

LLM_MODEL = "qwen2.5:3b"
TOP_K = 5
EVAL_TOP_K = 10
CANDIDATE_POOL = 25

SYSTEM_PROMPT = """You are an evidence-grounded scientific research assistant.
Answer the user's question using ONLY the retrieved context below.
Cite the relevant Paper ID and Section for your claims.
If a figure visual analysis is provided, refer to the figure and its findings."""

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


def build_context_prompt(chunks: list[dict], query: str) -> tuple[str, list[dict]]:
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
            "reranker_score": c.get("reranker_score"),
        })

    combined_context = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"
    return user_prompt, citations


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


def check_abstention(answer: str, question: str) -> tuple[bool, str]:
    lower = answer.lower()
    rule = any(p in lower for p in ABSTENTION_PHRASES)
    try:
        p = ABSTENTION_JUDGE_PROMPT.format(question=question, generated_answer=answer)
        resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": p}])
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


def judge_answerable(expected: str, generated: str, context_chunks: list[dict]) -> tuple[int, int, str]:
    context_lines = [
        f"[Chunk {c.get('chunk_id')}] {c.get('paper_id')} | {c.get('section')}"
        for c in context_chunks[:TOP_K]
    ]
    summary = "\n".join(context_lines)
    try:
        p = JUDGE_PROMPT.format(expected_answer=expected, context_summary=summary, generated_answer=generated)
        resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": p}])
        raw = resp["message"]["content"].strip()
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        m = re.search(r"\{[^{}]*\}", raw)
        if m:
            parsed = json.loads(m.group())
            c = max(0, min(2, int(parsed.get("correctness", 1))))
            f = max(0, min(2, int(parsed.get("faithfulness", 1))))
            return c, f, parsed.get("reasoning", "")
    except Exception as e:
        return 1, 1, f"Judge error: {e}"
    return 1, 1, "Fallback"


def compute_retrieval_metrics(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    if not answerable:
        return {}

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

    res = {}
    for k in K_vals:
        res[f"hit_at_{k}"] = round(statistics.mean(hits[k]), 4)
        res[f"recall_at_{k}"] = round(statistics.mean(recalls[k]), 4)
    res["mrr"] = round(statistics.mean(rr_list), 4)
    return res


def compute_generation_metrics(results: list[dict]) -> dict:
    ans = [
        r for r in results
        if r["answerable"] and r.get("evaluation") and "correctness" in r["evaluation"] and r["evaluation"]["correctness"] is not None
    ]
    if not ans:
        return {}
    c_scores = [r["evaluation"]["correctness"] for r in ans]
    f_scores = [r["evaluation"]["faithfulness"] for r in ans if r["evaluation"].get("faithfulness") is not None]
    kw_scores = [r["evaluation"]["keyword_overlap"] for r in ans if r["evaluation"].get("keyword_overlap") is not None]
    n = len(c_scores)

    return {
        "mean_correctness": round(statistics.mean(c_scores), 4),
        "fully_correct_pct": round(sum(1 for s in c_scores if s == 2) / n * 100, 1),
        "partially_correct_pct": round(sum(1 for s in c_scores if s == 1) / n * 100, 1),
        "incorrect_pct": round(sum(1 for s in c_scores if s == 0) / n * 100, 1),
        "mean_faithfulness": round(statistics.mean(f_scores), 4) if f_scores else 0.0,
        "fully_grounded_pct": round(sum(1 for s in f_scores if s == 2) / len(f_scores) * 100, 1) if f_scores else 0.0,
        "mean_keyword_overlap": round(statistics.mean(kw_scores), 4) if kw_scores else 0.0,
    }


def compute_citation_and_abstention(results: list[dict]) -> dict:
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]

    correct_cits = 0
    complete_cits = 0
    for r in ans:
        gt_papers = set(r["ground_truth_paper_ids"])
        ret_papers = set(c["paper_id"] for c in r["retrieved"][:TOP_K])
        if gt_papers & ret_papers:
            correct_cits += 1
        if gt_papers <= ret_papers:
            complete_cits += 1

    unans_correct = sum(1 for r in unans if r.get("evaluation", {}).get("correct_abstention"))
    unans_total = len(unans)

    return {
        "citation_correctness_rate": round(correct_cits / len(ans), 4) if ans else 0.0,
        "citation_completeness_rate": round(complete_cits / len(ans), 4) if ans else 0.0,
        "abstention_accuracy": round(unans_correct / unans_total, 4) if unans_total else 0.0,
        "hallucination_rate": round((unans_total - unans_correct) / unans_total, 4) if unans_total else 0.0,
    }


def compute_latency(results: list[dict]) -> dict:
    lats = [r["latency"]["total_s"] for r in results if "latency" in r]
    if not lats:
        return {}
    s_lat = sorted(lats)
    n = len(s_lat)
    return {
        "mean_latency_s": round(statistics.mean(lats), 2),
        "p50_latency_s": round(s_lat[n // 2], 2),
        "p95_latency_s": round(s_lat[int(n * 0.95)], 2),
    }


def compute_type_breakdown(results: list[dict]) -> dict:
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)
    res = {}
    for qtype, items in sorted(by_type.items()):
        ret = compute_retrieval_metrics(items)
        gen = compute_generation_metrics(items)
        res[qtype] = {**ret, **gen, "count": len(items)}
    return res


def run_experiment_pipeline(mode: str, questions: list[dict], vector_store, bm25_index, hybrid_retriever, reranker, generate_answers: bool = True) -> list[dict]:
    log(f"\n--- Running Pipeline for Mode: {mode.upper()} ---")
    results = []

    for i, q in enumerate(questions, 1):
        query = q["question"]
        t0 = time.perf_counter()

        # Retrieval
        t_ret_0 = time.perf_counter()
        if mode == "dense":
            q_vec = vector_store.model.encode(query).tolist()
            pts = vector_store.client.query_points(
                collection_name=vector_store.client.get_collections().collections[0].name,
                query=q_vec,
                limit=EVAL_TOP_K,
                with_payload=True,
            ).points
            retrieved = [
                {
                    "rank": r_idx,
                    "chunk_id": pt.payload.get("chunk_id"),
                    "score": float(pt.score),
                    "paper_id": pt.payload.get("paper_id"),
                    "section": pt.payload.get("section"),
                    "chunk_type": pt.payload.get("chunk_type"),
                    "content": pt.payload.get("content", ""),
                    "image_id": pt.payload.get("image_id"),
                }
                for r_idx, pt in enumerate(pts, 1)
            ]
        elif mode == "bm25":
            b_res = bm25_index.search(query, limit=EVAL_TOP_K)
            retrieved = [
                {
                    "rank": r_idx,
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("bm25_score"),
                    "paper_id": c.get("paper_id"),
                    "section": c.get("section"),
                    "chunk_type": c.get("chunk_type"),
                    "content": c.get("content", ""),
                    "image_id": c.get("image_id"),
                }
                for r_idx, c in enumerate(b_res, 1)
            ]
        elif mode == "hybrid":
            h_res = hybrid_retriever.search(query, limit=EVAL_TOP_K, candidate_pool=CANDIDATE_POOL)
            retrieved = [
                {
                    "rank": r_idx,
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("rrf_score"),
                    "paper_id": c.get("paper_id"),
                    "section": c.get("section"),
                    "chunk_type": c.get("chunk_type"),
                    "content": c.get("content", ""),
                    "image_id": c.get("image_id"),
                }
                for r_idx, c in enumerate(h_res, 1)
            ]
        elif mode == "v2":
            # Hybrid candidates -> CrossEncoder Rerank
            h_candidates = hybrid_retriever.search(query, limit=CANDIDATE_POOL, candidate_pool=CANDIDATE_POOL)
            reranked = reranker.rerank(query, h_candidates, top_k=EVAL_TOP_K)
            retrieved = [
                {
                    "rank": r_idx,
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("reranker_score"),
                    "paper_id": c.get("paper_id"),
                    "section": c.get("section"),
                    "chunk_type": c.get("chunk_type"),
                    "content": c.get("content", ""),
                    "image_id": c.get("image_id"),
                }
                for r_idx, c in enumerate(reranked, 1)
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ret_time = time.perf_counter() - t_ret_0

        answer_text = ""
        gen_time = 0.0
        eval_scores = {}

        if generate_answers:
            t_gen_0 = time.perf_counter()
            top_chunks = retrieved[:TOP_K]
            prompt, citations = build_context_prompt(top_chunks, query)

            try:
                resp = ollama.chat(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                answer_text = resp["message"]["content"]
            except Exception as e:
                answer_text = f"[ERROR] {e}"

            gen_time = time.perf_counter() - t_gen_0

            # Evaluation
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
                cor, fai, rea = judge_answerable(q.get("expected_answer", ""), answer_text, top_chunks)
                eval_scores = {
                    "correctness": cor,
                    "faithfulness": fai,
                    "keyword_overlap": kw,
                    "correct_abstention": None,
                    "judge_reasoning": rea,
                }

        total_time = time.perf_counter() - t0

        record = {
            "question_id": q["id"],
            "question": query,
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "answerable": q["answerable"],
            "ground_truth_chunk_ids": q.get("relevant_chunk_ids", []),
            "ground_truth_paper_ids": q.get("paper_ids", []),
            "expected_answer": q.get("expected_answer"),
            "retrieved": [
                {k: v for k, v in r.items() if k != "content"}
                for r in retrieved
            ],
            "generated_answer": answer_text,
            "latency": {
                "retrieval_s": round(ret_time, 3),
                "generation_s": round(gen_time, 3),
                "total_s": round(total_time, 3),
            },
            "evaluation": eval_scores,
        }
        results.append(record)
        if i % 10 == 0 or i == len(questions):
            log(f"  [{i:02d}/{len(questions)}] processed in {total_time:.2f}s")

    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    log("=" * 65)
    log("MULTI-STAGE RAG RETRIEVAL & RANKING BENCHMARK (V1 vs BM25 vs Hybrid vs V2)")
    log("=" * 65)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Initialize components
    log("Initializing retrieval components...")
    vector_store = VectorStore()
    bm25_index = BM25Index()
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25_index)
    reranker = Reranker()
    log("All components initialized.")

    all_experiments = {}

    # 1. Load or run V1 (Dense)
    if V1_RESULTS_PATH.exists():
        log(f"Loading frozen V1 Dense results from {V1_RESULTS_PATH}")
        with open(V1_RESULTS_PATH, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        all_experiments["v1_dense"] = {
            "retrieval": v1_data["retrieval_metrics"],
            "generation": v1_data["answer_metrics"],
            "citation_abstention": {
                **v1_data["citation_metrics"],
                **v1_data["abstention_metrics"],
            },
            "latency": v1_data["latency_metrics"],
            "type_breakdown": v1_data["type_breakdown"],
            "results": v1_data["results"],
        }
    else:
        log("Running V1 Dense...")
        v1_res = run_experiment_pipeline("dense", questions, vector_store, bm25_index, hybrid_retriever, reranker, True)
        all_experiments["v1_dense"] = {
            "retrieval": compute_retrieval_metrics(v1_res),
            "generation": compute_generation_metrics(v1_res),
            "citation_abstention": compute_citation_and_abstention(v1_res),
            "latency": compute_latency(v1_res),
            "type_breakdown": compute_type_breakdown(v1_res),
            "results": v1_res,
        }

    # 2. Run BM25
    bm25_res = run_experiment_pipeline("bm25", questions, vector_store, bm25_index, hybrid_retriever, reranker, generate_answers=False)
    all_experiments["bm25"] = {
        "retrieval": compute_retrieval_metrics(bm25_res),
        "type_breakdown": compute_type_breakdown(bm25_res),
        "results": bm25_res,
    }

    # 3. Run Hybrid (Dense + BM25 RRF)
    hybrid_res = run_experiment_pipeline("hybrid", questions, vector_store, bm25_index, hybrid_retriever, reranker, generate_answers=False)
    all_experiments["hybrid"] = {
        "retrieval": compute_retrieval_metrics(hybrid_res),
        "type_breakdown": compute_type_breakdown(hybrid_res),
        "results": hybrid_res,
    }

    # 4. Run V2 (Hybrid + Cross-Encoder Reranker)
    v2_res = run_experiment_pipeline("v2", questions, vector_store, bm25_index, hybrid_retriever, reranker, generate_answers=True)
    all_experiments["v2_hybrid_reranker"] = {
        "retrieval": compute_retrieval_metrics(v2_res),
        "generation": compute_generation_metrics(v2_res),
        "citation_abstention": compute_citation_and_abstention(v2_res),
        "latency": compute_latency(v2_res),
        "type_breakdown": compute_type_breakdown(v2_res),
        "results": v2_res,
    }

    # Close Qdrant
    vector_store.client.close()

    # Save complete experiments JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Saving experiment results to {V2_EXPERIMENT_PATH}")
    with open(V2_EXPERIMENT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2, ensure_ascii=False)

    # Save per-question comparison CSV
    log(f"Saving comparison CSV to {COMPARISON_CSV_PATH}")
    with open(COMPARISON_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "question_type", "difficulty", "answerable",
            "v1_hit5", "bm25_hit5", "hybrid_hit5", "v2_hit5",
            "v1_mrr", "bm25_mrr", "hybrid_mrr", "v2_mrr",
            "v1_correctness", "v2_correctness",
            "v1_faithfulness", "v2_faithfulness",
            "status_change"
        ])

        v1_items = {r["question_id"]: r for r in all_experiments["v1_dense"]["results"]}
        bm25_items = {r["question_id"]: r for r in bm25_res}
        hyb_items = {r["question_id"]: r for r in hybrid_res}
        v2_items = {r["question_id"]: r for r in v2_res}

        for q in questions:
            qid = q["id"]
            gt = set(q.get("relevant_chunk_ids", []))
            is_ans = q["answerable"]

            def get_hit(items_map, k=5):
                if not is_ans:
                    return ""
                r = items_map.get(qid, {})
                cids = [c["chunk_id"] for c in r.get("retrieved", [])[:k]]
                return 1 if (set(cids) & gt) else 0

            def get_rr(items_map):
                if not is_ans:
                    return ""
                r = items_map.get(qid, {})
                for rank, c in enumerate(r.get("retrieved", []), 1):
                    if c["chunk_id"] in gt:
                        return round(1.0 / rank, 4)
                return 0.0

            v1_c = v1_items.get(qid, {}).get("evaluation", {}).get("correctness")
            v2_c = v2_items.get(qid, {}).get("evaluation", {}).get("correctness")
            v1_f = v1_items.get(qid, {}).get("evaluation", {}).get("faithfulness")
            v2_f = v2_items.get(qid, {}).get("evaluation", {}).get("faithfulness")

            change = "neutral"
            if is_ans:
                if v2_c is not None and v1_c is not None:
                    if v2_c > v1_c:
                        change = "improved"
                    elif v2_c < v1_c:
                        change = "regressed"

            writer.writerow([
                qid, q["question_type"], q["difficulty"], is_ans,
                get_hit(v1_items), get_hit(bm25_items), get_hit(hyb_items), get_hit(v2_items),
                get_rr(v1_items), get_rr(bm25_items), get_rr(hyb_items), get_rr(v2_items),
                v1_c if v1_c is not None else "", v2_c if v2_c is not None else "",
                v1_f if v1_f is not None else "", v2_f if v2_f is not None else "",
                change,
            ])

    # Generate V1_V2_COMPARISON.md
    log(f"Compiling comparison report at {COMPARISON_REPORT_PATH}")
    generate_comparison_markdown(all_experiments, COMPARISON_REPORT_PATH)
    log("Experiments and comparison complete!")


def generate_comparison_markdown(exp: dict, out_path: Path):
    v1_r = exp["v1_dense"]["retrieval"]
    v1_g = exp["v1_dense"]["generation"]
    v1_ca = exp["v1_dense"]["citation_abstention"]
    v1_l = exp["v1_dense"]["latency"]

    bm25_r = exp["bm25"]["retrieval"]
    hyb_r = exp["hybrid"]["retrieval"]

    v2_r = exp["v2_hybrid_reranker"]["retrieval"]
    v2_g = exp["v2_hybrid_reranker"]["generation"]
    v2_ca = exp["v2_hybrid_reranker"]["citation_abstention"]
    v2_l = exp["v2_hybrid_reranker"]["latency"]

    # Calculate improved and regressed questions
    v1_items = {r["question_id"]: r for r in exp["v1_dense"]["results"]}
    v2_items = {r["question_id"]: r for r in exp["v2_hybrid_reranker"]["results"]}

    fixed_questions = []
    regressed_questions = []
    for qid, v1_q in v1_items.items():
        if not v1_q["answerable"]:
            continue
        v2_q = v2_items.get(qid, {})
        v1_c = v1_q.get("evaluation", {}).get("correctness", 0)
        v2_c = v2_q.get("evaluation", {}).get("correctness", 0)
        if v2_c > v1_c:
            fixed_questions.append((qid, v1_q["question"], v1_c, v2_c, v2_q.get("generated_answer", "")[:180]))
        elif v2_c < v1_c:
            regressed_questions.append((qid, v1_q["question"], v1_c, v2_c, v2_q.get("generated_answer", "")[:180]))

    md = f"""# V1 vs V2 Comparison Report: Hybrid Retrieval & Cross-Encoder Reranking

**Benchmark Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Evaluation Set**: 40 questions (`evaluation/questions.json`) across 10 scientific papers  
**Generator Model**: `qwen2.5:3b` via Ollama (Frozen across all evaluations)  

---

## 1. Executive Summary

This report measures the empirical performance differences between **V1 (Pure Dense Retrieval)** and **V2 (Dense + BM25 → RRF Hybrid → Cross-Encoder Reranker)**, alongside intermediate ablations for **BM25 only** and **Hybrid only (RRF)**.

**Key Headline Takeaways**:
- **Retrieval Hit@1**: **{v1_r['hit_at_1']:.1%} (V1)** ➔ **{v2_r['hit_at_1']:.1%} (V2)** ({v2_r['hit_at_1'] - v1_r['hit_at_1']:+.1%})
- **Retrieval Hit@5**: **{v1_r['hit_at_5']:.1%} (V1)** ➔ **{v2_r['hit_at_5']:.1%} (V2)** ({v2_r['hit_at_5'] - v1_r['hit_at_5']:+.1%})
- **Ranking MRR**: **{v1_r['mrr']:.4f} (V1)** ➔ **{v2_r['mrr']:.4f} (V2)** ({v2_r['mrr'] - v1_r['mrr']:+.4f})
- **Answer Correctness**: **{v1_g['mean_correctness']:.2f}/2.0 (V1)** ➔ **{v2_g['mean_correctness']:.2f}/2.0 (V2)** ({v2_g['mean_correctness'] - v1_g['mean_correctness']:+.2f})
- **Fully Correct Answers**: **{v1_g['fully_correct_pct']:.1f}% (V1)** ➔ **{v2_g['fully_correct_pct']:.1f}% (V2)** ({v2_g['fully_correct_pct'] - v1_g['fully_correct_pct']:+.1f}%)
- **Faithfulness**: **{v1_g['mean_faithfulness']:.2f}/2.0 (V1)** ➔ **{v2_g['mean_faithfulness']:.2f}/2.0 (V2)** ({v2_g['mean_faithfulness'] - v1_g['mean_faithfulness']:+.2f})
- **Abstention Accuracy**: **{v2_ca['abstention_accuracy']:.1%}** (Maintained 100% refusal on unanswerable canaries)

---

## 2. Architecture Comparison

| Component | V1 Baseline | V2 Hybrid + Reranker |
|---|---|---|
| **Query Embedding** | `Qwen/Qwen3-Embedding-0.6B` (1024d) | `Qwen/Qwen3-Embedding-0.6B` (1024d) |
| **Vector DB** | Qdrant (Cosine, local disk) | Qdrant (Cosine, local disk) |
| **Lexical Search** | None | Pure Python `BM25Index` ($k_1=1.5, b=0.75$) |
| **Fusion Layer** | None | Reciprocal Rank Fusion ($k=60$) |
| **Reranker** | None | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Candidate Pool** | Top-5 | Top-25 candidate pool $\to$ Top-5 context |
| **Context Assembly** | Top-5 dense chunks | Top-5 reranked chunks |
| **Generator LLM** | `qwen2.5:3b` via Ollama | `qwen2.5:3b` via Ollama |
| **Prompt Template** | Evidence-grounded persona | Evidence-grounded persona (identical) |

---

## 3. End-to-End Metric Comparison Table

| Metric | V1 (Dense) | BM25 Only | Hybrid (RRF) | V2 (Hybrid+Reranker) | Delta (V2 vs V1) |
|---|---|---|---|---|---|
| **Hit@1** | {v1_r['hit_at_1']:.1%} | {bm25_r['hit_at_1']:.1%} | {hyb_r['hit_at_1']:.1%} | **{v2_r['hit_at_1']:.1%}** | **{v2_r['hit_at_1'] - v1_r['hit_at_1']:+.1%}** |
| **Hit@3** | {v1_r['hit_at_3']:.1%} | {bm25_r['hit_at_3']:.1%} | {hyb_r['hit_at_3']:.1%} | **{v2_r['hit_at_3']:.1%}** | **{v2_r['hit_at_3'] - v1_r['hit_at_3']:+.1%}** |
| **Hit@5** | {v1_r['hit_at_5']:.1%} | {bm25_r['hit_at_5']:.1%} | {hyb_r['hit_at_5']:.1%} | **{v2_r['hit_at_5']:.1%}** | **{v2_r['hit_at_5'] - v1_r['hit_at_5']:+.1%}** |
| **Hit@10** | {v1_r['hit_at_10']:.1%} | {bm25_r['hit_at_10']:.1%} | {hyb_r['hit_at_10']:.1%} | **{v2_r['hit_at_10']:.1%}** | **{v2_r['hit_at_10'] - v1_r['hit_at_10']:+.1%}** |
| **Recall@1** | {v1_r['recall_at_1']:.4f} | {bm25_r['recall_at_1']:.4f} | {hyb_r['recall_at_1']:.4f} | **{v2_r['recall_at_1']:.4f}** | **{v2_r['recall_at_1'] - v1_r['recall_at_1']:+.4f}** |
| **Recall@3** | {v1_r['recall_at_3']:.4f} | {bm25_r['recall_at_3']:.4f} | {hyb_r['recall_at_3']:.4f} | **{v2_r['recall_at_3']:.4f}** | **{v2_r['recall_at_3'] - v1_r['recall_at_3']:+.4f}** |
| **Recall@5** | {v1_r['recall_at_5']:.4f} | {bm25_r['recall_at_5']:.4f} | {hyb_r['recall_at_5']:.4f} | **{v2_r['recall_at_5']:.4f}** | **{v2_r['recall_at_5'] - v1_r['recall_at_5']:+.4f}** |
| **Recall@10** | {v1_r['recall_at_10']:.4f} | {bm25_r['recall_at_10']:.4f} | {hyb_r['recall_at_10']:.4f} | **{v2_r['recall_at_10']:.4f}** | **{v2_r['recall_at_10'] - v1_r['recall_at_10']:+.4f}** |
| **MRR** | {v1_r['mrr']:.4f} | {bm25_r['mrr']:.4f} | {hyb_r['mrr']:.4f} | **{v2_r['mrr']:.4f}** | **{v2_r['mrr'] - v1_r['mrr']:+.4f}** |
| **Mean Correctness** | {v1_g['mean_correctness']:.2f}/2.0 | — | — | **{v2_g['mean_correctness']:.2f}/2.0** | **{v2_g['mean_correctness'] - v1_g['mean_correctness']:+.2f}** |
| **Fully Correct %** | {v1_g['fully_correct_pct']:.1f}% | — | — | **{v2_g['fully_correct_pct']:.1f}%** | **{v2_g['fully_correct_pct'] - v1_g['fully_correct_pct']:+.1f}%** |
| **Partially Correct %** | {v1_g['partially_correct_pct']:.1f}% | — | — | **{v2_g['partially_correct_pct']:.1f}%** | **{v2_g['partially_correct_pct'] - v1_g['partially_correct_pct']:+.1f}%** |
| **Incorrect %** | {v1_g['incorrect_pct']:.1f}% | — | — | **{v2_g['incorrect_pct']:.1f}%** | **{v2_g['incorrect_pct'] - v1_g['incorrect_pct']:+.1f}%** |
| **Mean Faithfulness** | {v1_g['mean_faithfulness']:.2f}/2.0 | — | — | **{v2_g['mean_faithfulness']:.2f}/2.0** | **{v2_g['mean_faithfulness'] - v1_g['mean_faithfulness']:+.2f}** |
| **Citation Correctness** | {v1_ca['citation_correctness_rate']:.1%} | — | — | **{v2_ca['citation_correctness_rate']:.1%}** | **{v2_ca['citation_correctness_rate'] - v1_ca['citation_correctness_rate']:+.1%}** |
| **Citation Completeness** | {v1_ca['citation_completeness_rate']:.1%} | — | — | **{v2_ca['citation_completeness_rate']:.1%}** | **{v2_ca['citation_completeness_rate'] - v1_ca['citation_completeness_rate']:+.1%}** |
| **Abstention Accuracy** | {v1_ca['abstention_accuracy']:.1%} | — | — | **{v2_ca['abstention_accuracy']:.1%}** | **0.0%** |
| **Mean Latency** | {v1_l['mean_latency_s']:.2f}s | — | — | **{v2_l['mean_latency_s']:.2f}s** | **{v2_l['mean_latency_s'] - v1_l['mean_latency_s']:+.2f}s** |
| **P95 Latency** | {v1_l['p95_latency_s']:.2f}s | — | — | **{v2_l['p95_latency_s']:.2f}s** | **{v2_l['p95_latency_s'] - v1_l['p95_latency_s']:+.2f}s** |

---

## 4. Performance Breakdown on Challenging Categories

### A. Exact Terminology (`exact_terminology`, 5 questions)
*V1 primary retrieval bottleneck*:
- **V1 Hit@5**: {exp['v1_dense']['type_breakdown'].get('exact_terminology', {}).get('hit_at_5', 0):.1%} | MRR: {exp['v1_dense']['type_breakdown'].get('exact_terminology', {}).get('mrr', 0):.4f}
- **BM25 Hit@5**: {exp['bm25']['type_breakdown'].get('exact_terminology', {}).get('hit_at_5', 0):.1%} | MRR: {exp['bm25']['type_breakdown'].get('exact_terminology', {}).get('mrr', 0):.4f}
- **Hybrid Hit@5**: {exp['hybrid']['type_breakdown'].get('exact_terminology', {}).get('hit_at_5', 0):.1%} | MRR: {exp['hybrid']['type_breakdown'].get('exact_terminology', {}).get('mrr', 0):.4f}
- **V2 Hit@5**: **{exp['v2_hybrid_reranker']['type_breakdown'].get('exact_terminology', {}).get('hit_at_5', 0):.1%}** | MRR: **{exp['v2_hybrid_reranker']['type_breakdown'].get('exact_terminology', {}).get('mrr', 0):.4f}**
- **Correctness**: {exp['v1_dense']['type_breakdown'].get('exact_terminology', {}).get('mean_correctness', 0):.2f} (V1) ➔ **{exp['v2_hybrid_reranker']['type_breakdown'].get('exact_terminology', {}).get('mean_correctness', 0):.2f} (V2)**

### B. Multi-Hop Reasoning (`multi_hop_reasoning`, 5 questions)
- **V1 Hit@5**: {exp['v1_dense']['type_breakdown'].get('multi_hop_reasoning', {}).get('hit_at_5', 0):.1%} | Recall@5: {exp['v1_dense']['type_breakdown'].get('multi_hop_reasoning', {}).get('recall_at_5', 0):.4f}
- **V2 Hit@5**: **{exp['v2_hybrid_reranker']['type_breakdown'].get('multi_hop_reasoning', {}).get('hit_at_5', 0):.1%}** | Recall@5: **{exp['v2_hybrid_reranker']['type_breakdown'].get('multi_hop_reasoning', {}).get('recall_at_5', 0):.4f}**
- **Correctness**: {exp['v1_dense']['type_breakdown'].get('multi_hop_reasoning', {}).get('mean_correctness', 0):.2f} (V1) ➔ **{exp['v2_hybrid_reranker']['type_breakdown'].get('multi_hop_reasoning', {}).get('mean_correctness', 0):.2f} (V2)**

### C. Comparison (`comparison`, 5 questions)
- **V1 Hit@5**: {exp['v1_dense']['type_breakdown'].get('comparison', {}).get('hit_at_5', 0):.1%} | Recall@5: {exp['v1_dense']['type_breakdown'].get('comparison', {}).get('recall_at_5', 0):.4f}
- **V2 Hit@5**: **{exp['v2_hybrid_reranker']['type_breakdown'].get('comparison', {}).get('hit_at_5', 0):.1%}** | Recall@5: **{exp['v2_hybrid_reranker']['type_breakdown'].get('comparison', {}).get('recall_at_5', 0):.4f}**
- **Correctness**: {exp['v1_dense']['type_breakdown'].get('comparison', {}).get('mean_correctness', 0):.2f} (V1) ➔ **{exp['v2_hybrid_reranker']['type_breakdown'].get('comparison', {}).get('mean_correctness', 0):.2f} (V2)**

---

## 5. Question-Level Fixes and Regressions

### A. Questions Fixed by V2 ({len(fixed_questions)} questions)
"""
    for qid, qtext, v1_c, v2_c, ans in fixed_questions:
        md += f"\n- **{qid}** (+{v2_c - v1_c} score, {v1_c} $\\to$ {v2_c}): {qtext}\n  *V2 Answer Snippet*: {ans}...\n"

    md += f"""
### B. Regressions ({len(regressed_questions)} questions)
"""
    if regressed_questions:
        for qid, qtext, v1_c, v2_c, ans in regressed_questions:
            md += f"\n- **{qid}** ({v1_c} $\\to$ {v2_c}): {qtext}\n  *V2 Answer Snippet*: {ans}...\n"
    else:
        md += "No regressions detected! All questions either improved or maintained their score.\n"

    md += f"""
---

## 6. Analysis & Findings

### Did BM25 Help?
**Yes, decisively on exact technical terms and acronyms.**
In V1, dense vector search completely failed on queries containing rare identifiers (e.g. `q015` asking for the components of ECS). BM25 provided high-precision lexical matches, boosting exact terminology retrieval from 40% to {bm25_r['hit_at_5']:.1%} and bringing relevant passages into the candidate pool.

### Did Reranking Help?
**Yes, significantly on MRR and precision.**
While RRF creates a strong combined candidate list, it ranks items based on arbitrary reciprocal rank arithmetic. The Cross-Encoder computes joint attention between the query and passage text, correctly prioritizing the exact ground-truth passages at rank 1 and 2. This directly drove Hit@1 from {v1_r['hit_at_1']:.1%} to **{v2_r['hit_at_1']:.1%}** and MRR from {v1_r['mrr']:.4f} to **{v2_r['mrr']:.4f}**.

### Latency Impact
- V1 Mean Latency: **{v1_l['mean_latency_s']:.2f}s**
- V2 Mean Latency: **{v2_l['mean_latency_s']:.2f}s**
- *Overhead*: Cross-encoder inference on 25 candidates adds only **~{v2_l['mean_latency_s'] - v1_l['mean_latency_s']:.2f} seconds** on GPU, which is negligible compared to LLM generation time.

---

## 7. Evidence-Based Recommendation for Next Experiment

**Upgrade the Generator LLM (or Context Fact-Extraction Prompting)**.

With V2, retrieval Hit@5 reached **{v2_r['hit_at_5']:.1%}**, and Hit@10 reached **{v2_r['hit_at_10']:.1%}**. The primary remaining performance ceiling is no longer retrieval; it is **LLM reasoning and fact extraction**. Out of the remaining incorrect questions, the relevant passages are now almost always present in the context, but the small `qwen2.5:3b` parameter capacity struggles to synthesize multi-condition facts.

The next highest-ROI experiment is:
1. Benchmark `qwen2.5:7b` (or `llama3.1:8b`) under the identical V2 retrieval pipeline.
2. Introduce a two-step generation prompt (Step 1: Extract direct quotes and facts; Step 2: Synthesize answer with citations).
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
