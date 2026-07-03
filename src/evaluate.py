"""
Benchmark harness: golden set -> retrieve -> generate -> DeepEval scoring (Gemini judge).

Free-tier friendly: paces judge calls, retries on 429, runs one method per invocation
so you stay under the requests-per-minute cap.

Usage (run each method separately to respect the free-tier RPM cap):
    python src/evaluate.py --chunks data/processed/chunks_recursive.jsonl --collection banking_recursive --method dense --limit 3
    python src/evaluate.py ... --method hybrid --limit 3
    python src/evaluate.py ... --method hybrid_rerank --limit 3

Each run appends its scores to results/benchmark.json so you build the full table
across separate invocations.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from retrieve import retrieve, _close_client
from generate import generate_answer

EVAL_PATH = "data/eval/golden_set.jsonl"
RESULTS_DIR = Path("results")
JUDGE_MODEL = "gemini-2.5-flash-lite"   # highest free-tier RPM
CALL_GAP = 8.0                          # seconds between judge measures (free-tier safety)


def load_golden(limit=None):
    rows = [json.loads(l) for l in Path(EVAL_PATH).read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def base_cfg(collection, chunks_path, method):
    return {
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "vector_store": {"collection": collection},
        "retrieval": {"method": method, "top_k_dense": 20,
                      "top_k_bm25": 20, "final_k": 5, "reranker": None},
        "generation": {"model": "gemini-2.5-flash-lite"},
        "_chunks_path": chunks_path,
    }


def measure_with_retry(metric, tc, retries=4):
    for attempt in range(retries):
        try:
            metric.measure(tc)
            return metric.score
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 20 * (attempt + 1)
                print(f"    rate limited; waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    print("    giving up on this measure after retries")
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--method", required=True, choices=["dense", "hybrid", "hybrid_rerank"])
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    golden = load_golden(args.limit)
    cfg = base_cfg(args.collection, args.chunks, args.method)

    judge = GeminiModel(model=JUDGE_MODEL, api_key=os.environ["GEMINI_API_KEY"], temperature=0)
    metrics = [
        ("Faithfulness", FaithfulnessMetric(model=judge, async_mode=False)),
        ("Answer Relevancy", AnswerRelevancyMetric(model=judge, async_mode=False)),
        ("Ctx Precision", ContextualPrecisionMetric(model=judge, async_mode=False)),
        ("Ctx Recall", ContextualRecallMetric(model=judge, async_mode=False)),
    ]

    print(f"\n{'='*60}\nMETHOD: {args.method}  ({len(golden)} questions)\n{'='*60}")

    # build test cases (retrieval + generation) first
    cases = []
    for row in golden:
        hits = retrieve(row["query"], cfg)
        gen = generate_answer(row["query"], hits, cfg)
        cases.append(LLMTestCase(
            input=row["query"],
            actual_output=gen["answer"],
            expected_output=row["reference"],
            retrieval_context=gen["contexts"],
        ))
        time.sleep(CALL_GAP)  # generation is also a Gemini call

    # score
    method_scores = {}
    for name, metric in metrics:
        scores = []
        for tc in cases:
            scores.append(measure_with_retry(metric, tc))
            time.sleep(CALL_GAP)
        method_scores[name] = sum(scores) / len(scores) if scores else 0.0
        print(f"  {name:18s}: {method_scores[name]:.3f}")

    # merge into results file
    out = RESULTS_DIR / "benchmark.json"
    all_results = {}
    if out.exists():
        all_results = json.loads(out.read_text(encoding="utf-8"))
    all_results[args.method] = method_scores
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.method} -> {out}")

    # print whatever we have so far
    if all_results:
        names = ["Faithfulness", "Answer Relevancy", "Ctx Precision", "Ctx Recall"]
        print(f"\n{'='*60}\nRESULTS SO FAR\n{'='*60}")
        header = f"{'method':16s}" + "".join(f"{n:>16s}" for n in names)
        print(header); print("-" * len(header))
        for m, s in all_results.items():
            print(f"{m:16s}" + "".join(f"{s.get(n,0):>16.3f}" for n in names))

    _close_client()


if __name__ == "__main__":
    main()