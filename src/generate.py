"""
Grounded answer generation with lightweight banking guardrails.

generate_answer(query, hits, cfg) -> dict with:
    answer        : the model's grounded response (or a refusal)
    contexts      : the retrieved chunk texts used (for DeepEval)
    used_sources  : chunk_ids cited

Grounding rule: answer ONLY from retrieved context; if the answer is not present,
say so rather than inventing. This is the anti-hallucination behavior that matters
most in a banking setting.
"""

from __future__ import annotations

import os
from google import genai
from google.genai import types

DEFAULT_GEN_MODEL = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for a European retail bank. "
    "Answer the user's question using ONLY the provided context passages. "
    "If the answer is not contained in the context, reply exactly: "
    "\"I don't have that information in my sources. Please contact support.\" "
    "Do not give personalised financial or legal advice. "
    "Do not perform or promise account-specific actions; direct the user to a human for those. "
    "Be concise and, where useful, mention which passage you used."
)


def _build_prompt(query: str, hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[Passage {i}] (source: {h.get('title','')})\n{h['text']}")
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return f"Context passages:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

import time

def generate_answer(query: str, hits: list[dict], cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    model = cfg.get("generation", {}).get("model", DEFAULT_GEN_MODEL)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = _build_prompt(query, hits)

    resp = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    max_output_tokens=400,
                ),
            )
            break
        except Exception as e:
            msg = str(e)
            if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
            raise
    return {
        "answer": (resp.text or "").strip(),
        "contexts": [h["text"] for h in hits],
        "used_sources": [h.get("chunk_id", "") for h in hits],
    }


if __name__ == "__main__":
    import argparse
    from retrieve import retrieve

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--method", default="hybrid_rerank",
                    choices=["dense", "hybrid", "hybrid_rerank"])
    args = ap.parse_args()

    cfg = {
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "vector_store": {"collection": args.collection},
        "retrieval": {"method": args.method, "top_k_dense": 20,
                      "top_k_bm25": 20, "final_k": 5, "reranker": None},
        "generation": {"model": DEFAULT_GEN_MODEL},
        "_chunks_path": args.chunks,
    }

    hits = retrieve(args.query, cfg)
    result = generate_answer(args.query, hits, cfg)
    print("\nQ:", args.query)
    print("\nA:", result["answer"])
    print("\nSources:", ", ".join(result["used_sources"]))