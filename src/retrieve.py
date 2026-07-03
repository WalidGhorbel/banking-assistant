"""
Retrieval strategies — the second benchmarked axis.

    dense          : embedding similarity over the Qdrant collection
    hybrid         : BM25 (lexical) + dense, fused with reciprocal rank fusion (RRF)
    hybrid_rerank  : hybrid candidates re-scored by a cross-encoder, keep final_k

One entry point:  retrieve(query, cfg) -> list[hit dict]
Each hit dict has: chunk_id, parent_id, title, section, source, source_url, text, score.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "qdrant_storage"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=4)
def _embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def _close_client() -> None:
    try:
        _client().close()
        _client.cache_clear()
    except Exception:
        pass


def _load_chunks(chunks_path: str) -> list[dict]:
    rows = []
    for line in Path(chunks_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _dense(query: str, cfg: dict, k: int) -> list[dict]:
    model = _embedder(cfg["embedding"]["model"])
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    res = _client().query_points(
        collection_name=cfg["vector_store"]["collection"],
        query=qvec,
        limit=k,
    ).points
    hits = []
    for r in res:
        h = dict(r.payload)
        h["score"] = float(r.score)
        hits.append(h)
    return hits


def _bm25(query: str, cfg: dict, k: int) -> list[dict]:
    from rank_bm25 import BM25Okapi

    chunks = _load_chunks(cfg["_chunks_path"])
    corpus_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:k]
    hits = []
    for c, s in ranked:
        h = dict(c)
        h["score"] = float(s)
        hits.append(h)
    return hits


def _rrf_fuse(dense_hits: list[dict], bm25_hits: list[dict], final_k: int, c: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion. Each list contributes 1/(c + rank) per doc; sum, then re-rank."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for ranking in (dense_hits, bm25_hits):
        for rank, hit in enumerate(ranking):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + rank)
            payloads.setdefault(cid, hit)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]
    out = []
    for cid, s in fused:
        h = dict(payloads[cid])
        h["score"] = float(s)
        out.append(h)
    return out


def _hybrid(query: str, cfg: dict) -> list[dict]:
    r = cfg["retrieval"]
    dense_hits = _dense(query, cfg, r.get("top_k_dense", 20))
    bm25_hits = _bm25(query, cfg, r.get("top_k_bm25", 20))
    return _rrf_fuse(dense_hits, bm25_hits, r.get("final_k", 5))


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _rerank(candidates: list[dict], query: str, cfg: dict) -> list[dict]:
    r = cfg["retrieval"]
    model_name = r.get("reranker") or DEFAULT_RERANKER
    ce = _cross_encoder(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    out = []
    for c, s in reranked[: r.get("final_k", 5)]:
        h = dict(c)
        h["score"] = float(s)
        out.append(h)
    return out


def retrieve(query: str, cfg: dict) -> list[dict]:
    method = cfg["retrieval"]["method"]
    if method == "dense":
        return _dense(query, cfg, cfg["retrieval"].get("final_k", 5))
    if method == "hybrid":
        return _hybrid(query, cfg)
    if method == "hybrid_rerank":
        wide = dict(cfg)
        wide_ret = dict(cfg["retrieval"])
        wide_ret["final_k"] = max(cfg["retrieval"].get("final_k", 5) * 3, 15)
        wide["retrieval"] = wide_ret
        candidates = _hybrid(query, wide)
        return _rerank(candidates, query, cfg)
    raise ValueError(f"unknown retrieval method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--chunks", required=True, help="chunks JSONL used to build the index")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--method", choices=["dense", "hybrid", "hybrid_rerank"], default="hybrid")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    cfg = {
        "embedding": {"model": args.model},
        "vector_store": {"collection": args.collection},
        "retrieval": {
            "method": args.method,
            "top_k_dense": 20,
            "top_k_bm25": 20,
            "final_k": args.k,
            "reranker": None,
        },
        "_chunks_path": args.chunks,
    }

    hits = retrieve(args.query, cfg)
    print(f"\n[{args.method}] Top {len(hits)} for: {args.query!r}\n" + "-" * 60)
    for i, h in enumerate(hits, 1):
        snippet = h["text"][:130].replace("\n", " ")
        print(f"{i}. [{h['score']:.4f}] {h['title']}  ({h['chunk_id']})")
        print(f"     {snippet}...")

    _close_client()


if __name__ == "__main__":
    main()