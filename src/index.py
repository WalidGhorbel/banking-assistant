"""
Indexing: embed chunks and load them into a local Qdrant collection.

Run:
    python src/chunk.py --strategy recursive --in data/processed/corpus.jsonl
    python src/index.py --chunks data/processed/chunks_recursive.jsonl --collection banking_recursive
    python src/index.py --query "How long does a SEPA transfer take?" --collection banking_recursive
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


QDRANT_PATH = "qdrant_storage"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_chunks(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_index(chunks_path: str, collection: str, model_name: str = DEFAULT_MODEL) -> None:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path}. Run chunk.py first.")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    client = QdrantClient(path=QDRANT_PATH)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={
                "chunk_id": c["chunk_id"],
                "parent_id": c["parent_id"],
                "source": c.get("source", ""),
                "source_url": c.get("source_url", ""),
                "title": c.get("title", ""),
                "section": c.get("section", ""),
                "text": c["text"],
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} chunks into collection '{collection}' at {QDRANT_PATH}/")


def query_index(query: str, collection: str, model_name: str = DEFAULT_MODEL, k: int = 5) -> None:
    model = SentenceTransformer(model_name)
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

    client = QdrantClient(path=QDRANT_PATH)
    hits = client.query_points(collection_name=collection, query=qvec, limit=k).points


    print(f"\nTop {k} for: {query!r}\n" + "-" * 60)
    for i, h in enumerate(hits, 1):
        p = h.payload
        snippet = p["text"][:140].replace("\n", " ")
        print(f"{i}. [{h.score:.3f}] {p['title']}  ({p['chunk_id']})")
        print(f"     {snippet}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", help="path to chunks JSONL (build mode)")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--query", help="run a search instead of building")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if args.query:
        query_index(args.query, args.collection, args.model, args.k)
    elif args.chunks:
        build_index(args.chunks, args.collection, args.model)
    else:
        ap.error("provide --chunks to build, or --query to search")


if __name__ == "__main__":
    main()