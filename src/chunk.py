"""
Chunking strategies — one of the two benchmarked axes.

    fixed      : fixed-size character windows with overlap (naive baseline)
    recursive  : split on paragraph/sentence boundaries, then pack to target size
    semantic   : group adjacent sentences by embedding similarity (breaks on topic shift)

Each function takes a corpus record (dict) and yields chunk dicts that carry the
parent id + source, so retrieved chunks remain traceable for citations and eval.

Run standalone to materialize chunks for a given strategy:
    python src/chunk.py --strategy recursive --in data/processed/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator


def _emit(parent: dict, idx: int, text: str) -> dict:
    return {
        "chunk_id": f"{parent['id']}::{idx}",
        "parent_id": parent["id"],
        "source": parent.get("source", ""),
        "source_url": parent.get("source_url", ""),
        "title": parent.get("title", ""),
        "section": parent.get("section", ""),
        "text": text.strip(),
    }


def chunk_fixed(record: dict, size: int = 800, overlap: int = 120) -> Iterator[dict]:
    text = record["text"]
    step = max(1, size - overlap)
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start:start + size]
        if piece.strip():
            yield _emit(record, i, piece)


def chunk_recursive(record: dict, size: int = 800, overlap: int = 120) -> Iterator[dict]:
    """Boundary-aware packing. Uses langchain's RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    for i, piece in enumerate(splitter.split_text(record["text"])):
        if piece.strip():
            yield _emit(record, i, piece)


def chunk_semantic(record: dict, model_name: str = "all-MiniLM-L6-v2",
                   threshold: float = 0.55) -> Iterator[dict]:
    """Split into sentences, start a new chunk when similarity to the running
    chunk centroid drops below `threshold`."""
    import re
    import numpy as np
    from sentence_transformers import SentenceTransformer

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", record["text"]) if s.strip()]
    if not sentences:
        return
    model = SentenceTransformer(model_name)
    embs = model.encode(sentences, normalize_embeddings=True)

    chunks: list[list[str]] = [[sentences[0]]]
    centroid = embs[0].copy()
    for sent, emb in zip(sentences[1:], embs[1:]):
        sim = float(np.dot(centroid, emb))
        if sim < threshold:
            chunks.append([sent])
            centroid = emb.copy()
        else:
            chunks[-1].append(sent)
            centroid = (centroid + emb) / 2.0
    for i, group in enumerate(chunks):
        yield _emit(record, i, " ".join(group))


STRATEGIES = {"fixed": chunk_fixed, "recursive": chunk_recursive, "semantic": chunk_semantic}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--in", dest="infile", default="data/processed/corpus.jsonl")
    ap.add_argument("--out", dest="outfile", default=None)
    args = ap.parse_args()

    fn = STRATEGIES[args.strategy]
    out = Path(args.outfile or f"data/processed/chunks_{args.strategy}.jsonl")

    n = 0
    with out.open("w", encoding="utf-8") as w:
        for line in Path(args.infile).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for chunk in fn(record):
                w.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n += 1
    print(f"{args.strategy}: wrote {n} chunks -> {out}")


if __name__ == "__main__":
    main()
