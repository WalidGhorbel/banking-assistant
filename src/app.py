"""
FastAPI backend for the banking RAG chatbot.

Wraps the existing retrieve + generate pipeline behind a /chat endpoint and
serves the static frontend. Run:

    uvicorn src.app:app --reload --port 8000

Then open http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.retrieve import retrieve
from src.generate import generate_answer

# --- config: adjust if your collection / chunks path differ ---
COLLECTION = os.environ.get("RAG_COLLECTION", "banking_recursive")
CHUNKS_PATH = os.environ.get("RAG_CHUNKS", "data/processed/chunks_recursive.jsonl")
METHOD = os.environ.get("RAG_METHOD", "hybrid_rerank")

FRONTEND_DIR = Path("frontend")

app = FastAPI(title="Banking RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cfg() -> dict:
    return {
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "vector_store": {"collection": COLLECTION},
        "retrieval": {"method": METHOD, "top_k_dense": 20,
                      "top_k_bm25": 20, "final_k": 5, "reranker": None},
        "generation": {"model": "gemini-2.5-flash-lite"},
        "_chunks_path": CHUNKS_PATH,
    }


class ChatRequest(BaseModel):
    message: str


class Source(BaseModel):
    title: str
    text: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    cfg = _cfg()
    hits = retrieve(req.message, cfg)
    gen = generate_answer(req.message, hits, cfg)

    sources = [
        Source(
            title=h.get("title", ""),
            text=h.get("text", "")[:400],
            chunk_id=h.get("chunk_id", ""),
        )
        for h in hits
    ]
    return ChatResponse(answer=gen["answer"], sources=sources)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# serve any other static assets (css/js) if you split them out later
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")