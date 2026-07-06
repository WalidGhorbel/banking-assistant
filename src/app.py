"""
FastAPI backend for the banking assistant.

Routes each question:
  - data question  -> structured-data query layer (exact pandas, no LLM math)
  - text question  -> RAG pipeline (retrieve + grounded generate)

Run:
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
from src.router import route

COLLECTION = os.environ.get("RAG_COLLECTION", "banking_recursive")
CHUNKS_PATH = os.environ.get("RAG_CHUNKS", "data/processed/chunks_recursive.jsonl")
METHOD = os.environ.get("RAG_METHOD", "hybrid_rerank")

FRONTEND_DIR = Path("frontend")
REFUSAL_MARKER = "I don't have that information in my sources"

app = FastAPI(title="Banking Assistant")

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
    kind: str            # "data" or "text"
    sources: list[Source]
    chart: dict | None = None   # structured data for optional charting
    wants_chart: bool = False   # user explicitly asked for a chart


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.message

    # 1) try the structured-data router first
    routed = route(question)
    if routed["kind"] == "data":
        return ChatResponse(
            answer=routed["text"],
            kind="data",
            sources=[],
            chart=routed.get("chart"),
            wants_chart=routed.get("wants_chart", False),
        )

    # 2) otherwise, RAG
    cfg = _cfg()
    hits = retrieve(question, cfg)
    gen = generate_answer(question, hits, cfg)
    answer = gen["answer"]

    if REFUSAL_MARKER.lower() in answer.lower():
        sources: list[Source] = []
    else:
        sources = [
            Source(title=h.get("title", ""), text=h.get("text", "")[:400],
                   chunk_id=h.get("chunk_id", ""))
            for h in hits
        ]

    return ChatResponse(answer=answer, kind="text", sources=sources, chart=None)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
