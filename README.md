# RAG Banking Assistant — Method Benchmark

A retrieval-augmented generation (RAG) chatbot for European retail-banking questions,
built as a **benchmark of RAG methods** rather than a single fixed pipeline. The goal
is to measure — with numbers — which chunking strategy and which retrieval method
actually produce the best grounded answers on a banking corpus.

> Portfolio project for an AI Engineer application. Everything here is built on
> **openly-licensed or self-authored data**, so the whole repo is safe to publish.

## Why this project

Most RAG demos wire one embedding model to one vector store and stop. Real AI
engineering is about **evaluation and trade-offs**: chunking, retrieval strategy,
reranking, grounding, cost, and latency. This repo makes those trade-offs measurable.

## What gets benchmarked

The core experiment is a grid over two axes (embedding model and vector store are
held fixed, because they are infrastructure choices, not answer-quality levers):

| Axis | Variants |
|------|----------|
| **Chunking** | fixed-size · recursive · semantic |
| **Retrieval** | dense-only · hybrid (BM25 + dense) · hybrid + reranker |

That is 3 x 3 = 9 configurations, each scored on the same golden eval set.

Fixed choices (documented, not benchmarked):
- **Embeddings:** one model (configurable — e.g. a Gemini embedding model or `bge-large`)
- **Vector store:** one store (Qdrant). Chroma vs Qdrant is an ops choice; both return
  near-identical results on a corpus this size, so it is not treated as a quality lever.

## Data sources (all publishable)

| Layer | Source | Licence | Role |
|-------|--------|---------|------|
| Regulatory / payments | Deutsche Bundesbank SEPA pages (public authority) | Public-sector info | Term-dense corpus (IBAN, BIC, SEPA) |
| Government FAQ (PDF) | DPMA SEPA info leaflet | Government document | Tests the PDF ingestion path |
| Neobank FAQ | **Self-authored** synthetic FAQ (`data/raw/neobank_faq.jsonl`) | This repo (MIT) | Conversational customer-support layer |
| Test queries | Banking77 (PolyAI) | CC-BY-4.0 | Realistic customer query text for the eval set |

The self-authored FAQ deliberately replaces scraping a private bank's site: authoring
it ourselves keeps the repo fully publishable and demonstrates domain understanding.

## Pipeline

```
ingest → chunk → embed → index (Qdrant) → retrieve → (rerank) → generate → evaluate
```

- **Retrieval default:** hybrid (BM25 + dense) → rerank (cross-encoder) → top-k → LLM
- **Grounding:** the generator answers only from retrieved context and says so when the
  answer is not present ("if it isn't in the context, refuse") — important in banking
- **Guardrails (lightweight):** out-of-scope refusal, no personalised financial advice,
  no account-specific actions → hand off to a human

## Evaluation

Golden eval set (~40 Q&A pairs with reference answers, in `data/eval/`). Metrics via
[DeepEval](https://github.com/confident-ai/deepeval):
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer relevancy**
- **Contextual precision / recall** — did retrieval fetch the right chunks?

Results are written to `results/` as a table: every config scored, best combo highlighted.
That table is the deliverable.

## Repo layout

```
rag-banking-benchmark/
├── README.md
├── requirements.txt
├── configs/               # experiment configs (chunking x retrieval)
├── data/
│   ├── raw/               # self-authored FAQ + fetched public docs (gitignored)
│   ├── processed/         # cleaned, chunked JSONL
│   └── eval/              # golden Q&A set
├── src/
│   ├── ingest.py          # fetch + clean public pages/PDFs -> JSONL
│   ├── chunk.py           # fixed / recursive / semantic chunkers
│   ├── index.py           # embed + build Qdrant index
│   ├── retrieve.py        # dense / hybrid / hybrid+rerank
│   ├── generate.py        # grounded answer + guardrails
│   └── evaluate.py        # DeepEval harness over the config grid
├── results/               # metrics tables + plots
└── tests/
```

## Legal note

Fetched public documents are **not** committed to the repo — only the fetch script is.
Run `python src/ingest.py` locally to build your corpus. The self-authored FAQ and the
CC-BY-4.0 Banking77 queries are safe to include. This is general information, not legal
advice; check each source's terms before fetching.

## Quickstart

```bash
pip install -r requirements.txt
python src/ingest.py          # build corpus locally (public sources + self-authored FAQ)
python src/index.py --config configs/recursive_hybrid.yaml
python src/evaluate.py --grid  # run the full 3x3 benchmark
```
