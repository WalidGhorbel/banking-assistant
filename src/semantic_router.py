"""
Semantic routing fallback.

The rule-based router (router.py) is fast and free but brittle to phrasing:
"how many clients" matches, "what's our customer base size" does not. This module
adds an embedding-similarity fallback. When the rules don't recognise a data
question, we embed the question and compare it to labelled example questions; if it
is clearly closer to the data examples (above a threshold), we treat it as a data
question. Otherwise it stays on the RAG path.

Embeddings decide WHERE a question goes. The actual answering is still exact pandas
(for data) or retrieval (for knowledge) — similarity never computes a number.

route_semantically(question) -> (is_data: bool, score: float, hint: str)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"          # same model already used for retrieval
DATA_THRESHOLD = 0.45               # min similarity to the best data example
MARGIN = 0.06                       # data must beat knowledge by at least this

# Labelled examples. These are the router's "training set" — cheap to extend.
# They don't need to match user wording exactly; they're anchors in embedding
# space. More varied phrasings per intent = a wider net for similar questions.
DATA_EXAMPLES = [
    # per-client spending
    "how much does a client spend on groceries",
    "what does client C0001 spend each month",
    "spending breakdown for a customer",
    "where does this client's money go",
    "what are the expenses of a customer",
    # wealth / balance
    "how wealthy are our clients",
    "which clients have the most money",
    "what are the account balances",
    "how rich is the customer base",
    "who has the highest balance",
    "how much money do clients hold",
    # top / ranking
    "who are the top spenders on clothes",
    "which client spends the most",
    "rank customers by spending",
    "biggest spenders in the bank",
    # totals / aggregates
    "total spending across all clients",
    "how much do clients spend altogether",
    "combined monthly expenses",
    "overall spend by category",
    # averages / comparisons
    "average spending by account tier",
    "compare spending between tiers",
    "typical spend per customer",
    # counts
    "how many clients do we have",
    "what is our customer base size",
    "number of premium customers",
    "how many people bank with us",
    "count of active clients",
    # distributions / demographics
    "age distribution of clients",
    "how old are our customers",
    "how are clients split by city",
    "break down customers by region",
    "where are our clients located",
    "gender breakdown of customers",
    "customer demographics",
    "how are clients distributed across tiers",
]

KNOWLEDGE_EXAMPLES = [
    # how-to / process
    "how long does a SEPA transfer take",
    "how do I open an account",
    "what do I need to open an account",
    "how do I get a refund on a direct debit",
    "how do I close my account",
    "how do I change my phone number",
    "what should I do if my card is stolen",
    # definitions / facts
    "how many digits is a German IBAN",
    "what is a BIC",
    "what is SEPA",
    "what does IBAN mean",
    "what is a direct debit",
    # policy / security
    "how is my account secured",
    "can I use my card abroad",
    "what are the account types",
    "are there fees for foreign payments",
    "what is two-factor authentication",
    "how do standing orders work",
    "where can I withdraw cash",
]


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL)


@lru_cache(maxsize=1)
def _example_embeddings():
    m = _model()
    data_emb = m.encode(DATA_EXAMPLES, normalize_embeddings=True)
    know_emb = m.encode(KNOWLEDGE_EXAMPLES, normalize_embeddings=True)
    return data_emb, know_emb


def route_semantically(question: str) -> tuple[bool, float, str]:
    """Return (is_data, best_data_score, explanation)."""
    m = _model()
    q = m.encode([question], normalize_embeddings=True)[0]
    data_emb, know_emb = _example_embeddings()

    data_score = float(np.max(data_emb @ q))
    know_score = float(np.max(know_emb @ q))

    is_data = (data_score >= DATA_THRESHOLD) and (data_score - know_score >= MARGIN)
    hint = f"data={data_score:.3f} knowledge={know_score:.3f}"
    return is_data, data_score, hint


if __name__ == "__main__":
    import sys
    tests = sys.argv[1:] or [
        # data questions phrased unusually — should be DATA
        "what's our customer base size",
        "how wealthy are our clients",
        "break down customers by region",
        "who's got the fattest bank balance",
        "how old are the people who bank with us",
        "which customers burn the most on eating out",
        "give me a spending overview",
        # knowledge questions — should be RAG
        "how long does a SEPA transfer take",
        "what is an IBAN",
        "can I use my card in another country",
        "how do I report a stolen card",
        # neither — should stay RAG (safe default)
        "tell me a joke",
        "what's the weather today",
    ]
    for t in tests:
        is_data, score, hint = route_semantically(t)
        tag = "DATA" if is_data else "RAG "
        print(f"  {tag} | {hint} | {t!r}")
