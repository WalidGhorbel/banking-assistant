"""
Multi-class semantic router.

Instead of a binary "data vs knowledge" decision, this matches a question against
labelled examples grouped by *intent*. Each intent maps to a specific query. The
router embeds the question, finds the nearest intent by cosine similarity, and — if
it's confident enough and clearly a data intent — returns that intent so the caller
can dispatch to the right query function.

Embeddings decide WHICH query fits. The query itself still runs exact pandas; the
embedding never computes a number.

classify(question) -> (intent, score, all_scores)
    intent is one of the INTENTS keys, or "knowledge" (-> RAG), or None (uncertain).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
MIN_SCORE = 0.42          # a data intent must reach at least this similarity
MARGIN = 0.04             # and beat the best knowledge example by this much

# Each intent is a list of example phrasings. The key is the intent name the
# caller dispatches on. Extend an intent by adding example sentences — no new code.
INTENTS: dict[str, list[str]] = {
    "client_summary": [
        "tell me about client C0001",
        "show me a client's profile",
        "what does this customer look like",
        "give me an overview of a client",
    ],
    "client_spend_category": [
        "how much does a client spend on groceries",
        "what does a customer spend on clothes",
        "a client's spending in one category",
    ],
    "wealth": [
        "how wealthy are our clients",
        "who has the highest balance",
        "which clients have the most money",
        "how rich is the customer base",
        "show me account balances",
        "who's got the biggest bank balance",
        "richest customers",
    ],
    "top_spenders": [
        "who are the top spenders on clothes",
        "which client spends the most",
        "biggest spenders in the bank",
        "rank customers by spending",
        "who burns the most on eating out",
    ],
    "category_totals": [
        "total spending across all clients",
        "how much do clients spend altogether",
        "overall spend by category",
        "combined monthly expenses",
        "give me a spending overview",
    ],
    "average_by": [
        "average spending by account tier",
        "compare spending between tiers",
        "typical spend per customer",
        "mean grocery spend by city",
    ],
    "count": [
        "how many clients do we have",
        "what is our customer base size",
        "number of premium customers",
        "how many people bank with us",
        "count of clients",
    ],
    "distribution": [
        "age distribution of clients",
        "how old are our customers",
        "how are clients split by city",
        "break down customers by region",
        "where are our clients located",
        "gender breakdown of customers",
        "customer demographics",
        "distribution of clients across tiers",
    ],
}

KNOWLEDGE_EXAMPLES = [
    "how long does a SEPA transfer take",
    "how do I open an account",
    "what do I need to open an account",
    "how do I get a refund on a direct debit",
    "how do I close my account",
    "how do I change my phone number",
    "what should I do if my card is stolen",
    "how many digits is a German IBAN",
    "what is a BIC",
    "what is SEPA",
    "what does IBAN mean",
    "how is my account secured",
    "can I use my card abroad",
    "what are the account types",
    "how do standing orders work",
    "where can I withdraw cash",
]


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL)


@lru_cache(maxsize=1)
def _index():
    """Pre-embed every example. Returns (intent_names, matrix) and knowledge matrix."""
    m = _model()
    intent_names, intent_vecs = [], []
    for intent, examples in INTENTS.items():
        embs = m.encode(examples, normalize_embeddings=True)
        for e in embs:
            intent_names.append(intent)
            intent_vecs.append(e)
    know = m.encode(KNOWLEDGE_EXAMPLES, normalize_embeddings=True)
    return intent_names, np.array(intent_vecs), know


def classify(question: str):
    """Return (intent, score, detail).
    intent: an INTENTS key if confidently a data question,
            "knowledge" if closer to knowledge examples,
            None if too uncertain to call.
    """
    m = _model()
    q = m.encode([question], normalize_embeddings=True)[0]
    intent_names, intent_mat, know = _index()

    sims = intent_mat @ q
    best_i = int(np.argmax(sims))
    best_intent = intent_names[best_i]
    best_data = float(sims[best_i])
    best_know = float(np.max(know @ q))

    detail = f"intent={best_intent} data={best_data:.3f} knowledge={best_know:.3f}"

    if best_know > best_data:
        return "knowledge", best_know, detail
    if best_data < MIN_SCORE or (best_data - best_know) < MARGIN:
        return None, best_data, detail
    return best_intent, best_data, detail


if __name__ == "__main__":
    import sys
    tests = sys.argv[1:] or [
        "how wealthy are our clients",
        "who's got the fattest bank balance",
        "break down customers by region",
        "how old are the people who bank with us",
        "which customers burn the most on eating out",
        "give me a spending overview",
        "how many people bank with us",
        "how long does a SEPA transfer take",
        "what is an IBAN",
        "tell me a joke",
    ]
    for t in tests:
        intent, score, detail = classify(t)
        print(f"  {str(intent):22} | {detail} | {t!r}")
