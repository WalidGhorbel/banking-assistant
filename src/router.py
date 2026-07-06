"""
Router: decide whether a question is a structured-data question or a text/RAG
question, and (for data questions) dispatch to the right query_clients function.

Rule-based on purpose — deterministic, quota-free, debuggable. No LLM call is made
just to route. Text questions fall through to the existing RAG pipeline.

route(question) -> dict:
    {"kind": "data", "text": ..., "data": ...}   # answered here
    {"kind": "text"}                               # caller should use RAG

CLI self-test:
    python src/router.py "how much does C0001 spend on Lebensmittel"
    python src/router.py "top 5 grocery spenders in Leipzig"
    python src/router.py "how long does a SEPA transfer take"
"""

from __future__ import annotations

import re
import sys

try:
    from query_clients import (
        client_summary, client_category, category_totals,
        top_spenders, average_by, client_count, distribution,
        resolve_category, CATEGORIES, CATEGORY_ALIASES,
    )
except ImportError:
    from src.query_clients import (
        client_summary, client_category, category_totals,
        top_spenders, average_by, client_count, distribution,
        resolve_category, CATEGORIES, CATEGORY_ALIASES,
    )

CLIENT_RE = re.compile(r"\bC\d{3,4}\b", re.IGNORECASE)

AGG_WORDS = ("top", "most", "highest", "average", "avg", "total", "totals",
             "sum", "across all", "by city", "by tier", "by age", "compare")

ALL_CATEGORY_WORDS = set(CATEGORIES) | set(CATEGORY_ALIASES.keys())

CITIES = ["leipzig", "berlin", "munich", "hamburg", "cologne", "frankfurt",
          "dresden", "stuttgart"]
TIERS = ["standard", "premium", "metal"]


def _find_category(q: str) -> str | None:
    ql = q.lower()
    for word in ALL_CATEGORY_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", ql):
            return resolve_category(word)
    return None


def _find_filters(q: str) -> dict:
    ql = q.lower()
    filters = {}
    for city in CITIES:
        if re.search(rf"\b{city}\b", ql):
            filters["city"] = city.capitalize()
            break
    for tier in TIERS:
        if re.search(rf"\b{tier}\b", ql):
            filters["account_tier"] = tier
            break
    return filters or None


def _find_number(q: str, default: int = 5) -> int:
    m = re.search(r"\btop\s+(\d+)\b", q.lower()) or re.search(r"\b(\d+)\s+(?:top|highest)\b", q.lower())
    return int(m.group(1)) if m else default


DISTRIBUTION_FIELDS = {
    "age": "age", "ages": "age",
    "city": "city", "cities": "city",
    "tier": "tier", "tiers": "tier",
    "gender": "gender", "genders": "gender",
    "job": "job", "jobs": "job",
}
DIST_WORDS = ("distribution", "breakdown", "how many", "by age", "by city",
              "by tier", "by gender", "spread", "ages of", "diagram of ages")


def _find_distribution_field(q: str) -> str | None:
    ql = q.lower()
    # only treat as distribution when a client-metadata field is named
    for word, field in DISTRIBUTION_FIELDS.items():
        if re.search(rf"\b{word}\b", ql):
            # avoid clashing with spending categories (none overlap, but be safe)
            return field
    return None


def _is_distribution_question(q: str) -> bool:
    ql = q.lower()
    field = _find_distribution_field(q)
    if not field:
        return False
    has_dist_intent = any(w in ql for w in DIST_WORDS) or "clients" in ql or "client" in ql
    return has_dist_intent


def _is_count_question(q: str) -> bool:
    ql = q.lower()
    mentions_clients = "client" in ql or "clients" in ql or "customers" in ql
    count_intent = ("how many" in ql or "number of" in ql or "count" in ql)
    return mentions_clients and count_intent


def looks_like_data_question(q: str) -> bool:
    ql = q.lower()
    if CLIENT_RE.search(q):
        return True
    if _is_distribution_question(q):
        return True
    if _is_count_question(q):
        return True
    has_category = _find_category(q) is not None
    has_agg = any(w in ql for w in AGG_WORDS)
    # "total spending by category", "totals across categories" — aggregate over
    # the general notion of categories/spending, no specific category named
    mentions_spending = any(w in ql for w in ("category", "categories", "spending", "spend"))
    if has_agg and mentions_spending:
        return True
    return has_category and has_agg


CHART_WORDS = ("chart", "graph", "diagram", "plot", "visualize", "visualise",
               "diagramm", "zeige", "show me", "pie", "bar chart")


def wants_chart(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in CHART_WORDS)


def route(question: str, use_semantic: bool = True) -> dict:
    q = question.strip()

    # 1) fast rule-based check
    if looks_like_data_question(q):
        result = _route_data(q)
        if result.get("kind") == "data":
            result["wants_chart"] = wants_chart(q)
            result["routed_by"] = "rules"
        return result

    # 2) semantic fallback: rules didn't match, but embedding similarity might
    #    recognise an unusually-phrased data question. Only re-route if we can
    #    actually resolve it to a concrete data answer; otherwise stay on RAG.
    if use_semantic:
        try:
            from semantic_router import route_semantically
        except ImportError:
            from src.semantic_router import route_semantically
        is_data, score, _ = route_semantically(q)
        if is_data:
            result = _route_data(q)
            # only accept if the data path produced a real answer
            if result.get("kind") == "data" and result.get("data") is not None:
                result["wants_chart"] = wants_chart(q)
                result["routed_by"] = "semantic"
                return result
            # semantic said "data" but we can't resolve specifics -> give the
            # most useful general data answer we can: overall category totals
            fallback = category_totals(None)
            fallback["wants_chart"] = wants_chart(q)
            fallback["routed_by"] = "semantic_fallback"
            return {"kind": "data", **fallback}

    # 3) knowledge question -> RAG
    return {"kind": "text"}


def _route_data(q: str) -> dict:
    client_match = CLIENT_RE.search(q)
    category = _find_category(q)
    filters = _find_filters(q)
    ql = q.lower()

    # 0. Distribution ("age distribution", "clients by city")
    if _is_distribution_question(q):
        field = _find_distribution_field(q)
        res = distribution(field)
        return {"kind": "data", **res}

    # 0b. Count question ("how many clients")
    if _is_count_question(q):
        res = client_count(filters)
        return {"kind": "data", **res}

    # 1. Specific client + specific category  ->  exact lookup
    if client_match and category:
        res = client_category(client_match.group(0), category)
        return {"kind": "data", **res}

    # 2. Specific client, no category  ->  full summary
    if client_match:
        res = client_summary(client_match.group(0))
        return {"kind": "data", **res}

    # 3. Average by group
    if ("average" in ql or "avg" in ql) and category:
        group = "account_tier"
        if re.search(r"\bcity\b", ql) or re.search(r"\bcities\b", ql):
            group = "city"
        elif re.search(r"\bage\b", ql) or re.search(r"\bages\b", ql):
            group = "age"
        elif re.search(r"\btier\b", ql) or re.search(r"\btiers\b", ql):
            group = "account_tier"
        res = average_by(group, category)
        return {"kind": "data", **res}

    # 4. Top spenders
    if any(w in ql for w in ("top", "most", "highest")) and category:
        res = top_spenders(category, _find_number(q), filters)
        return {"kind": "data", **res}

    # 5. Totals (with optional filters)
    if any(w in ql for w in ("total", "totals", "sum", "across all")):
        res = category_totals(filters)
        return {"kind": "data", **res}

    # category present but intent unclear -> totals as a safe default
    if category:
        res = category_totals(filters)
        return {"kind": "data", **res}

    return {"kind": "text"}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python src/router.py \"your question\"")
        return
    question = " ".join(sys.argv[1:])
    result = route(question)
    if result["kind"] == "text":
        print(f"[ROUTE: text/RAG]  '{question}'\n  -> would go to the RAG pipeline")
    else:
        print(f"[ROUTE: data]  '{question}'\n  -> {result['text']}")


if __name__ == "__main__":
    main()
