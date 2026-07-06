"""
Structured-data query layer over the synthetic client table.

Predefined pandas functions — NO LLM does arithmetic, so answers are exact and
reproducible. Each function returns a dict with a `text` summary (for the chatbot)
and structured `data` (for charts). This is the reliable alternative to text-to-SQL.

The chatbot's data path calls these functions; it does not compute numbers itself.

CLI self-test:
    python src/query_clients.py --client C0001
    python src/query_clients.py --top groceries --n 5
    python src/query_clients.py --category-totals
    python src/query_clients.py --filter-city Leipzig --category groceries
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd

CSV_PATH = "data/processed/clients.csv"

CATEGORIES = ["groceries", "clothes", "bills", "dining", "transport",
              "entertainment", "health", "savings"]

# Map common German/English aliases to canonical column names
CATEGORY_ALIASES = {
    "lebensmittel": "groceries", "food": "groceries", "grocery": "groceries",
    "kleidung": "clothes", "clothing": "clothes",
    "rechnungen": "bills", "utilities": "bills",
    "essen": "dining", "restaurants": "dining", "eating out": "dining",
    "transport": "transport", "travel": "transport",
    "unterhaltung": "entertainment", "fun": "entertainment",
    "gesundheit": "health", "medical": "health",
    "sparen": "savings", "saving": "savings",
}


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
    path = Path(CSV_PATH)
    if not path.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run: python src/gen_clients.py --n 500")
    return pd.read_csv(path)


def resolve_category(name: str) -> str | None:
    key = name.strip().lower()
    if key in CATEGORIES:
        return key
    return CATEGORY_ALIASES.get(key)


# --------------------------- per-client lookup ----------------------------- #

def client_summary(client_id: str) -> dict:
    df = _df()
    row = df[df["client_id"].str.lower() == client_id.strip().lower()]
    if row.empty:
        return {"text": f"No client found with id {client_id}.", "data": None}
    r = row.iloc[0]
    spend = {c: float(r[c]) for c in CATEGORIES}
    total = sum(spend.values())
    text = (
        f"{r['name']} ({r['client_id']}), age {r['age']}, {r['city']}, "
        f"{r['account_tier']} tier. Balance: EUR {r['balance']:,.2f}. "
        f"Total monthly spending: EUR {total:,.2f}. "
        f"Top category: {max(spend, key=spend.get)} (EUR {max(spend.values()):,.2f})."
    )
    return {"text": text, "data": {"client": r.to_dict(), "spending": spend, "total": total},
            "chart": {"type": "doughnut",
                      "title": f"{r['name']} — monthly spending by category",
                      "labels": list(spend.keys()),
                      "values": [round(v, 2) for v in spend.values()]}}


def client_category(client_id: str, category: str) -> dict:
    df = _df()
    cat = resolve_category(category)
    if not cat:
        return {"text": f"Unknown category '{category}'.", "data": None}
    row = df[df["client_id"].str.lower() == client_id.strip().lower()]
    if row.empty:
        return {"text": f"No client found with id {client_id}.", "data": None}
    val = float(row.iloc[0][cat])
    name = row.iloc[0]["name"]
    return {
        "text": f"{name} ({client_id}) spends EUR {val:,.2f} per month on {cat}.",
        "data": {"client_id": client_id, "category": cat, "amount": val},
    }


# --------------------------- aggregates ------------------------------------ #

def category_totals(filters: dict | None = None) -> dict:
    df = _apply_filters(_df(), filters)
    totals = {c: round(float(df[c].sum()), 2) for c in CATEGORIES}
    scope = _describe_filters(filters, len(df))
    text = "Total monthly spending by category " + scope + ": " + ", ".join(
        f"{c} EUR {v:,.0f}" for c, v in totals.items()
    )
    return {"text": text, "data": {"totals": totals, "n_clients": len(df)},
            "chart": {"type": "bar",
                      "title": f"Total monthly spending by category {scope}",
                      "labels": list(totals.keys()),
                      "values": list(totals.values())}}


def top_spenders(category: str, n: int = 5, filters: dict | None = None) -> dict:
    cat = resolve_category(category)
    if not cat:
        return {"text": f"Unknown category '{category}'.", "data": None}
    df = _apply_filters(_df(), filters)
    top = df.nlargest(n, cat)[["client_id", "name", cat]]
    scope = _describe_filters(filters, len(df))
    lines = [f"{row['name']} ({row['client_id']}): EUR {row[cat]:,.2f}"
             for _, row in top.iterrows()]
    text = f"Top {n} spenders on {cat} {scope}:\n" + "\n".join(lines)
    return {"text": text, "data": {"category": cat, "top": top.to_dict("records")},
            "chart": {"type": "bar",
                      "title": f"Top {n} spenders on {cat} {scope}",
                      "labels": [f"{r['name']}" for _, r in top.iterrows()],
                      "values": [round(float(r[cat]), 2) for _, r in top.iterrows()]}}


def average_by(group_col: str, category: str) -> dict:
    cat = resolve_category(category)
    if not cat:
        return {"text": f"Unknown category '{category}'.", "data": None}
    df = _df()
    if group_col not in df.columns:
        return {"text": f"Unknown grouping column '{group_col}'.", "data": None}
    grp = df.groupby(group_col)[cat].mean().round(2).sort_values(ascending=False)
    text = f"Average {cat} spending by {group_col}: " + ", ".join(
        f"{k} EUR {v:,.2f}" for k, v in grp.items()
    )
    return {"text": text, "data": {"group_col": group_col, "category": cat,
                                    "averages": grp.to_dict()},
            "chart": {"type": "bar",
                      "title": f"Average {cat} spending by {group_col}",
                      "labels": [str(k) for k in grp.index.tolist()],
                      "values": [round(float(v), 2) for v in grp.values.tolist()]}}


def client_count(filters: dict | None = None) -> dict:
    df = _apply_filters(_df(), filters)
    scope = _describe_filters(filters, len(df))
    return {"text": f"There are {len(df)} clients {scope}.",
            "data": {"count": len(df)}}


def distribution(field: str) -> dict:
    """Count of clients grouped by a metadata field. Age is bucketed."""
    df = _df()
    field = field.strip().lower()

    if field in ("age", "ages"):
        bins = [17, 25, 35, 45, 55, 65, 120]
        labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        buckets = pd.cut(df["age"], bins=bins, labels=labels)
        counts = buckets.value_counts().reindex(labels).fillna(0).astype(int)
        title = "Client age distribution"
        keys, vals = list(counts.index), [int(v) for v in counts.values]
    else:
        col_map = {"city": "city", "cities": "city",
                   "tier": "account_tier", "tiers": "account_tier",
                   "account_tier": "account_tier",
                   "gender": "gender", "job": "job", "jobs": "job"}
        col = col_map.get(field)
        if not col:
            return {"text": f"Cannot show distribution for '{field}'.", "data": None}
        counts = df[col].value_counts()
        title = f"Client distribution by {col}"
        keys, vals = list(counts.index), [int(v) for v in counts.values]

    text = f"{title}: " + ", ".join(f"{k}: {v}" for k, v in zip(keys, vals))
    return {"text": text, "data": {"field": field, "counts": dict(zip(keys, vals))},
            "chart": {"type": "bar", "title": title,
                      "labels": [str(k) for k in keys], "values": vals}}


# --------------------------- filter helpers -------------------------------- #

def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return df
    out = df
    for col, val in filters.items():
        if col in ("min_age", "max_age"):
            continue
        if col in df.columns:
            out = out[out[col].astype(str).str.lower() == str(val).lower()]
    if filters.get("min_age") is not None:
        out = out[out["age"] >= filters["min_age"]]
    if filters.get("max_age") is not None:
        out = out[out["age"] <= filters["max_age"]]
    return out


def _describe_filters(filters: dict | None, n: int) -> str:
    if not filters:
        return f"(all {n} clients)"
    parts = [f"{k}={v}" for k, v in filters.items() if v is not None]
    return f"({', '.join(parts)}; {n} clients)"


# --------------------------- CLI ------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client")
    ap.add_argument("--category")
    ap.add_argument("--top")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--category-totals", action="store_true")
    ap.add_argument("--average-by")
    ap.add_argument("--filter-city")
    ap.add_argument("--filter-tier")
    args = ap.parse_args()

    filters = {}
    if args.filter_city:
        filters["city"] = args.filter_city
    if args.filter_tier:
        filters["account_tier"] = args.filter_tier
    filters = filters or None

    if args.client and args.category:
        res = client_category(args.client, args.category)
    elif args.client:
        res = client_summary(args.client)
    elif args.top:
        res = top_spenders(args.top, args.n, filters)
    elif args.category_totals:
        res = category_totals(filters)
    elif args.average_by and args.category:
        res = average_by(args.average_by, args.category)
    else:
        ap.error("provide a query, e.g. --client C0001, --top groceries, --category-totals")

    print(res["text"])


if __name__ == "__main__":
    main()
