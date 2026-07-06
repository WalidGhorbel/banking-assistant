"""
Generate synthetic client spending data — safe to publish (no real people).

Produces data/processed/clients.csv with per-client metadata plus monthly spending
across 8 categories. Deterministic (fixed seed) so results are reproducible.

Spending is correlated with account tier and age so the data has realistic structure
(premium clients spend more; older clients save more), which makes the analysis
queries and charts more interesting than pure random noise.

Run:
    python src/gen_clients.py --n 500
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT_PATH = Path("data/processed/clients.csv")

CITIES = ["Leipzig", "Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Dresden", "Stuttgart"]
JOBS = ["Engineer", "Teacher", "Nurse", "Designer", "Driver", "Student", "Manager",
        "Chef", "Electrician", "Accountant", "Retired", "Freelancer"]
TIERS = ["standard", "premium", "metal"]
GENDERS = ["female", "male", "other"]

CATEGORIES = ["groceries", "clothes", "bills", "dining", "transport",
              "entertainment", "health", "savings"]

FIRST_NAMES = ["Anna", "Lukas", "Mia", "Jonas", "Emma", "Felix", "Sophie", "Paul",
               "Laura", "Max", "Marie", "Ben", "Lena", "Tim", "Sara", "Leon",
               "Nora", "Elias", "Clara", "Noah"]
LAST_NAMES = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner",
              "Becker", "Hoffmann", "Koch", "Richter", "Klein", "Wolf", "Schröder"]


def _spending_profile(rng: random.Random, tier: str, age: int) -> dict:
    """Generate correlated monthly spending. Higher tiers spend more; older save more."""
    tier_mult = {"standard": 1.0, "premium": 1.6, "metal": 2.4}[tier]
    age_factor = 1.0 + (age - 35) / 100.0  # older -> slightly higher

    base = {
        "groceries": rng.uniform(200, 500),
        "clothes": rng.uniform(30, 250),
        "bills": rng.uniform(150, 600),
        "dining": rng.uniform(40, 300),
        "transport": rng.uniform(30, 200),
        "entertainment": rng.uniform(20, 200),
        "health": rng.uniform(10, 150),
        "savings": rng.uniform(50, 800),
    }
    out = {}
    for cat, val in base.items():
        v = val * tier_mult
        if cat == "savings":
            v *= age_factor * 1.3   # older save more
        out[cat] = round(v, 2)
    return out


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    start = date(2018, 1, 1)
    span_days = (date(2025, 1, 1) - start).days

    rows = []
    for i in range(1, n + 1):
        age = rng.randint(18, 75)
        tier = rng.choices(TIERS, weights=[0.6, 0.3, 0.1])[0]
        signup = start + timedelta(days=rng.randint(0, span_days))
        row = {
            "client_id": f"C{i:04d}",
            "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "age": age,
            "gender": rng.choices(GENDERS, weights=[0.48, 0.48, 0.04])[0],
            "city": rng.choice(CITIES),
            "job": rng.choice(JOBS),
            "account_tier": tier,
            "signup_date": signup.isoformat(),
            "balance": round(rng.uniform(-500, 25000), 2),
        }
        row.update(_spending_profile(rng, tier, age))
        rows.append(row)

    cols = ["client_id", "name", "age", "gender", "city", "job",
            "account_tier", "signup_date", "balance"] + CATEGORIES
    return pd.DataFrame(rows)[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} clients -> {OUT_PATH}")
    print(f"Columns: {', '.join(df.columns)}")
    print("\nSample:")
    print(df.head(3).to_string(index=False))
    print(f"\nTotal monthly groceries across all clients: EUR {df['groceries'].sum():,.2f}")


if __name__ == "__main__":
    main()
