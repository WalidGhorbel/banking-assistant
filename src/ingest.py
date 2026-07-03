"""
Ingestion: build the corpus from openly-licensed sources.

Sources:
  1. Self-authored neobank FAQ  -> data/raw/neobank_faq.jsonl (already in repo, MIT)
  2. Deutsche Bundesbank SEPA pages (public authority)  -> fetched at runtime
  3. DPMA SEPA info leaflet (government PDF)  -> fetched at runtime

Fetched public content is NOT committed to the repo. This script pulls it locally,
cleans it, and writes normalized records to data/processed/corpus.jsonl.

Design notes:
  - trafilatura extracts main article text and strips nav/footer/cookie boilerplate.
  - A Playwright fallback is included for JS-rendered pages (not needed for the
    public sources here, which are server-rendered, but shown for completeness).
  - Every record is a structured dict so retrieval can be traced back to a source
    (needed for citations and DeepEval contextual-precision).

Run:
    python src/ingest.py
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable

# Third-party (see requirements.txt): trafilatura, pdfplumber, requests
import trafilatura
import requests
import pdfplumber


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"

USER_AGENT = "rag-banking-benchmark/0.1 (portfolio project; contact: you@example.com)"
REQUEST_DELAY_SEC = 1.0  # be polite to real servers

# Public-sector sources. Bundesbank = public authority; DPMA leaflet = government PDF.
BUNDESBANK_PAGES = [
    ("bundesbank_sepa_overview",
     "https://www.bundesbank.de/en/tasks/payment-systems/services/sepa/content"),
    ("bundesbank_sepa_credit_transfer",
     "https://www.bundesbank.de/en/tasks/payment-systems/services/sepa/content/sepa-credit-transfer-626664"),
    ("bundesbank_sepa_direct_debit",
     "https://www.bundesbank.de/en/tasks/payment-systems/services/sepa/content/sepa-direct-debit-626654"),
    ("bundesbank_technical_standard_iban",
     "https://www.bundesbank.de/en/tasks/payment-systems/services/sepa/content/the-technical-standard-in-sepa-626774"),
]

DPMA_PDF = (
    "dpma_sepa_leaflet",
    "https://www.dpma.de/docs/english/formulare/allg_eng/a9534_1.pdf",
)


@dataclass
class Record:
    id: str
    source: str          # "self-authored" | "bundesbank" | "dpma"
    source_url: str
    title: str
    section: str
    text: str
    fetched_at: str


# ----------------------------- cleaning ------------------------------------ #

def clean_text(text: str) -> str:
    """Normalize unicode, collapse whitespace, drop common boilerplate lines."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)

    junk_markers = (
        "cookie", "accept all", "was this article helpful",
        "privacy settings", "manage consent",
    )
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(m in stripped.lower() for m in junk_markers):
            continue
        kept.append(stripped)

    # collapse to single-spaced paragraphs
    return "\n".join(kept).strip()


# ----------------------------- fetchers ------------------------------------ #

def fetch_html(url: str) -> str | None:
    """Fetch a server-rendered page and extract main text via trafilatura."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,   # keep SEPA fee/timing tables
        favor_precision=True,
    )
    return clean_text(text) if text else None


def fetch_html_js(url: str) -> str | None:
    """Fallback for JS-rendered pages. Requires: pip install playwright && playwright install chromium."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [skip] Playwright not installed; cannot render JS page.")
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, wait_until="networkidle", timeout=30000)
        html = page.content()
        browser.close()

    text = trafilatura.extract(html, include_tables=True, favor_precision=True)
    return clean_text(text) if text else None


def fetch_pdf(url: str) -> str | None:
    """Download a PDF and extract text (pdfplumber handles tables reasonably)."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tmp = PROCESSED_DIR / "_tmp.pdf"
    tmp.write_bytes(resp.content)

    pages_text: list[str] = []
    with pdfplumber.open(tmp) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    tmp.unlink(missing_ok=True)
    return clean_text("\n".join(pages_text))


# ----------------------------- loaders ------------------------------------- #

def load_self_authored() -> Iterable[Record]:
    """Read the committed, self-authored FAQ (already clean)."""
    path = RAW_DIR / "neobank_faq.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        yield Record(
            id=d["id"],
            source="self-authored",
            source_url="",
            title=d.get("title", ""),
            section=d.get("section", ""),
            text=d["text"],
            fetched_at=str(date.today()),
        )


def load_bundesbank() -> Iterable[Record]:
    for rec_id, url in BUNDESBANK_PAGES:
        print(f"  fetching {url}")
        text = fetch_html(url)
        if not text:
            print("    (empty via HTTP; trying JS fallback)")
            text = fetch_html_js(url)
        if not text:
            print("    [warn] no text extracted; skipping")
            continue
        yield Record(
            id=rec_id, source="bundesbank", source_url=url,
            title=rec_id.replace("_", " ").title(),
            section="SEPA / Payments", text=text,
            fetched_at=str(date.today()),
        )
        time.sleep(REQUEST_DELAY_SEC)


def load_dpma() -> Iterable[Record]:
    rec_id, url = DPMA_PDF
    print(f"  fetching {url}")
    text = fetch_pdf(url)
    if text:
        yield Record(
            id=rec_id, source="dpma", source_url=url,
            title="DPMA SEPA Payment Schemes (leaflet)",
            section="SEPA / Payments", text=text,
            fetched_at=str(date.today()),
        )
    time.sleep(REQUEST_DELAY_SEC)


# ----------------------------- main ---------------------------------------- #

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records: list[Record] = []

    print("[1/3] self-authored FAQ")
    records.extend(load_self_authored())

    print("[2/3] Bundesbank SEPA pages")
    try:
        records.extend(load_bundesbank())
    except Exception as e:  # noqa: BLE001 - keep ingestion resilient
        print(f"    [warn] Bundesbank fetch failed: {e}")

    print("[3/3] DPMA PDF leaflet")
    try:
        records.extend(load_dpma())
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] DPMA fetch failed: {e}")

    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    by_source: dict[str, int] = {}
    for r in records:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    print(f"\nWrote {len(records)} records to {CORPUS_PATH}")
    for src, n in sorted(by_source.items()):
        print(f"  {src:15s} {n}")


if __name__ == "__main__":
    main()
