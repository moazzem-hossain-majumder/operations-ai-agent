"""
make_sample_data.py
-------------------
Generates a deliberately MESSY synthetic procurement/sales dataset, plus a
couple of company policy documents.

Why messy on purpose: the data-cleaning step is a graded part of this project
(the JD explicitly asks for data cleaning, transformation and exploratory
analysis). Clean input data would make that step look trivial.

Injected problems, all of which data_prep.py has to handle:
  - inconsistent date formats (3 different ones, plus blanks)
  - currency stored as strings: "BDT 12,500.00", "12500", "৳12,500"
  - inconsistent category casing / whitespace: "Cement", " cement ", "CEMENT"
  - duplicate rows (exact + near-duplicate order ids)
  - missing values in quantity / unit_price
  - negative quantities (returns recorded inconsistently)
  - outlier unit prices (data entry slips: extra zero)
  - supplier names with trailing whitespace and inconsistent suffixes

Run:  python src/make_sample_data.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DOCS_DIR = ROOT / "data" / "docs"

SEED = 42

BUSINESS_UNITS = ["Cement", "Textiles", "Steel", "Jute", "Real Estate"]
CATEGORIES = ["Raw Material", "Packaging", "Machinery Parts", "Logistics", "Office Supplies"]
SUPPLIERS = [
    "Meghna Traders Ltd", "Padma Industrial", "Delta Supply Co",
    "Jamuna Enterprise", "Karnaphuli Logistics", "Surma Materials Ltd",
]
STATUSES = ["Delivered", "Pending", "Cancelled", "Partially Delivered"]


def _messy_date(dt: pd.Timestamp, rng: random.Random) -> str:
    """Return the same date in one of several inconsistent string formats."""
    style = rng.choice(["iso", "dmy", "mdy_slash", "blank", "textual"])
    if style == "iso":
        return dt.strftime("%Y-%m-%d")
    if style == "dmy":
        return dt.strftime("%d/%m/%Y")
    if style == "mdy_slash":
        return dt.strftime("%m-%d-%Y")
    if style == "textual":
        return dt.strftime("%d %b %Y")
    return ""


def _messy_amount(value: float, rng: random.Random) -> str:
    style = rng.choice(["bdt_prefix", "plain", "comma", "symbol"])
    if style == "bdt_prefix":
        return f"BDT {value:,.2f}"
    if style == "comma":
        return f"{value:,.2f}"
    if style == "symbol":
        return f"\u09f3{value:,.0f}"
    return f"{value:.2f}"


def _messy_category(cat: str, rng: random.Random) -> str:
    style = rng.choice(["clean", "lower_pad", "upper", "pad"])
    if style == "lower_pad":
        return f"  {cat.lower()} "
    if style == "upper":
        return cat.upper()
    if style == "pad":
        return f" {cat}"
    return cat


def build_orders(n: int = 600) -> pd.DataFrame:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    start = pd.Timestamp("2025-01-01")
    rows = []

    for i in range(n):
        order_date = start + pd.Timedelta(days=int(np_rng.integers(0, 540)))
        unit = rng.choice(BUSINESS_UNITS)
        category = rng.choice(CATEGORIES)
        supplier = rng.choice(SUPPLIERS)

        qty = int(np_rng.integers(1, 500))
        unit_price = float(np.round(np_rng.uniform(50, 5000), 2))

        # Inject a data-entry slip: extra zero on ~1.5% of prices
        if np_rng.random() < 0.015:
            unit_price *= 10

        # Returns sometimes recorded as negative quantity
        if np_rng.random() < 0.03:
            qty = -qty

        # Missing values
        qty_val = "" if np_rng.random() < 0.04 else qty
        price_val = "" if np_rng.random() < 0.035 else _messy_amount(unit_price, rng)

        rows.append({
            "order_id": f"PO-{2025000 + i}",
            "order_date": _messy_date(order_date, rng),
            "business_unit": unit,
            "category": _messy_category(category, rng),
            "supplier": supplier + rng.choice(["", " ", "  ", "."]),
            "quantity": qty_val,
            "unit_price": price_val,
            "status": rng.choice(STATUSES),
            "lead_time_days": int(np_rng.integers(2, 45)) if np_rng.random() > 0.05 else "",
        })

    df = pd.DataFrame(rows)

    # Exact duplicate rows (~3%)
    dupes = df.sample(frac=0.03, random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle so duplicates aren't all at the end
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


POLICY_DOCS = {
    "procurement_policy.md": """# Procurement Policy

## Purchase Order Approval Thresholds
- Purchase orders below BDT 100,000 may be approved by the Unit Procurement Officer.
- Purchase orders between BDT 100,000 and BDT 1,000,000 require Head of Procurement approval.
- Purchase orders above BDT 1,000,000 require Director-level approval and a minimum of
  three competitive supplier quotations.

## Supplier Onboarding
New suppliers must complete vendor registration, submit a valid trade licence and TIN
certificate, and pass a compliance screening before any purchase order is raised.

## Lead Time Standards
Standard expected lead time for raw material procurement is 21 days. Any purchase order
exceeding 30 days lead time must be escalated to the Head of Procurement with a written
justification.

## Returns and Cancellations
Cancelled orders must be recorded with a cancellation reason within 5 working days.
Returned goods are recorded as a separate credit note, never as a negative quantity on
the original purchase order.
""",
    "data_governance_policy.md": """# Data Governance Policy

## Data Quality Standards
All operational datasets must meet the following minimum standards before being used for
management reporting:
- No duplicate transaction records.
- All dates stored in ISO 8601 format (YYYY-MM-DD).
- All monetary values stored as numeric values in BDT, without currency symbols.
- Missing critical fields (quantity, unit price) must be flagged, not silently imputed.

## Anomaly Escalation
Any transaction whose value deviates more than three standard deviations from the
category mean must be flagged for manual review before inclusion in reporting.

## Retention
Transactional procurement data is retained for seven years. Derived analytical datasets
are retained for three years.

## Access
Access to raw supplier pricing data is restricted to the Procurement and Finance
functions. AI systems querying this data must operate on read-only credentials.
""",
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_orders()
    out_path = RAW_DIR / "procurement_orders_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} messy rows -> {out_path}")

    for name, content in POLICY_DOCS.items():
        (DOCS_DIR / name).write_text(content, encoding="utf-8")
        print(f"Wrote policy doc -> {DOCS_DIR / name}")


if __name__ == "__main__":
    main()
