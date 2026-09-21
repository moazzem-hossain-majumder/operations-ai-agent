"""
data_prep.py
------------
Pandas/NumPy cleaning pipeline: messy CSV -> validated DataFrame -> SQLite.

This is the "data cleaning, transformation, exploratory analysis and basic
statistical analysis" layer. It deliberately produces a *data quality report*
alongside the cleaned table, because the governance policy in data/docs says
missing critical fields must be flagged rather than silently imputed — the
agent can then answer questions about data quality itself.

Run:  python src/data_prep.py
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "procurement_orders_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "operations.db"
CLEAN_CSV = PROCESSED_DIR / "procurement_orders_clean.csv"

# Anomaly threshold from the data governance policy: >3 sigma from category mean
SIGMA_THRESHOLD = 3.0


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Handle the several date formats present in the raw export.

    pandas can infer most of these, but mixed day-first and month-first formats
    in one column are genuinely ambiguous, so we try explicit formats in order
    of specificity and fall back to a flexible parse for the remainder.
    """
    s = series.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y"]
    for fmt in formats:
        mask = result.isna() & s.notna() & (s != "")
        if not mask.any():
            break
        parsed = pd.to_datetime(s[mask], format=fmt, errors="coerce")
        result[mask] = parsed

    return result


def _parse_money(series: pd.Series) -> pd.Series:
    """Strip currency prefixes/symbols/commas and coerce to float."""
    s = (
        series.astype("string")
        .str.strip()
        .str.replace("BDT", "", regex=False)
        .str.replace("\u09f3", "", regex=False)   # Bangladeshi Taka sign
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    s = s.replace("", pd.NA)
    return pd.to_numeric(s, errors="coerce")


def _normalize_text(series: pd.Series) -> pd.Series:
    """Collapse whitespace and normalize casing to Title Case."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )


def _normalize_supplier(series: pd.Series) -> pd.Series:
    """Strip trailing punctuation/whitespace drift from supplier names."""
    return (
        series.astype("string")
        .str.strip()
        .str.rstrip(".")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw orders frame and return (clean_df, quality_report)."""
    report: Dict[str, Any] = {"rows_in": len(df)}

    # --- 1. Remove exact duplicate rows -----------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = before - len(df)

    # --- 2. Normalize text columns ----------------------------------------
    df["business_unit"] = _normalize_text(df["business_unit"])
    df["category"] = _normalize_text(df["category"])
    df["status"] = _normalize_text(df["status"])
    df["supplier"] = _normalize_supplier(df["supplier"])

    # --- 3. Parse dates ----------------------------------------------------
    df["order_date"] = _parse_dates(df["order_date"])
    report["missing_or_unparseable_dates"] = int(df["order_date"].isna().sum())

    # --- 4. Parse numerics -------------------------------------------------
    df["quantity"] = pd.to_numeric(
        df["quantity"].astype("string").str.strip().replace("", pd.NA), errors="coerce"
    )
    df["unit_price"] = _parse_money(df["unit_price"])
    df["lead_time_days"] = pd.to_numeric(
        df["lead_time_days"].astype("string").str.strip().replace("", pd.NA),
        errors="coerce",
    )

    report["missing_quantity"] = int(df["quantity"].isna().sum())
    report["missing_unit_price"] = int(df["unit_price"].isna().sum())
    report["missing_lead_time"] = int(df["lead_time_days"].isna().sum())

    # --- 5. Handle negative quantities ------------------------------------
    # Policy says returns are credit notes, not negative quantities. We keep the
    # rows but tag them so they can be excluded from spend totals rather than
    # silently dropped.
    df["is_return"] = df["quantity"] < 0
    report["negative_quantity_rows_flagged"] = int(df["is_return"].sum())
    df["quantity_abs"] = df["quantity"].abs()

    # --- 6. Derived measure ------------------------------------------------
    df["order_value"] = df["quantity_abs"] * df["unit_price"]

    # --- 7. Flag statistical outliers per category (z-score > 3) ----------
    df["is_price_anomaly"] = False
    for category, group in df.groupby("category", dropna=True):
        prices = group["unit_price"].dropna()
        if len(prices) < 10:
            continue
        mean, std = prices.mean(), prices.std(ddof=1)
        if std == 0 or np.isnan(std):
            continue
        z = (group["unit_price"] - mean).abs() / std
        df.loc[group.index, "is_price_anomaly"] = (z > SIGMA_THRESHOLD).fillna(False)

    report["price_anomalies_flagged"] = int(df["is_price_anomaly"].sum())

    # --- 8. Completeness flag ---------------------------------------------
    df["is_complete"] = (
        df["order_date"].notna() & df["quantity"].notna() & df["unit_price"].notna()
    )
    report["complete_rows"] = int(df["is_complete"].sum())
    report["rows_out"] = len(df)

    return df, report


def exploratory_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Basic EDA: spend and lead time by business unit."""
    usable = df[df["is_complete"] & ~df["is_return"]]
    summary = (
        usable.groupby("business_unit")
        .agg(
            orders=("order_id", "count"),
            total_spend=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            median_lead_time=("lead_time_days", "median"),
        )
        .round(2)
        .sort_values("total_spend", ascending=False)
    )
    return summary


def write_sqlite(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    """Persist cleaned data + quality report to SQLite for the agent to query."""
    import sqlite3

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["order_date"] = out["order_date"].dt.strftime("%Y-%m-%d")

    with sqlite3.connect(DB_PATH) as conn:
        out.to_sql("orders", conn, if_exists="replace", index=False)
        pd.DataFrame([report]).to_sql(
            "data_quality_report", conn, if_exists="replace", index=False
        )

    out.to_csv(CLEAN_CSV, index=False)


def main() -> Dict[str, Any]:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run `python src/make_sample_data.py` first."
        )

    raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)
    clean, report = clean_orders(raw)

    write_sqlite(clean, report)

    print("=== Data quality report ===")
    for k, v in report.items():
        print(f"  {k:<34} {v}")

    print("\n=== Spend by business unit ===")
    print(exploratory_summary(clean).to_string())
    print(f"\nWrote SQLite DB -> {DB_PATH}")
    return report


if __name__ == "__main__":
    main()
