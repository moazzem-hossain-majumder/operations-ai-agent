"""
test_tools.py
-------------
Tests that run without a live Ollama server, so the data and tool layers can be
verified in isolation before any model call happens.

Covers the read-only SQL guards, the cleaning pipeline's handling of each
injected data defect, and the eval scorer's edge cases.

Run:  python tests/test_tools.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  {detail}")


def test_sql_guards() -> None:
    from tools import query_operations_db

    print("\nRead-only SQL guards")
    blocked = [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET unit_price = 0",
        "INSERT INTO orders VALUES (1)",
        "ALTER TABLE orders ADD COLUMN x TEXT",
        "CREATE TABLE evil (id INT)",
        "PRAGMA table_info(orders)",
        "SELECT 1; DELETE FROM orders",
        "ATTACH DATABASE 'x.db' AS x",
    ]
    for sql in blocked:
        out = query_operations_db.invoke({"sql_query": sql})
        check(f"blocked: {sql[:38]}", out.startswith("REJECTED"), f"got: {out[:60]}")

    allowed = query_operations_db.invoke(
        {"sql_query": "SELECT COUNT(*) AS n FROM orders"}
    )
    check("plain SELECT allowed", "n" in allowed and not allowed.startswith("REJECTED"))

    cte = query_operations_db.invoke(
        {"sql_query": "WITH t AS (SELECT * FROM orders LIMIT 5) SELECT COUNT(*) AS n FROM t"}
    )
    check("WITH ... SELECT allowed", not cte.startswith("REJECTED"), f"got: {cte[:60]}")


def test_cleaning() -> None:
    from data_prep import clean_orders, RAW_CSV

    print("\nCleaning pipeline")
    raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)
    clean, report = clean_orders(raw)

    check("duplicates removed", report["exact_duplicates_removed"] > 0,
          f"got {report['exact_duplicates_removed']}")
    check("returns flagged", report["negative_quantity_rows_flagged"] > 0)
    check("anomalies flagged", report["price_anomalies_flagged"] > 0)

    # Dates: every non-blank source date must parse
    orig = raw.drop_duplicates()["order_date"].astype("string").str.strip()
    parsed_na = clean["order_date"].isna()
    blanks = (orig == "").sum()
    check("no non-blank date silently dropped",
          int(parsed_na.sum()) == int(blanks),
          f"{int(parsed_na.sum())} NaT vs {int(blanks)} blanks")

    # Money must be numeric
    check("unit_price is numeric",
          pd.api.types.is_numeric_dtype(clean["unit_price"]))

    # Category normalization: no leading/trailing whitespace, no case variants
    cats = clean["category"].dropna().unique().tolist()
    check("categories normalized",
          all(c == c.strip() for c in cats) and len(cats) <= 5,
          f"got {cats}")

    # Supplier normalization: no trailing dots or padding
    sups = clean["supplier"].dropna().unique().tolist()
    check("suppliers normalized",
          all(s == s.strip() and not s.endswith(".") for s in sups),
          f"got {sups}")

    # Returns must not inflate spend
    returns_in_value = clean.loc[clean["is_return"], "order_value"].dropna()
    check("return rows have non-negative order_value",
          bool((returns_in_value >= 0).all()))


def test_scorer() -> None:
    from evaluate import score_answer, score_tools

    print("\nEval scorer")
    check("exact number matches", score_answer("There are 20 returns", "20"))
    check("no false substring match", not score_answer("There are 215 rows", "21"))
    check("comma normalization", score_answer("Total 1,234 units", "1234"))
    check("name match", score_answer("Cement leads spend", "Cement"))
    check("empty answer fails", not score_answer("", "18"))
    check("routing pass", score_tools(["query_operations_db"], ["query_operations_db"]))
    check("routing fail", not score_tools(["query_operations_db"], ["search_policy_docs"]))
    check("no expectation passes", score_tools([], []))


def test_stats_tool() -> None:
    from tools import compute_statistics

    print("\nStatistics tool")
    out = compute_statistics.invoke({"column": "order_value"})
    check("numeric column works", "mean" in out, f"got {out[:60]}")

    bad = compute_statistics.invoke({"column": "supplier"})
    check("non-numeric column rejected", bad.startswith("ERROR"))

    grouped = compute_statistics.invoke(
        {"column": "lead_time_days", "group_by": "business_unit"}
    )
    check("group_by works", "Cement" in grouped, f"got {grouped[:60]}")

    bad_where = compute_statistics.invoke(
        {"column": "order_value", "where": "1=1; DROP TABLE orders"}
    )
    check("malicious WHERE rejected", bad_where.startswith("REJECTED"))


def main() -> int:
    print("=" * 60)
    print("Offline test suite (no API key required)")
    print("=" * 60)

    test_sql_guards()
    test_cleaning()
    test_scorer()
    test_stats_tool()

    print("\n" + "=" * 60)
    print(f"  {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
