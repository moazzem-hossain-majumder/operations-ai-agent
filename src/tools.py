"""
tools.py
--------
The tools the agent can call. This is the "tool calling" half of the agentic
workflow — the agent decides which of these to invoke, in what order, and
feeds results from one into the next (multi-step reasoning).

Four tools, covering the three data sources the JD mentions
(databases, documents, and computed analytics):

  1. query_operations_db   - read-only SQL against the cleaned SQLite table
  2. search_policy_docs    - semantic search over company policy documents
  3. compute_statistics    - Pandas/NumPy stats on a filtered slice
  4. check_data_quality    - surfaces the cleaning-stage quality report

Safety note: query_operations_db enforces read-only. An LLM writing free-form
SQL against a live system is a real risk, so anything that isn't a single
SELECT is rejected before it reaches the database.
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "operations.db"
DOCS_DIR = ROOT / "data" / "docs"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma_policies"

MAX_ROWS_RETURNED = 50

SCHEMA_DESCRIPTION = """
Table: orders
  order_id          TEXT     purchase order id, e.g. 'PO-2025049'
  order_date        TEXT     ISO date 'YYYY-MM-DD', may be NULL if missing in source
  business_unit     TEXT     one of: Cement, Textiles, Steel, Jute, Real Estate
  category          TEXT     one of: Raw Material, Packaging, Machinery Parts,
                             Logistics, Office Supplies
  supplier          TEXT     supplier company name
  quantity          REAL     may be negative for returns; may be NULL
  quantity_abs      REAL     absolute quantity
  unit_price        REAL     price in BDT, numeric; may be NULL
  order_value       REAL     quantity_abs * unit_price
  status            TEXT     Delivered / Pending / Cancelled / Partially Delivered
  lead_time_days    REAL     days from order to delivery; may be NULL
  is_return         INTEGER  1 if the row is a return (negative quantity)
  is_price_anomaly  INTEGER  1 if unit_price is >3 std devs from its category mean
  is_complete       INTEGER  1 if date, quantity and unit_price are all present

Table: data_quality_report  - single row, one column per data quality metric.
"""

_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


def _assert_read_only(sql: str) -> None:
    stripped = sql.strip().rstrip(";").strip()
    if not stripped.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT (or WITH ... SELECT) queries are permitted.")
    if ";" in stripped:
        raise ValueError("Multiple statements are not permitted.")
    if _FORBIDDEN_SQL.search(stripped):
        raise ValueError("Write/DDL operations are not permitted on this database.")


@tool(description=f"""Run a read-only SQL SELECT query against the cleaned procurement orders database.

Use this for any question about orders, spend, suppliers, business units,
lead times, statuses, returns or anomalies. Only SELECT queries are allowed.

The schema is:
{SCHEMA_DESCRIPTION}

Args:
    sql_query: A single SQLite SELECT statement.

Returns:
    The query result as a readable text table, truncated to 50 rows.
""")
def query_operations_db(sql_query: str) -> str:
    if not DB_PATH.exists():
        return (
            "ERROR: database not found. Run `python src/make_sample_data.py` then "
            "`python src/data_prep.py` to build it."
        )
    try:
        _assert_read_only(sql_query)
    except ValueError as e:
        return f"REJECTED: {e}"

    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
            df = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        return f"SQL ERROR: {e}. Check column names against the schema and retry."

    if df.empty:
        return "Query returned no rows."

    truncated = len(df) > MAX_ROWS_RETURNED
    body = df.head(MAX_ROWS_RETURNED).to_string(index=False)
    if truncated:
        body += f"\n... ({len(df)} rows total, showing first {MAX_ROWS_RETURNED})"
    return body


@tool
def compute_statistics(column: str, group_by: str = "", where: str = "") -> str:
    """Compute descriptive statistics (count, mean, median, std, min, max) for a numeric column.

    Use this when the question asks about averages, spread, variability or
    distribution rather than raw records. Prefer this over writing complex
    SQL aggregate queries by hand.

    Args:
        column: numeric column to analyse, e.g. 'order_value', 'lead_time_days', 'unit_price'.
        group_by: optional column to group by, e.g. 'business_unit' or 'category'.
        where: optional SQL WHERE clause body without the WHERE keyword,
               e.g. "status = 'Delivered' AND is_return = 0".

    Returns:
        A formatted statistics table.
    """
    if not DB_PATH.exists():
        return "ERROR: database not found. Run the data prep steps first."

    numeric_cols = {"quantity", "quantity_abs", "unit_price", "order_value", "lead_time_days"}
    if column not in numeric_cols:
        return f"ERROR: '{column}' is not a numeric column. Choose from: {sorted(numeric_cols)}"

    sql = f"SELECT * FROM orders"
    if where.strip():
        if _FORBIDDEN_SQL.search(where) or ";" in where:
            return "REJECTED: invalid WHERE clause."
        sql += f" WHERE {where}"

    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
            df = pd.read_sql_query(sql, conn)
    except Exception as e:
        return f"SQL ERROR: {e}"

    if df.empty:
        return "No rows matched the filter."

    series = pd.to_numeric(df[column], errors="coerce")

    def _stats(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return pd.Series({"count": 0})
        return pd.Series({
            "count": int(s.count()),
            "mean": float(np.round(s.mean(), 2)),
            "median": float(np.round(s.median(), 2)),
            "std": float(np.round(s.std(ddof=1), 2)) if len(s) > 1 else 0.0,
            "min": float(np.round(s.min(), 2)),
            "max": float(np.round(s.max(), 2)),
        })

    if group_by.strip():
        if group_by not in df.columns:
            return f"ERROR: '{group_by}' is not a column in orders."
        df["_value"] = series
        result = df.groupby(group_by)["_value"].apply(_stats).unstack()
        return result.to_string()

    return _stats(series).to_string()


@tool
def check_data_quality() -> str:
    """Report data quality issues found during the cleaning stage.

    Use this when asked about data completeness, missing values, duplicates,
    anomalies, or whether the data is reliable enough for a decision.

    Returns:
        The data quality report produced by the cleaning pipeline.
    """
    if not DB_PATH.exists():
        return "ERROR: database not found. Run the data prep steps first."
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query("SELECT * FROM data_quality_report", conn)
    if df.empty:
        return "No data quality report available."
    return df.T.rename(columns={0: "value"}).to_string()


def _load_policy_retriever():
    """Build or load the Chroma index over policy documents (lazy — needs API key)."""
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    # Local embedding model via Ollama — no API key, no cost.
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        store = Chroma(
            collection_name="policies",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        return store.as_retriever(search_kwargs={"k": 3})

    docs: List[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        ))
    if not docs:
        raise FileNotFoundError(f"No policy documents found in {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="policies",
        persist_directory=str(CHROMA_DIR),
    )
    return store.as_retriever(search_kwargs={"k": 3})


@tool
def search_policy_docs(question: str) -> str:
    """Search internal company policy documents (procurement policy, data governance policy).

    Use this whenever a question involves rules, thresholds, approval limits,
    compliance requirements or what the company's stated standard is — for
    example 'what approval does a 2 million BDT order need?' or 'what is the
    standard lead time?'. Combine with database results to judge whether actual
    operations comply with policy.

    Args:
        question: the policy question, in natural language.

    Returns:
        Relevant policy excerpts with their source filenames.
    """
    try:
        retriever = _load_policy_retriever()
        docs = retriever.invoke(question)
    except Exception as e:
        return f"ERROR searching policy documents: {e}"

    if not docs:
        return "No relevant policy text found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


ALL_TOOLS = [
    query_operations_db,
    compute_statistics,
    check_data_quality,
    search_policy_docs,
]
