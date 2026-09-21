"""
evaluate.py
-----------
Evaluation harness for the agent.

Most portfolio GenAI projects stop at "it produced an answer". This measures
whether the answer was *right*, and whether the agent took a sensible route to
it. Two scored dimensions per case:

  1. Factual accuracy   - does the answer contain the ground-truth value?
                          Ground truth is computed independently with Pandas
                          straight from the cleaned data, so the eval doesn't
                          depend on the LLM being right about its own output.
  2. Tool routing       - did the agent call the tools the question required?
                          A question about policy that never touches
                          search_policy_docs is answering from model memory,
                          which is exactly the failure mode we want to catch.

Run:  python src/evaluate.py
      python src/evaluate.py --verbose
"""

import argparse
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import build_agent, ask  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "operations.db"


def _load_orders() -> pd.DataFrame:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return pd.read_sql_query("SELECT * FROM orders", conn)


# --------------------------------------------------------------------------
# Ground truth, computed independently of the agent
# --------------------------------------------------------------------------

def gt_top_spend_unit() -> str:
    df = _load_orders()
    usable = df[(df["is_complete"] == 1) & (df["is_return"] == 0)]
    return usable.groupby("business_unit")["order_value"].sum().idxmax()


def gt_return_count() -> str:
    df = _load_orders()
    return str(int((df["is_return"] == 1).sum()))


def gt_anomaly_count() -> str:
    df = _load_orders()
    return str(int((df["is_price_anomaly"] == 1).sum()))


def gt_duplicate_count() -> str:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        rep = pd.read_sql_query("SELECT * FROM data_quality_report", conn)
    return str(int(rep["exact_duplicates_removed"].iloc[0]))


def gt_cancelled_count() -> str:
    df = _load_orders()
    return str(int((df["status"] == "Cancelled").sum()))


def gt_lead_time_policy() -> str:
    """Policy states the escalation threshold is 30 days."""
    return "30"


def gt_approval_threshold() -> str:
    """Orders above BDT 1,000,000 need Director approval + 3 quotations."""
    return "3"


# --------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------

@dataclass
class EvalCase:
    name: str
    question: str
    ground_truth: Callable[[], str]
    expected_tools: List[str] = field(default_factory=list)
    # Optional looser matcher when the answer is a name rather than a number
    match_mode: str = "substring"


CASES: List[EvalCase] = [
    EvalCase(
        name="top_spend_unit",
        question="Which business unit has the highest total spend? Name it.",
        ground_truth=gt_top_spend_unit,
        expected_tools=["query_operations_db"],
    ),
    EvalCase(
        name="return_count",
        question="How many order rows are recorded as returns?",
        ground_truth=gt_return_count,
        expected_tools=["query_operations_db"],
    ),
    EvalCase(
        name="price_anomalies",
        question="How many orders are flagged as price anomalies?",
        ground_truth=gt_anomaly_count,
        expected_tools=["query_operations_db", "check_data_quality"],
    ),
    EvalCase(
        name="duplicates_removed",
        question="How many exact duplicate rows were removed during data cleaning?",
        ground_truth=gt_duplicate_count,
        expected_tools=["check_data_quality"],
    ),
    EvalCase(
        name="cancelled_orders",
        question="How many purchase orders have status Cancelled?",
        ground_truth=gt_cancelled_count,
        expected_tools=["query_operations_db"],
    ),
    EvalCase(
        name="policy_lead_time",
        question="Above how many days of lead time must a purchase order be escalated to the Head of Procurement?",
        ground_truth=gt_lead_time_policy,
        expected_tools=["search_policy_docs"],
    ),
    EvalCase(
        name="policy_quotations",
        question="How many competitive supplier quotations are required for an order above BDT 1,000,000?",
        ground_truth=gt_approval_threshold,
        expected_tools=["search_policy_docs"],
    ),
    EvalCase(
        name="multi_step_compliance",
        question=(
            "What is the median lead time for Cement, and does it comply with the "
            "standard expected lead time in our procurement policy?"
        ),
        ground_truth=lambda: "21",
        expected_tools=["search_policy_docs"],
    ),
]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _normalize_number(text: str) -> str:
    """Strip thousands separators so '1,234' matches '1234'."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def score_answer(answer: str, truth: str) -> bool:
    a = _normalize_number(answer.lower())
    t = _normalize_number(truth.lower())
    if t.replace(".", "").isdigit():
        # Match the number as a standalone token, so '21' doesn't match '215'
        return re.search(rf"(?<!\d){re.escape(t)}(?!\d)", a) is not None
    return t in a


def score_tools(called: List[str], expected: List[str]) -> bool:
    """Pass if at least one expected tool was called (routing sanity check)."""
    if not expected:
        return True
    return any(tool in called for tool in expected)


def run_evaluation(verbose: bool = False) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run `python src/make_sample_data.py` and "
            "`python src/data_prep.py` first."
        )
    from agent import ollama_is_running, model_is_pulled, CHAT_MODEL

    if not ollama_is_running():
        raise RuntimeError("Ollama isn't running. Start it with `ollama serve` first.")
    if not model_is_pulled(CHAT_MODEL):
        raise RuntimeError(f"Model '{CHAT_MODEL}' not found. Run: ollama pull {CHAT_MODEL}")

    agent = build_agent(thread_memory=False)
    rows = []

    for i, case in enumerate(CASES, start=1):
        truth = case.ground_truth()
        try:
            result = ask(agent, case.question, thread_id=f"eval-{case.name}")
            answer = result["answer"]
            called = [c["tool"] for c in result["tool_calls"]]
            error = ""
        except Exception as e:
            answer, called, error = "", [], str(e)

        accurate = score_answer(answer, truth) if not error else False
        routed = score_tools(called, case.expected_tools) if not error else False

        rows.append({
            "case": case.name,
            "accurate": accurate,
            "tool_routing_ok": routed,
            "ground_truth": truth,
            "tools_called": ", ".join(called) or "-",
            "error": error,
        })

        status = "PASS" if accurate and routed else "FAIL"
        print(f"[{i}/{len(CASES)}] {case.name:<24} {status}")
        if verbose:
            print(f"      expected : {truth}")
            print(f"      tools    : {called}")
            print(f"      answer   : {answer[:300]}")
            if error:
                print(f"      ERROR    : {error}")
            print()

    df = pd.DataFrame(rows)

    print("\n=== Summary ===")
    print(f"  Factual accuracy : {df['accurate'].mean():.0%} "
          f"({df['accurate'].sum()}/{len(df)})")
    print(f"  Tool routing     : {df['tool_routing_ok'].mean():.0%} "
          f"({df['tool_routing_ok'].sum()}/{len(df)})")

    out = ROOT / "evals" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Full results -> {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print answers and traces")
    args = parser.parse_args()
    run_evaluation(verbose=args.verbose)
