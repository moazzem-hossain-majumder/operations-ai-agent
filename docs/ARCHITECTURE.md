# Solution Architecture

## 1. Problem statement

Operational questions in a large conglomerate ("which unit is overspending?",
"are our lead times within policy?") currently require someone to pull a report,
cross-reference a policy PDF, and manually reconcile the two. The answer depends
on *both* transactional data and written policy, which live in different systems.

This project builds an AI agent that answers such questions end to end: it
queries the transactional database, looks up the relevant policy clause, and
reasons over the combination.

## 2. Why an agent rather than a fixed pipeline

A fixed RAG chain retrieves once and generates once. That fails on questions
like *"is Cement's median lead time within policy?"*, which needs:

1. a database aggregation (median lead time for Cement), **then**
2. a policy lookup (what the standard lead time is), **then**
3. a comparison between the two.

The second step depends on the first being understood, and the agent cannot know
in advance how many steps a question needs. An agent loop — model decides, calls
a tool, reads the result, decides again — handles this. A fixed chain cannot.

## 3. Component diagram

```
                       ┌──────────────────────────┐
  Raw CSV export  ───► │  data_prep.py            │
  (messy)              │  Pandas/NumPy cleaning   │
                       │  + quality report        │
                       └────────────┬─────────────┘
                                    │
                       ┌────────────▼─────────────┐
                       │  SQLite: operations.db   │
                       │  orders + quality report │
                       └────────────┬─────────────┘
                                    │ (read-only)
  Policy docs ──► Chroma ───┐       │
  (.md)           vectors   │       │
                            ▼       ▼
                       ┌──────────────────────────┐
                       │  tools.py                │
                       │  4 tools exposed to LLM  │
                       └────────────┬─────────────┘
                                    │
                       ┌────────────▼─────────────┐
                       │  agent.py                │
                       │  create_agent + memory   │
                       │  model ⇄ tools loop      │
                       └──────┬────────────┬──────┘
                              │            │
                    ┌─────────▼──┐   ┌─────▼────────┐
                    │  app.py    │   │ evaluate.py  │
                    │  chat UI   │   │ eval harness │
                    └────────────┘   └──────────────┘
```

## 4. Data layer

The raw export deliberately contains realistic defects: three inconsistent date
formats plus blanks, currency stored as strings (`BDT 12,500.00`, `৳12,500`),
category casing/whitespace drift, exact duplicate rows, missing quantities and
prices, returns recorded as negative quantities, and data-entry slips (an extra
zero on unit price).

`data_prep.py` resolves these and, critically, **flags rather than imputes**.
The data governance policy states that missing critical fields must be surfaced,
not silently filled — so the cleaned table carries `is_complete`, `is_return`
and `is_price_anomaly` columns, and the agent is instructed to mention when a
figure excludes incomplete rows.

Anomaly detection uses a per-category z-score with a 3σ threshold, matching the
escalation rule written in the governance policy. The threshold lives in one
constant so it can be tuned in one place.

## 5. Tool layer

| Tool | Purpose | Backed by |
|---|---|---|
| `query_operations_db` | Factual record/aggregate questions | SQLite (read-only) |
| `compute_statistics` | Mean/median/std/spread questions | Pandas + NumPy |
| `check_data_quality` | Reliability and completeness questions | Quality report table |
| `search_policy_docs` | Rules, thresholds, compliance | Chroma + local Ollama embeddings |

### Read-only enforcement

An LLM emitting free-form SQL against a business database is a genuine risk.
Three layers of defence:

1. The connection is opened with SQLite's `mode=ro` URI flag.
2. Queries must begin with `SELECT` or `WITH`.
3. A regex rejects `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/...`, and
   multiple statements (`;`) are refused so a query can't be chained.

This is verified by tests in `tests/test_tools.py`.

## 6. Agent layer

Built with `create_agent` (LangChain 1.x / LangGraph), which compiles to a
`model ⇄ tools` graph that loops until the model stops requesting tools.

**Memory** is a LangGraph checkpointer keyed by `thread_id`. Each Streamlit
session gets its own thread, so a follow-up like *"what about Steel?"* resolves
against the previous turn instead of being treated as a fresh question.

**Prompt design** encodes the analysis rules that the model would otherwise get
wrong: exclude returns from spend totals, prefer complete rows for financial
figures and say so, and never answer a policy threshold from model memory —
always look it up. That last rule matters because the model has plausible-sounding
priors about procurement thresholds that have nothing to do with *this* company's
policy.

## 7. Evaluation

`evaluate.py` scores two dimensions across a golden set of 8 questions:

- **Factual accuracy** — ground truth is recomputed independently in Pandas from
  the cleaned data, so the eval never trusts the LLM to grade itself.
- **Tool routing** — did the agent call the tool the question required? A policy
  question answered without calling `search_policy_docs` is answering from model
  memory. That's a failure even if the number happens to be right, because it
  will not generalise to a policy the model has never seen.

Numeric matching is token-boundary aware so that an expected `21` does not
falsely match `215` in the answer text.

## 8. Known limitations

- Policy corpus is two short documents; a real deployment would index hundreds
  and need reranking plus metadata filtering by department and document version.
- No row-level access control. The governance policy restricts supplier pricing
  to Procurement and Finance; production would enforce this per-user, not just
  read-only at the connection level.
- Evaluation set is small (8 cases) and single-run. LLM outputs vary between
  runs, so a real eval would run each case several times and report variance.
- SQLite is a stand-in for an enterprise warehouse; the tool interface would
  be swapped for the real connector, not the agent logic.
- The model runs locally via Ollama rather than a large hosted model. This
  was a deliberate choice — no data ever leaves the machine, which suits the
  governance policy's access rules — but a small local model is measurably
  less reliable at tool selection than a frontier hosted model. The tool
  interface (`tools.py`, `agent.py`) is provider-agnostic: swapping in a
  hosted model later is a one-line change to `build_agent()`, not a rewrite.

## 9. Possible extensions

- Add a write-capable tool behind a human approval step (draft a purchase
  requisition, require sign-off before submission).
- Scheduled automation: run the compliance checks nightly and email exceptions.
- Add a reranking stage before generation for a larger policy corpus.
- Swap SQLite for the production warehouse and Chroma for a managed vector store.
