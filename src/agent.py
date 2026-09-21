"""
agent.py
--------
The agentic core: an LLM that plans, calls tools, reads the results, and
decides whether it needs another tool call before answering.

Three things here map directly to the JD's "agentic workflows, tool calling,
memory and multi-step reasoning":

  * Tool calling     - the four tools in tools.py are bound to the model.
  * Multi-step       - the agent loops: it can query the DB, notice the answer
                       depends on a policy threshold, search the policy docs,
                       and only then answer. No fixed chain.
  * Memory           - a checkpointer persists conversation state per thread_id,
                       so follow-up questions ('what about Steel?') resolve
                       against the earlier turn.

Run a one-off question:
    python src/agent.py "Which business unit has the highest spend?"
"""

import os
import sys
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import ALL_TOOLS  # noqa: E402

load_dotenv()

# Runs entirely locally via Ollama — no API key, no cost, no rate limit.
# llama3.2 is small enough to run on a laptop CPU and supports tool calling.
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def ollama_is_running() -> bool:
    """Check the local Ollama server is up before we try to use it."""
    try:
        requests.get(OLLAMA_HOST, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def model_is_pulled(model: str) -> bool:
    """Check the requested model has actually been downloaded."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        # Ollama tags include a version suffix like ':latest' — match the base name too
        return any(n == model or n.startswith(f"{model}:") for n in names)
    except requests.exceptions.RequestException:
        return False

SYSTEM_PROMPT = """You are an operations analyst assistant for a large manufacturing
and trading conglomerate. You answer questions about procurement data and company policy.

How to work:
- Use `query_operations_db` for factual questions about orders, spend, suppliers,
  lead times and statuses. Write precise SQL; check the schema in the tool description.
- Use `compute_statistics` when the question is about averages, spread or distribution.
- Use `search_policy_docs` whenever the question touches rules, thresholds, approvals
  or compliance. Never guess a policy threshold from memory — look it up.
- Use `check_data_quality` when asked whether the data is reliable, or when a result
  looks surprising and might be a data artifact.

Important analysis rules:
- Exclude returns (is_return = 1) from spend totals unless asked otherwise.
- Prefer rows where is_complete = 1 for financial figures, and say so when you do,
  since incomplete rows would understate totals.
- If a question requires both data and policy (e.g. "are we compliant with lead time
  standards?"), call both tools and reason over the combination. Do not answer from
  only one source.
- Flag data quality caveats when they materially affect the answer.
- Report monetary values in BDT with thousands separators.

Be concise. Show the key numbers. If a tool returns an error, read it, correct your
input, and retry rather than giving up.
"""


def build_agent(thread_memory: bool = True):
    """Construct the tool-calling agent. Returns a compiled LangGraph agent."""
    model = ChatOllama(model=CHAT_MODEL, temperature=0, base_url=OLLAMA_HOST)
    checkpointer = InMemorySaver() if thread_memory else None
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def ask(agent, question: str, thread_id: str = "default") -> dict:
    """
    Send one question to the agent and return a dict with the final answer
    plus the tool calls it made along the way (useful for demos and evals —
    showing the reasoning trace is far more convincing than the answer alone).
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)

    messages = result["messages"]
    final = messages[-1].content

    tool_calls = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_calls.append({"tool": call["name"], "args": call["args"]})

    return {"answer": final, "tool_calls": tool_calls, "messages": messages}


def _cli(argv: Iterable[str]) -> None:
    args = list(argv)
    if not args:
        question = "Which business unit has the highest total spend, and is its average lead time within policy?"
        print(f"(no question given, using demo question)\n> {question}\n")
    else:
        question = " ".join(args)

    if not ollama_is_running():
        print(
            "ERROR: Ollama isn't running. Start it with `ollama serve` "
            "(or open the Ollama app), then try again."
        )
        sys.exit(1)
    if not model_is_pulled(CHAT_MODEL):
        print(f"ERROR: model '{CHAT_MODEL}' not found locally. Run: ollama pull {CHAT_MODEL}")
        sys.exit(1)

    agent = build_agent()
    result = ask(agent, question)

    print("=== Tool calls ===")
    for i, call in enumerate(result["tool_calls"], start=1):
        print(f"  {i}. {call['tool']}({call['args']})")
    print("\n=== Answer ===")
    print(result["answer"])


if __name__ == "__main__":
    _cli(sys.argv[1:])
