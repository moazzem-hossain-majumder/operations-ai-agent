"""
app.py
------
Streamlit chat interface for the operations agent.

Deliberately shows the agent's tool-call trace alongside each answer. In a
business setting nobody trusts an unexplained number from an LLM — showing
"I ran this SQL, and looked up this policy clause" is what makes it usable.

Run:  streamlit run src/app.py
"""

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import build_agent, ask, ollama_is_running, model_is_pulled, CHAT_MODEL  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "operations.db"

st.set_page_config(page_title="Operations AI Agent", page_icon="🤖", layout="centered")

st.title("🤖 Operations AI Agent")
st.caption(
    "Ask questions about procurement data and company policy. "
    "Runs entirely locally via Ollama — no API key, no cost."
)

# --- Preconditions --------------------------------------------------------
if not ollama_is_running():
    st.error(
        "Ollama isn't running. Start it with `ollama serve` (or open the Ollama app) "
        "and refresh this page.",
        icon="🦙",
    )
    st.stop()

if not model_is_pulled(CHAT_MODEL):
    st.error(
        f"Model `{CHAT_MODEL}` isn't downloaded yet. Run `ollama pull {CHAT_MODEL}` "
        "in a terminal, then refresh.",
        icon="📥",
    )
    st.stop()

if not DB_PATH.exists():
    st.error(
        "Database not built yet. Run:\n\n"
        "```\npython src/make_sample_data.py\npython src/data_prep.py\n```",
        icon="🗄️",
    )
    st.stop()

# --- Session state --------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

with st.sidebar:
    st.subheader("Try asking")
    st.markdown(
        "- Which business unit has the highest spend?\n"
        "- What's the median lead time for Cement, and does it meet policy?\n"
        "- Which suppliers have the most cancelled orders?\n"
        "- How reliable is this data?\n"
        "- What approval does a BDT 2,000,000 order need?"
    )
    st.divider()
    st.caption(
        "Memory is per-session: follow-ups like *'what about Steel?'* "
        "resolve against the previous turn."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# --- Replay history -------------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("tool_calls"):
            with st.expander(f"Tool calls ({len(turn['tool_calls'])})"):
                for i, call in enumerate(turn["tool_calls"], start=1):
                    st.markdown(f"**{i}. `{call['tool']}`**")
                    st.code(
                        "\n".join(f"{k}: {v}" for k, v in call["args"].items()) or "(no args)",
                        language="text",
                    )

# --- Input ----------------------------------------------------------------
prompt = st.chat_input("Ask about procurement data or policy...")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Planning, calling tools, and reasoning..."):
            try:
                result = ask(
                    st.session_state.agent,
                    prompt,
                    thread_id=st.session_state.thread_id,
                )
                answer = result["answer"]
                tool_calls = result["tool_calls"]
            except Exception as e:
                answer = f"Something went wrong: {e}"
                tool_calls = []

        st.markdown(answer)
        if tool_calls:
            with st.expander(f"Tool calls ({len(tool_calls)})"):
                for i, call in enumerate(tool_calls, start=1):
                    st.markdown(f"**{i}. `{call['tool']}`**")
                    st.code(
                        "\n".join(f"{k}: {v}" for k, v in call["args"].items()) or "(no args)",
                        language="text",
                    )

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "tool_calls": tool_calls}
    )
