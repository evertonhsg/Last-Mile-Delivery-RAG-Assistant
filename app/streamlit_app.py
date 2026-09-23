"""
Phase 5 — Streamlit chat interface for the Last-Mile Delivery RAG Assistant.

Pure UI layer: every retrieval/generation call goes straight through
generate_answer() from src/rag_chain.py, unchanged from the REPL — this file
adds no new RAG logic of its own, only a chat surface on top of an already
built and tested pipeline (Phase 2's retrieval/generation, Phase 3's
fine-tuned backend, both validated end-to-end before this file existed).

Chat history is kept as the exact list[tuple[str, str]] shape
generate_answer() already expects (see rag_chain.contextualize_question()),
derived on the fly from st.session_state.turns rather than duplicated in a
second, parallel format — so multi-turn follow-ups ("what about zone 2?")
get resolved by contextualize_question() exactly as they do in the REPL,
not by some UI-specific re-implementation of the same idea.

Model/embedding/adapter loading is NOT re-cached here with
st.cache_resource: rag_chain.py already lazily caches the cross-encoder and
the MLX model+adapter at module level (see _get_cross_encoder(),
_get_mlx_model()). Streamlit reruns this script top-to-bottom on every
interaction, but `import rag_chain` only actually executes the module body
once per server process — Python's own import cache — so those module-level
singletons persist across reruns for free, the same way they persist across
turns of the REPL's while-loop. Adding a second caching layer on top would
just be redundant.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# app/ is a sibling of src/, not inside it (unlike every other script in this
# project, which lives in src/ itself and can import siblings directly) —
# resolved relative to this file rather than the CWD, so this works
# regardless of what directory `streamlit run` happens to be invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from rag_chain import generate_answer

st.set_page_config(page_title="Last-Mile Delivery RAG Assistant", page_icon="🚚")

st.title("🚚 Last-Mile Delivery RAG Assistant")
st.caption("Ask about zones, SLAs, routing rules, capacity planning, or station operations.")

with st.sidebar:
    st.header("Settings")
    use_finetuned = st.toggle(
        "Use fine-tuned model",
        value=False,
        help=(
            "Off: base Mistral-7B via Ollama (Phase 2). "
            "On: the Phase 3 LoRA adapter, served locally via MLX "
            "(adapters/lastmile-lora-v2-best)."
        ),
    )
    backend = "mlx-finetuned" if use_finetuned else "ollama"
    st.caption(f"Active backend: `{backend}`")

# st.session_state.turns is the single source of truth for the conversation:
# each entry holds everything one turn produced (question, answer, sources,
# standalone_question, backend), both for re-rendering past turns on every
# rerun and for deriving the chat_history tuples generate_answer() expects.
if "turns" not in st.session_state:
    st.session_state.turns = []


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        # The rewritten standalone question is only meaningfully different
        # from the raw question on follow-ups ("what about zone 2?") — a
        # small always-visible caption, not tucked inside a closed
        # expander, so it's there to debug a follow-up without hiding it.
        if turn["standalone_question"] != turn["question"]:
            st.caption(f"🔍 Retrieved using: _{turn['standalone_question']}_")
        with st.expander("Sources"):
            if turn["sources"]:
                for source in turn["sources"]:
                    st.markdown(f"- `{source}`")
            else:
                st.markdown("_No sources retrieved._")
        st.caption(f"Answered by: `{turn['backend']}`")


for turn in st.session_state.turns:
    render_turn(turn)

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # MLX inference in particular is not instant (a full local forward
        # pass per generated token) — a spinner makes that latency visible
        # rather than leaving the UI looking hung.
        with st.spinner(f"Generating via {backend}..."):
            chat_history = [(t["question"], t["answer"]) for t in st.session_state.turns]
            result = generate_answer(prompt, chat_history=chat_history, backend=backend)

        st.write(result["answer"])
        if result["standalone_question"] != prompt:
            st.caption(f"🔍 Retrieved using: _{result['standalone_question']}_")
        with st.expander("Sources"):
            if result["sources"]:
                for source in result["sources"]:
                    st.markdown(f"- `{source}`")
            else:
                st.markdown("_No sources retrieved._")
        st.caption(f"Answered by: `{backend}`")

    st.session_state.turns.append({
        "question": prompt,
        "answer": result["answer"],
        "sources": result["sources"],
        "standalone_question": result["standalone_question"],
        "backend": backend,
    })
