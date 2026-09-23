"""
Phase 4 Step 1 — Build a held-out RAGAS evaluation set.

This set is deliberately isolated from everything Phase 3 touched:
  - Built directly from data/raw/*.md, NOT from data/processed/chunks.json
    (Phase 1's chunking output that generate_finetune_data.py worked from).
  - Never passed through generate_finetune_data.py or
    generate_refusal_examples.py, and doesn't overlap with either script's
    output or topic list (see the refusal questions below).
  - Not sampled from data/finetune/*.jsonl, which the LoRA adapter was
    trained AND checkpoint-selected against.
None of this was seen by training or by checkpoint selection — that's what
makes a RAGAS score against it meaningful, rather than a circular "does the
model do well on data shaped like what it trained on."

Question design is DELIBERATE, not freely LLM-generated (unlike
generate_finetune_data.py, where letting the LLM pick both question and
chunk was fine — here we specifically need controlled coverage of three
different retrieval difficulty levels, which random sampling wouldn't
guarantee):
  - 15 single-document questions (3 per source doc). Each doc gets a mix of
    a direct single-fact lookup and a question that needs 2+ facts
    synthesized from different SECTIONS of that same document — RAGAS's
    context-precision/recall metrics behave differently for each, so a
    real eval set needs both represented, not just the easy case.
  - 3 cross-document questions whose full answer requires combining facts
    from two different source docs — no single chunk, or even single
    document, contains the whole answer. This is what actually exercises
    MMR-style diversity-aware retrieval rather than plain top-k similarity.
  - 3 out-of-scope refusal questions, on topics that do NOT overlap with
    generate_refusal_examples.py's 18-topic training list, or the "delivery
    drones" question compare_models.py already probed. The point is testing
    whether the refusal fix generalizes to genuinely novel off-topic
    questions, not re-confirming what it was directly trained on.

Only single-doc/cross-doc reference answers are LLM-generated (via
ChatOllama, temperature=0), and they're grounded in the FULL raw document
text, not the RAFT context-chunk format training used — these are meant to
read like "what a knowledgeable human would say" after reading the actual
policy doc, not a chunk-citation-styled RAG answer. Refusal questions get a
fixed reference answer instead (reused from generate_refusal_examples.py's
get_fallback_phrase(), not retyped): there's no real content to ground an
LLM answer in by definition, and generating one via LLM would risk exactly
the wording drift Phase 3's refusal fix was built to prevent.

Run from the repo root (imports sibling module generate_refusal_examples.py):
    python src/generate_eval_set.py
"""

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from generate_refusal_examples import get_fallback_phrase

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/eval/eval_set.json")

# (question, category, source_docs) — source_docs are filenames under
# data/raw/; [] for refusal questions, since nothing in the KB is the source.
EVAL_QUESTIONS = [
    # ── zone_definitions.md ──────────────────────────────────────────────
    ("What is the maximum number of stops per route for Zone 2?",
     "single-doc", ["zone_definitions.md"]),
    ("What is the primary hub for Zone 3, and what is its secondary hub?",
     "single-doc", ["zone_definitions.md"]),
    ("Under what conditions would a zone be reclassified to a lower zone number, "
     "and what does that mean for its allowed vendor fleet share?",
     "single-doc", ["zone_definitions.md"]),

    # ── capacity_planning.md ─────────────────────────────────────────────
    ("What percentage weight does the XGBoost model carry in Meridian's demand "
     "forecasting ensemble?",
     "single-doc", ["capacity_planning.md"]),
    ("What is the minimum on-time delivery rate vendors must maintain over a "
     "28-day period?",
     "single-doc", ["capacity_planning.md"]),
    ("What actions does Meridian take in October to prepare for peak season, and "
     "what specific thresholds trigger additional surge capacity once peak begins?",
     "single-doc", ["capacity_planning.md"]),

    # ── routing_rules.md ─────────────────────────────────────────────────
    ("What is the maximum daily driving time allowed for a driver under EU "
     "Regulation 561/2006?",
     "single-doc", ["routing_rules.md"]),
    ("What are the default weights for the VRPTW objective function's four cost "
     "components?",
     "single-doc", ["routing_rules.md"]),
    ("How does the VRPTW objective function's weighting change during peak "
     "season, and what route type is more commonly used in Zone 2 during that "
     "same period?",
     "single-doc", ["routing_rules.md"]),

    # ── sla_policies.md ──────────────────────────────────────────────────
    ("What is the delivery commitment time for the Priority Overnight service "
     "tier?",
     "single-doc", ["sla_policies.md"]),
    ("How many delivery attempts does Meridian make before a parcel is held for "
     "pickup?",
     "single-doc", ["sla_policies.md"]),
    ("What compensation is owed for a Priority Overnight SLA breach, and how does "
     "that compare to the compensation for a Standard tier breach?",
     "single-doc", ["sla_policies.md"]),

    # ── station_operations.md ────────────────────────────────────────────
    ("What is the minimum staffing level required per shift at an Urban Hub "
     "Station?",
     "single-doc", ["station_operations.md"]),
    ("How long is vehicle telematics data retained for compliance purposes?",
     "single-doc", ["station_operations.md"]),
    ("What happens to an undelivered parcel at end of day depending on whether "
     "it has remaining delivery attempts, and what event is logged when parcels "
     "are first unloaded each morning?",
     "single-doc", ["station_operations.md"]),

    # ── cross-document ───────────────────────────────────────────────────
    ("A Zone 1 Priority Overnight parcel breaches its SLA — what compensation is "
     "the customer owed, and what fleet types typically deliver in that zone?",
     "cross-doc", ["sla_policies.md", "zone_definitions.md"]),
    ("What is Zone 4's fleet buffer rate under normal capacity planning, and "
     "what is the maximum load of the HGV vehicles assigned to Zone 4 routes?",
     "cross-doc", ["capacity_planning.md", "routing_rules.md"]),
    ("If a Priority Overnight parcel is added to a route after the driver has "
     "already been dispatched, what happens to it, and what is the dispatch "
     "cut-off time for Priority routes at an Urban Hub Station?",
     "cross-doc", ["routing_rules.md", "station_operations.md"]),

    # ── refusal (new topics — not in generate_refusal_examples.py's list) ──
    ("What is Meridian Logistics' policy on carbon offset purchases for its "
     "delivery fleet?",
     "refusal", []),
    ("What retirement or pension benefits does Meridian Logistics offer its "
     "corporate employees?",
     "refusal", []),
    ("Where is Meridian Logistics' corporate headquarters located?",
     "refusal", []),
]

EVAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable Meridian Logistics operations analyst. "
            "Given the internal documentation below, answer the question "
            "thoroughly and accurately, the way you'd explain it to a "
            "colleague who needs the specifics — precise numbers, "
            "thresholds, and names, not a vague summary. Write in clear "
            "prose (a sentence or two; a short list only if the question "
            "genuinely calls for one). Use ONLY the information in the "
            "documentation below — do not add anything it doesn't state.\n\n"
            "Documentation:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def load_doc(filename: str) -> str:
    return (RAW_DIR / filename).read_text(encoding="utf-8")


def build_context(source_docs: list[str]) -> str:
    """Full raw document text, not pre-chunked RAFT-style context — see the
    module docstring for why: these reference answers are meant to read
    like a knowledgeable human's complete answer, not a chunk-citation-
    styled RAG response."""
    blocks = [f"=== {doc} ===\n{load_doc(doc)}" for doc in source_docs]
    return "\n\n".join(blocks)


def generate_reference_answer(llm: ChatOllama, question: str, source_docs: list[str]) -> str:
    context = build_context(source_docs)
    messages = EVAL_ANSWER_PROMPT.invoke({"context": context, "question": question})
    return llm.invoke(messages).content.strip()


def main() -> None:
    llm = ChatOllama(model="mistral", temperature=0)
    fallback = get_fallback_phrase()

    eval_set = []
    for question, category, source_docs in EVAL_QUESTIONS:
        if category == "refusal":
            # No real content to ground an LLM answer in by definition —
            # and generating one via LLM risks the same wording drift
            # Phase 3's refusal fix exists to prevent. The fixed fallback
            # phrase IS the correct ground truth here, not an LLM's guess
            # at a plausible-sounding refusal.
            reference_answer = fallback
        else:
            reference_answer = generate_reference_answer(llm, question, source_docs)

        eval_set.append({
            "question": question,
            "reference_answer": reference_answer,
            "category": category,
            "source_docs": source_docs,
        })
        print(f"[{len(eval_set)}/{len(EVAL_QUESTIONS)}] ({category}) {question}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)
    print(f"\nWrote {len(eval_set)} examples to {OUTPUT_PATH}")

    # ── Print the full set for review — every example, not a sample ────────
    print("\n" + "=" * 70)
    print("Full eval set (for review):")
    for i, ex in enumerate(eval_set, start=1):
        print(f"\n--- {i}/{len(eval_set)} [{ex['category']}] source_docs={ex['source_docs']} ---")
        print(f"Q: {ex['question']}")
        print(f"A: {ex['reference_answer']}")


if __name__ == "__main__":
    main()
