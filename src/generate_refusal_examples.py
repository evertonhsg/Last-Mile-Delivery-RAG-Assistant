"""
Phase 3 — Deliberately-constructed refusal examples, fixing a gap left by
generate_finetune_data.py's Gate 2 refusal filter.

Gate 2 correctly removed every ACCIDENTAL refusal from the training set
(question drifted off its source chunk, model correctly hedged for the
wrong reason — see that module's docstring). But the side effect: the 94
examples that survived are ALL confident, fully-answered, source-cited
examples. The model never saw a single correctly-labeled "this genuinely
isn't in the KB" example during training. Comparing the fine-tuned model
against the base model on an out-of-scope question (compare_models.py, run
against adapters/lastmile-lora-earlystop) showed exactly the predicted
symptom: the fine-tuned model still refused, but drifted off RAG_PROMPT's
exact required wording ("The provided context does not contain
information...") instead of using it verbatim, because nothing in training
ever reinforced that specific string.

This script is the fix: for each of ~18 genuinely out-of-scope topics (not
covered by any of the 5 source docs), pair a natural-sounding question with
a RANDOMLY CHOSEN, deliberately mismatched chunk — the exact same
mechanism as the accidental drift Gate 2 caught, just intentional this
time — and label it with the literal required fallback string, not a
paraphrase and not something LLM-generated. Consistency of that EXACT
wording is the entire point, so:

  - The fallback phrase itself is pulled straight out of RAG_PROMPT via a
    throwaway .invoke() + regex (see get_fallback_phrase()), not retyped by
    hand — so it can never silently drift out of sync with the wording the
    model is actually instructed to use at inference time.
  - Each example is built via generate_finetune_data.build_chat_example(),
    the exact same function (not a look-alike) used for every other
    training example, so the format is identical down to the byte.

Run from the repo root (imports sibling modules):
    python src/generate_refusal_examples.py
"""

import json
import re
from pathlib import Path
from random import Random

from langchain_core.documents import Document

from generate_finetune_data import build_chat_example
from rag_chain import RAG_PROMPT, format_context

CHUNKS_PATH = Path("data/processed/chunks.json")
OUTPUT_PATH = Path("data/finetune/refusal_examples.json")
MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
RANDOM_SEED = 7

# (short topic phrase, natural-sounding question) — genuinely NOT covered by
# any of the 5 source docs (capacity_planning.md, routing_rules.md,
# sla_policies.md, station_operations.md, zone_definitions.md). The topic
# phrase doubles as the reason clause appended after the literal fallback
# sentence (see build_refusal_examples()).
OUT_OF_SCOPE_TOPICS = [
    ("customs clearance for international shipments",
     "What is the process for international customs clearance on cross-border shipments?"),
    ("driver certification and licensing requirements",
     "What certifications or licenses are required for drivers to operate company vehicles?"),
    ("fuel surcharge policy",
     "How is the fuel surcharge calculated and when is it applied to a shipment?"),
    ("warehouse safety incident reporting",
     "What is the procedure for reporting a safety incident at a warehouse?"),
    ("third-party carrier insurance requirements",
     "What insurance coverage is required from third-party carriers we contract with?"),
    ("vehicle leasing terms",
     "What are the terms and conditions of our vehicle leasing agreements?"),
    ("union labor agreement terms",
     "What does the union labor agreement specify about driver overtime pay?"),
    ("employee expense reimbursement policy",
     "What is the policy for reimbursing employee travel expenses?"),
    ("data privacy and GDPR compliance",
     "How do we ensure GDPR compliance when handling customer delivery data?"),
    ("marketing budget allocation",
     "What is the marketing budget allocation for this quarter?"),
    ("warehouse lease renewal terms",
     "What are the renewal terms for our warehouse leases?"),
    ("cybersecurity incident response plan",
     "What is our cybersecurity incident response plan?"),
    ("driver recruitment and hiring process",
     "What is the hiring process for new delivery drivers?"),
    ("employee holiday and PTO policy",
     "How many paid time off days do employees receive per year?"),
    ("packaging material sustainability certification",
     "What sustainability certifications does our packaging material hold?"),
    ("EV charging infrastructure investment plan",
     "What is the investment plan for electric vehicle charging infrastructure?"),
    ("tax withholding for contractor drivers",
     "What are the tax withholding requirements for contractor drivers?"),
    ("supplier payment terms",
     "What are the standard payment terms for our suppliers?"),
]


def get_fallback_phrase() -> str:
    """Pull the exact required refusal string out of RAG_PROMPT itself,
    rather than retyping it, so this can't silently drift out of sync."""
    rendered = RAG_PROMPT.invoke({"context": "x", "question": "y"}).to_messages()[0].content
    match = re.search(r'respond exactly with: "([^"]+)"', rendered)
    if not match:
        raise ValueError("Could not find the fallback phrase in RAG_PROMPT — did its wording change?")
    return match.group(1)


def build_refusal_examples() -> list[dict]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    fallback = get_fallback_phrase()
    rng = Random(RANDOM_SEED)

    examples = []
    for topic, question in OUT_OF_SCOPE_TOPICS:
        # Deliberately mismatched: a random chunk that has nothing to do
        # with `question` — same mechanism as the accidental drift Gate 2
        # caught in generate_finetune_data.py, just on purpose this time.
        chunk = rng.choice(chunks)
        doc = Document(page_content=chunk["text"], metadata={"source": chunk["source"]})
        context = format_context([doc])

        # Verbatim fallback sentence + a short, deterministic (not
        # LLM-generated) grounded-reason clause, matching the style already
        # seen in real caught examples (e.g. "...to answer that. The
        # context provided does not contain information about ...").
        answer = f"{fallback} The provided context does not contain information about {topic}."

        examples.append(build_chat_example(context, question, answer))

    return examples


def main() -> None:
    examples = build_refusal_examples()
    print(f"Generated {len(examples)} deliberate refusal examples")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")

    # ── Validate against the real tokenizer, same as generate_finetune_data.py's
    # post-fix check — prove none of these raise, don't assume.
    from mlx_lm.utils import load_tokenizer

    print(f"\nValidating against {MODEL_ID}'s tokenizer...")
    tokenizer = load_tokenizer(MODEL_ID)

    failures = 0
    for i, ex in enumerate(examples):
        try:
            tokenizer.apply_chat_template(ex["messages"], add_generation_prompt=False)
        except Exception as e:
            failures += 1
            print(f"  [{i}] FAILED: {type(e).__name__}: {e}")
    print(f"{len(examples) - failures}/{len(examples)} examples pass apply_chat_template()")

    print("\nSample example:")
    print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    main()
