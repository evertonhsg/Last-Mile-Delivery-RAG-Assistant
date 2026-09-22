"""
Phase 3 — Qualitative side-by-side: base model vs. LoRA fine-tuned model.

Purpose: a gut-check before Phase 4's real RAGAS evaluation, NOT a scored
benchmark. This prints raw output for a human to read; it doesn't judge
anything automatically.

Both models are given IDENTICAL context for each question — retrieved once
via rerank_retrieve() (or read back out of an already-generated fine-tuning
example) and reused for both — so any difference in the two answers is
attributable to the LoRA adapter, not to retrieval variance.

Both models are also given the prompt in the EXACT shape the fine-tuned
model was trained on: RAG_PROMPT's system+human messages, merged into a
single user turn via merge_system_into_user() (same function
generate_finetune_data.py used to build the training set — reused here
rather than re-implemented, so there's no chance of the comparison silently
using a differently-shaped prompt than what was actually trained on).

Test set (same as the previous comparison run against lastmile-lora-earlystop,
plus one new out-of-scope question, per this round's instructions):
  1. The SAME 4 questions sampled last time from data/finetune/valid_v1.jsonl
     (the original, confident-answer-only validation set, preserved under
     that name when the refusal examples were merged in) — deliberately
     NOT data/finetune/valid.jsonl, which now contains the merged/reshuffled
     set with refusal examples mixed in and would sample a DIFFERENT 4
     examples even with the same seed. Reading from valid_v1.jsonl is what
     makes this an apples-to-apples rerun rather than a coincidentally
     similar one. Their original training-target answer is shown for
     reference, not as a "correct answer" to grade against.
  2. The recurring Phase 2 test question ("What is the SLA for Zone 1
     priority deliveries?"), retrieved FRESH via rerank_retrieve().
  3. The same out-of-scope question as last time (international customs
     clearance) — nothing in data/raw/ covers cross-border shipping,
     confirmed against live retrieval. This is the specific case we're
     checking got fixed: last run, the fine-tuned model refused but drifted
     off RAG_PROMPT's exact required wording.
  4. NEW: a second, harder out-of-scope question — delivery drones in Zone
     1. Harder because retrieval for it actually pulls Zone/SLA chunks
     (the question mentions "Zone 1," so it superficially looks on-topic),
     but none of those chunks say anything about drones — checked against
     live retrieval beforehand to confirm that.

Run from the repo root (imports sibling modules rag_chain.py and
generate_finetune_data.py):
    python src/compare_models.py
"""

import json
import random
from pathlib import Path

from mlx_lm import generate, load

from generate_finetune_data import merge_system_into_user
from rag_chain import RAG_PROMPT, format_context, rerank_retrieve

MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
ADAPTER_PATH = "adapters/lastmile-lora-v2-best"  # copy of the iter-340 checkpoint — see train_lora.py run notes
VALID_PATH = Path("data/finetune/valid_v1.jsonl")  # see module docstring — NOT valid.jsonl

N_VALID_QUESTIONS = 4
MAX_TOKENS = 300
SEED = 11

ZONE1_QUESTION = "What is the SLA for Zone 1 priority deliveries?"
OOD_QUESTION = "What is the process for international customs clearance on cross-border shipments?"
OOD_QUESTION_2 = "What is the company policy on using delivery drones in Zone 1?"


def build_prompt_messages(context: str, question: str) -> list[dict]:
    """Build the exact (merged, user-only) chat message the fine-tuned model
    was trained on, for a question retrieved fresh at comparison time."""
    prompt_value = RAG_PROMPT.invoke({"context": context, "question": question})
    role_map = {"system": "system", "human": "user"}
    raw_messages = [{"role": role_map[m.type], "content": m.content} for m in prompt_value.to_messages()]
    return merge_system_into_user(raw_messages)


def parse_valid_example(ex: dict) -> tuple[str, str, str]:
    """Pull (context, question, reference_answer) back out of an already-
    merged valid.jsonl example — for DISPLAY only. See module docstring for
    why generation itself reuses the stored message verbatim instead."""
    content = ex["messages"][0]["content"]
    before_question, question = content.rsplit("\n\nQuestion: ", 1)
    _, context = before_question.split("Context:\n", 1)
    reference_answer = ex["messages"][1]["content"]
    return context, question, reference_answer


def generate_answer(model, tokenizer, messages: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, verbose=False).strip()


def print_case(
    title: str,
    context: str,
    question: str,
    base_answer: str,
    tuned_answer: str,
    reference_answer: str | None,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"\n--- Context used (identical for both models) ---\n{context}")
    print(f"\n--- Question ---\n{question}")
    print(f"\n--- BASE model answer ---\n{base_answer}")
    print(f"\n--- FINE-TUNED model answer ---\n{tuned_answer}")
    if reference_answer is not None:
        print(f"\n--- Original training-target answer (reference only, not ground truth) ---\n{reference_answer}")


def main() -> None:
    print(f"Loading base model: {MODEL_ID}")
    base_model, tokenizer = load(MODEL_ID)

    print(f"Loading fine-tuned model: {MODEL_ID} + adapter {ADAPTER_PATH}")
    tuned_model, _ = load(MODEL_ID, adapter_path=ADAPTER_PATH)

    cases = []

    # 1. Held-out validation examples
    with open(VALID_PATH) as f:
        valid_examples = [json.loads(line) for line in f]
    random.seed(SEED)
    sampled = random.sample(valid_examples, min(N_VALID_QUESTIONS, len(valid_examples)))
    for i, ex in enumerate(sampled, start=1):
        context, question, reference_answer = parse_valid_example(ex)
        cases.append({
            "title": f"[Held-out validation example {i}/{len(sampled)}]",
            "context": context,
            "question": question,
            "messages": ex["messages"][:1],  # the exact stored user turn — no re-derivation
            "reference_answer": reference_answer,
        })

    # 2. Recurring Phase 2 test question, retrieved fresh
    zone1_context = format_context(rerank_retrieve(ZONE1_QUESTION, k=4))
    cases.append({
        "title": "[Phase 2 recurring test question]",
        "context": zone1_context,
        "question": ZONE1_QUESTION,
        "messages": build_prompt_messages(zone1_context, ZONE1_QUESTION),
        "reference_answer": None,
    })

    # 3. Out-of-scope question (same as last run)
    ood_context = format_context(rerank_retrieve(OOD_QUESTION, k=4))
    cases.append({
        "title": "[Out-of-scope refusal check — same question as last run]",
        "context": ood_context,
        "question": OOD_QUESTION,
        "messages": build_prompt_messages(ood_context, OOD_QUESTION),
        "reference_answer": None,
    })

    # 4. Second, harder out-of-scope question (new this round)
    ood2_context = format_context(rerank_retrieve(OOD_QUESTION_2, k=4))
    cases.append({
        "title": "[Out-of-scope refusal check #2 — new, harder: superficially on-topic retrieval]",
        "context": ood2_context,
        "question": OOD_QUESTION_2,
        "messages": build_prompt_messages(ood2_context, OOD_QUESTION_2),
        "reference_answer": None,
    })

    for case in cases:
        base_answer = generate_answer(base_model, tokenizer, case["messages"])
        tuned_answer = generate_answer(tuned_model, tokenizer, case["messages"])
        print_case(
            case["title"], case["context"], case["question"],
            base_answer, tuned_answer, case["reference_answer"],
        )


if __name__ == "__main__":
    main()
