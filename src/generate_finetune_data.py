"""
Phase 3 — Generate a LoRA fine-tuning dataset from the Phase 1 chunks, for
training with MLX-LM (Apple Silicon).

Why examples are shaped as (context, question) -> answer, not bare Q&A:

At inference time, src/rag_chain.py NEVER asks the model a bare question. It
always retrieves chunks first, wraps them in RAG_PROMPT's "Answer ONLY using
the information in the provided context" system message, and only then asks
the question. If we fine-tuned on bare question -> answer pairs instead, we'd
be training the model on a completely different input distribution than the
one it actually sees in production — it would learn to answer domain
questions from its own (memorized, possibly wrong) parametric knowledge
rather than learning to read and cite the context it's handed. Grounded-in,
grounded-out examples are what teach the model to actually attend to
{context} and stay faithful to it, which is the behavior this whole RAG
system depends on.

Concretely, this script builds each training example the same way a real
RAG turn is assembled — reusing format_context() and RAG_PROMPT directly
from rag_chain.py, rather than re-typing similar-looking strings here, so
the fine-tuning data can't silently drift out of sync with the actual
inference-time prompt.

Two quality gates, both driven by the same underlying guarantee: every
question here is generated FROM a specific chunk, so that chunk is
guaranteed BY CONSTRUCTION to answer it. Anything that breaks that
guarantee needs to be caught before it reaches the training set:

  1. Thin chunks (mostly a markdown header, little body text) don't give
     the question-generation step enough to work with, so it tends to fall
     back to a generic domain question that isn't really about that chunk.
     These are skipped up front, before any LLM calls are spent on them.

  2. Even with (1), a generated question can still drift off-chunk. When
     that happens, the by-construction guarantee is violated and the answer
     step (correctly, given only that chunk) hedges or refuses — e.g. a
     station_operations.md KPI-header chunk paired with a stray "What's the
     SLA for Zone 1?" question, answered "not stated in this context." That
     answer is technically correct for the (mismatched) pair, but it's
     POISONOUS training data: Zone 1's SLA *is* answered elsewhere in the
     knowledge base, so fine-tuning on this pair would teach the model to
     wrongly refuse a perfectly answerable question. Any answer matching
     refusal/hedge language is filtered out and the question is regenerated
     once and retried, rather than silently kept or silently dropped.

Data format: MLX-LM (checked against the installed mlx-lm 0.31.3 source,
mlx_lm/tuner/datasets.py) supports three JSONL schemas — "text": {"text"},
"completions": {"prompt", "completion"}, and "chat": {"messages": [...]}
(OpenAI chat-format). We use "chat".

IMPORTANT: no system role. mlx-community/Mistral-7B-Instruct-v0.3-4bit's
chat template (like most Mistral-family templates) only accepts
user/assistant turns — its Jinja template raises "Only user and assistant
roles are supported!" for anything else, which is exactly what mlx-lm's
ChatDataset would trip over via tokenizer.apply_chat_template() the moment
training started. So each example's RAG_PROMPT-derived system content (the
grounding rules + Context block) is folded into the START of the user
message instead (see merge_system_into_user() below) — same information the
model sees, just without a role its template can't handle.

Requires Ollama running locally with the "mistral" model pulled:
    ollama pull mistral

Run from the repo root (imports the sibling rag_chain.py module):
    python src/generate_finetune_data.py
"""

import json
import random
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_chain import RAG_PROMPT, format_context

# ── Paths ───────────────────────────────────────────────────────────────────
CHUNKS_PATH = Path("data/processed/chunks.json")
FINETUNE_DIR = Path("data/finetune")

# Roughly 100-150 examples total is the target for this small-scale learning
# project (55 chunks * 2-3 questions each) — not thousands. MIN/MAX_QUESTIONS
# bounds how many questions we ask the LLM to generate per chunk.
MIN_QUESTIONS_PER_CHUNK = 2
MAX_QUESTIONS_PER_CHUNK = 3

# Chunks with less body text than this (headers excluded) rarely give the
# question-generation step enough to write a genuinely grounded question —
# see the module docstring for why that matters.
MIN_BODY_WORDS = 40

VALID_FRACTION = 0.15
RANDOM_SEED = 42


# ── Thin-chunk filtering ─────────────────────────────────────────────────────
_HEADER_LINE_RE = re.compile(r"^\s*#{1,6}\s")


def count_body_words(text: str) -> int:
    """Count words in `text`, excluding markdown header lines.

    Headers ("## Zone Reclassification Policy") pad out a chunk's raw
    length without giving the question-generation step any actual facts to
    ask about, so they're excluded before judging whether a chunk has
    enough substance to generate a grounded question from.
    """
    body_lines = [line for line in text.splitlines() if not _HEADER_LINE_RE.match(line)]
    return len(re.findall(r"[A-Za-z0-9]+", " ".join(body_lines)))


# ── Refusal / hedge filtering ────────────────────────────────────────────────
# Broad on purpose: by construction, the chunk a question was generated from
# should always be able to answer it, so ANY hedge or refusal here is a
# signal of question drift (see module docstring), not a legitimate "out of
# scope" answer the way it would be at real inference time.
REFUSAL_PATTERNS = [
    "not explicitly stated",
    "does not provide",
    "doesn't provide",
    "does not contain",
    "doesn't contain",
    "does not include",
    "doesn't include",
    "does not mention",
    "doesn't mention",
    "not mentioned",
    "not specified",
    "not addressed",
    "not covered",
    "no information",
    "not enough information",
    "i don't have enough information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "refer to a different source",
    "outside the scope",
]


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(pattern in answer_lower for pattern in REFUSAL_PATTERNS)


# ── Question generation ──────────────────────────────────────────────────────
# temperature=0.7 here (vs. the 0 used everywhere else in this project) is
# deliberate: we WANT sampling variety across the 2-3 questions per chunk so
# the fine-tuning set contains a natural mix of phrasing rather than several
# near-identical questions the model would have produced deterministically.
QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write realistic questions that dispatchers, planners, and "
            "engineers would ask an internal Q&A assistant about last-mile "
            "delivery operations.\n"
            "Given an excerpt from an internal policy/operations document, "
            "write {n} different questions that the excerpt directly "
            "answers.\n"
            "Mix the phrasing: include at least one formally worded "
            "question (e.g. \"What is the SLA commitment for Zone 1 "
            "deliveries?\") and at least one casually worded question (e.g. "
            "\"How long does Zone 2 usually take?\").\n"
            "Return ONLY the questions, one per line, each ending in a "
            "question mark. No numbering, no bullets, no preamble, no "
            "commentary.",
        ),
        ("human", "{chunk_text}"),
    ]
)

# Strips leading list markers ("1.", "1)", "-", "*", "•") that models
# sometimes add back despite being told not to — kept permissive since a
# malformed line just gets filtered out below rather than crashing the run.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")


def generate_questions(llm: ChatOllama, chunk_text: str, n: int) -> list[str]:
    """Ask the LLM for `n` questions that `chunk_text` answers, and parse them
    out of its response into a clean list of question strings.

    Parsing is deliberately lenient (strip list markers, drop blank lines,
    keep only lines that actually look like a question) since we're relying
    on the model to follow a formatting instruction, not a strict schema.
    """
    messages = QUESTION_GEN_PROMPT.invoke({"chunk_text": chunk_text, "n": n})
    response = llm.invoke(messages)

    questions = []
    for line in response.content.splitlines():
        line = _LIST_MARKER_RE.sub("", line).strip().strip('"')
        if line and "?" in line:
            questions.append(line)

    return questions[:MAX_QUESTIONS_PER_CHUNK]


# ── Answer generation ────────────────────────────────────────────────────────
# Deliberately reuses RAG_PROMPT from rag_chain.py rather than writing a new
# prompt here. That prompt already IS "answer grounded only in the context,
# cite the source filename, professional tone" — and reusing it (instead of
# a look-alike copy) guarantees the fine-tuning examples match the exact
# system message the model will be run behind at inference time.
def generate_answer(llm: ChatOllama, context: str, question: str) -> str:
    messages = RAG_PROMPT.invoke({"context": context, "question": question})
    return llm.invoke(messages).content.strip()


# ── Example assembly ─────────────────────────────────────────────────────────
def merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Fold a leading {"role": "system"} message into the user message that
    follows it, dropping the system role entirely.

    Needed because the target model's chat template only accepts
    user/assistant turns (see module docstring). The system content becomes
    a preamble in front of the user's actual question, so the final user
    turn reads like "<grounding rules + Context:...>\\n\\nQuestion: <question>"
    — same information, same order, just one fewer (unsupported) role.
    """
    if not messages or messages[0]["role"] != "system":
        return messages

    system_content = messages[0]["content"]
    rest = messages[1:]

    if rest and rest[0]["role"] == "user":
        merged = f"{system_content}\n\nQuestion: {rest[0]['content']}"
        return [{"role": "user", "content": merged}] + rest[1:]

    return [{"role": "user", "content": system_content}] + rest


def build_chat_example(context: str, question: str, answer: str) -> dict:
    """Build one MLX-LM "chat" example: {"messages": [user, assistant]}.

    The user message is derived from RAG_PROMPT.invoke(...) itself (via
    .to_messages(), then merge_system_into_user()) rather than reconstructed
    by hand, so its wording is byte-for-byte what generate_answer() in
    rag_chain.py actually sends the model at inference time — just with the
    system content folded in rather than dropped.
    """
    prompt_value = RAG_PROMPT.invoke({"context": context, "question": question})

    role_map = {"system": "system", "human": "user"}
    raw_messages = [
        {"role": role_map[msg.type], "content": msg.content}
        for msg in prompt_value.to_messages()
    ]
    messages = merge_system_into_user(raw_messages)
    messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def print_example(index: int, example: dict) -> None:
    print(f"\n--- Example {index} " + "-" * 40)
    for message in example["messages"]:
        print(f"[{message['role']}]\n{message['content']}\n")


class ExampleResult:
    """Outcome of trying to build one training example from one question."""

    def __init__(self, example: dict | None, caught_pair: tuple[str, str] | None, retried: bool, retry_fixed: bool):
        self.example = example
        # (question, answer) that tripped the refusal filter, or None if it never did —
        # kept purely so main() can print a few caught examples for eyeballing.
        self.caught_pair = caught_pair
        self.retried = retried
        self.retry_fixed = retry_fixed


def make_example(question_llm, answer_llm, context: str, chunk_text: str, question: str) -> ExampleResult:
    """Generate an answer for `question` and check it for refusal/hedge
    language. On a hit, regenerate ONE new question from `chunk_text` and
    retry once rather than dropping the slot outright.

    `example` on the result is None if the pair was dropped (refusal on
    both the original attempt and the retry, or the retry produced no
    usable question).
    """
    answer = generate_answer(answer_llm, context, question)
    if not is_refusal(answer):
        return ExampleResult(build_chat_example(context, question, answer), None, False, False)

    caught_pair = (question, answer)

    # Refusal on the original question — regenerate a fresh one from the
    # same chunk and try exactly once more.
    retry_candidates = generate_questions(question_llm, chunk_text, n=1)
    if not retry_candidates:
        return ExampleResult(None, caught_pair, True, False)

    retry_question = retry_candidates[0]
    retry_answer = generate_answer(answer_llm, context, retry_question)
    if is_refusal(retry_answer):
        return ExampleResult(None, caught_pair, True, False)

    return ExampleResult(build_chat_example(context, retry_question, retry_answer), caught_pair, True, True)


def main() -> None:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    question_llm = ChatOllama(model="mistral", temperature=0.7)
    answer_llm = ChatOllama(model="mistral", temperature=0)

    examples = []
    caught_examples = []  # (question, answer) pairs the refusal filter caught, for eyeballing
    chunks_skipped = 0
    pairs_caught = 0
    pairs_retried = 0
    pairs_retry_fixed = 0

    for i, chunk in enumerate(chunks, start=1):
        word_count = count_body_words(chunk["text"])
        if word_count < MIN_BODY_WORDS:
            chunks_skipped += 1
            print(f"[{i}/{len(chunks)}] {chunk['source']} (chunk {chunk['chunk_index']}): "
                  f"skipped — only {word_count} body words (< {MIN_BODY_WORDS})")
            continue

        # Build the same context string format_context() would produce for
        # this chunk if it were the sole retrieved document — a
        # single-element Document list gives us that for free.
        doc = Document(page_content=chunk["text"], metadata={"source": chunk["source"]})
        context = format_context([doc])

        n_questions = random.randint(MIN_QUESTIONS_PER_CHUNK, MAX_QUESTIONS_PER_CHUNK)
        questions = generate_questions(question_llm, chunk["text"], n_questions)
        if len(questions) < MIN_QUESTIONS_PER_CHUNK:
            print(f"  [warning] chunk {chunk['id']} only yielded {len(questions)} usable question(s)")

        chunk_added = 0
        for question in questions:
            result = make_example(question_llm, answer_llm, context, chunk["text"], question)
            if result.caught_pair is not None:
                pairs_caught += 1
                if len(caught_examples) < 5:
                    caught_question, caught_answer = result.caught_pair
                    caught_examples.append((chunk["source"], caught_question, caught_answer))
            if result.retried:
                pairs_retried += 1
            if result.retry_fixed:
                pairs_retry_fixed += 1
            if result.example is not None:
                examples.append(result.example)
                chunk_added += 1

        print(f"[{i}/{len(chunks)}] {chunk['source']} (chunk {chunk['chunk_index']}): "
              f"+{chunk_added} examples — {len(examples)} total")

    # ── Split and write ──────────────────────────────────────────────────────
    random.Random(RANDOM_SEED).shuffle(examples)
    split_at = int(len(examples) * (1 - VALID_FRACTION))
    train_examples, valid_examples = examples[:split_at], examples[split_at:]

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_examples), ("valid", valid_examples)]:
        path = FINETUNE_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for example in split:
                f.write(json.dumps(example) + "\n")
        print(f"Wrote {len(split)} examples to {path}")

    print(f"\nChunks skipped (too thin): {chunks_skipped}")
    print(f"Pairs caught by refusal filter: {pairs_caught}")
    print(f"Pairs retried: {pairs_retried} (fixed by retry: {pairs_retry_fixed}, "
          f"dropped after retry still failed: {pairs_retried - pairs_retry_fixed})")
    print(f"Total: {len(examples)} examples ({len(train_examples)} train / {len(valid_examples)} valid)")

    # ── Show what the refusal filter actually caught ────────────────────────
    if caught_examples:
        print("\n" + "=" * 60)
        print(f"Examples the refusal filter caught (showing {len(caught_examples)}):")
        for source, question, answer in caught_examples:
            print(f"\n  source: {source}")
            print(f"  question: {question}")
            print(f"  answer: {answer}")

    # ── Eyeball a few full examples ──────────────────────────────────────────
    sample = random.sample(examples, min(5, len(examples)))
    print("\n" + "=" * 60)
    print("Sample examples (for eyeballing format/quality):")
    for i, example in enumerate(sample, start=1):
        print_example(i, example)


if __name__ == "__main__":
    main()
