"""
Phase 4 Step 2 — RAGAS evaluation harness: score both generate_answer()
backends ("ollama" and "mlx-finetuned") against the held-out
data/eval/eval_set.json (~20 questions, never touched by Phase 3 training or
checkpoint selection — see generate_eval_set.py).

Surprises found wiring this up, checked against the ACTUALLY INSTALLED
ragas==0.4.3 (not memory — its API has moved a lot since the 0.1.x/0.2.x
docs most existing tutorials describe):

  - Dataset schema is a class, not a plain dict-of-lists fed into a
    HuggingFace Dataset. ragas.SingleTurnSample's fields are user_input,
    retrieved_contexts, response, reference — NOT question/answer/contexts/
    ground_truth(s), which is what pre-0.2 docs show. A list of
    SingleTurnSamples goes into ragas.EvaluationDataset(samples=...).
    Confirmed directly from each metric's `_required_columns` attribute,
    not assumed.

  - ragas.metrics.faithfulness / .answer_relevancy / .context_precision /
    .context_recall — the flat top-level singleton instances the classic
    docs use — still work in 0.4.3, but importing them now raises a
    DeprecationWarning pointing at a new ragas.metrics.collections module
    ("will be removed in v1.0"). That collections module is a genuinely
    different, class-based architecture (e.g. Faithfulness(llm=...)
    instantiated directly, async-first) — not a drop-in swap. Since ragas
    is still pre-1.0 and the classic path is what evaluate() documents and
    what actually got verified working here, this file uses the classic
    (deprecated-but-functional) metrics and just suppresses that one
    specific warning, rather than adopting a still-settling newer API for a
    small local eval harness.

  - ragas.llms.LangchainLLMWrapper / ragas.embeddings.LangchainEmbeddingsWrapper
    (the exact wrapper names this task asked for) are ALSO now deprecation-
    shim proxies at the top level — calling them works (they forward to the
    real class and warn once via the shim). The real classes live at
    ragas.llms.base.LangchainLLMWrapper / ragas.embeddings.base.LangchainEmbeddingsWrapper,
    imported from there below to skip that shim layer — but the warning
    ITSELF turned out to be baked into the real class's own __init__, not
    just the top-level shim (confirmed by reading its source), so it fires
    either way. Importing from .base avoids a redundant SECOND warning
    (shim + real class both warning), not the warning entirely; the
    warnings.catch_warnings() blocks below are what actually silence it.
    Worth noting: ragas's own suggested replacement (llm_factory/
    embedding_factory) is oriented at hosted APIs (OpenAI, etc.) with no
    first-class "local Ollama" option — exactly why wrapping ChatOllama via
    the Langchain wrapper, as asked, remains the right call for a fully
    local, no-API-key setup.

  - `pip show ragas` lists the project home page as
    github.com/vibrantlabsai/ragas, not the explodinggradients/ragas most
    existing docs/tutorials reference — same package/PyPI name, but worth
    knowing if you go looking for current docs.

  - RunConfig's default max_workers is 16 — fine against a hosted API, but
    16 concurrent judge calls hammering a single local Ollama process on a
    laptop is a recipe for timeouts/contention, not speed. Dialed down to
    2 below (see get_run_config()).

Run from the repo root (imports sibling module rag_chain.py):
    python src/run_eval.py --dry-run   # 1 question, both backends — sanity check
    python src/run_eval.py             # full ~20-question x 2-backend run
"""

import argparse
import json
import math
import warnings
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from rag_chain import EMBEDDING_MODEL_NAME, generate_answer

EVAL_SET_PATH = Path("data/eval/eval_set.json")
RESULTS_DIR = Path("data/eval/results")

BACKENDS = ["ollama", "mlx-finetuned"]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_judge_llm_and_embeddings():
    """The judge model/embeddings for ragas's metrics — deliberately the
    SAME local models the rest of this project uses (ChatOllama/mistral,
    all-MiniLM-L6-v2), wrapped via ragas's Langchain wrappers, so scoring
    needs no API key and no network call. See module docstring for why
    these specific import paths were chosen over the top-level ones, and
    why the DeprecationWarning both wrappers raise on construction (baked
    into the real classes themselves, not just the deprecated top-level
    re-export) is expected and suppressed here rather than left to print.
    """
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        judge_llm = LangchainLLMWrapper(ChatOllama(model="mistral", temperature=0))
        judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    return judge_llm, judge_embeddings


def get_run_config():
    """See module docstring — default max_workers=16 would fire that many
    concurrent judge calls at one local Ollama process. 2 keeps some
    overlap (embedding calls run separately from LLM calls anyway) without
    swamping it."""
    from ragas.run_config import RunConfig

    return RunConfig(max_workers=2)


def build_ragas_samples(eval_set: list[dict], backend: str) -> tuple[list, list[dict]]:
    """Run generate_answer() for every eval question against `backend`, and
    build one ragas SingleTurnSample per question. Also returns the raw
    per-question records (question/category/generated answer/etc.), since
    EvaluationResult itself doesn't carry "category" — that join back onto
    ragas's scores has to happen on our side (see merge_scores()).
    """
    from ragas import SingleTurnSample  # deferred — see module docstring

    samples = []
    records = []
    for i, ex in enumerate(eval_set, start=1):
        print(f"  [{i}/{len(eval_set)}] ({backend}) {ex['question'][:70]}")
        # Fresh single-turn call each time — chat_history=[] deliberately.
        # This eval measures retrieval + generation quality per question in
        # isolation, not multi-turn behavior, which contextualize_question()
        # only engages when chat_history is non-empty anyway.
        result = generate_answer(ex["question"], chat_history=[], backend=backend)

        samples.append(SingleTurnSample(
            user_input=ex["question"],
            response=result["answer"],
            retrieved_contexts=result["retrieved_contexts"],
            reference=ex["reference_answer"],
        ))
        records.append({
            "question": ex["question"],
            "category": ex["category"],
            "source_docs": ex["source_docs"],
            "reference_answer": ex["reference_answer"],
            "generated_answer": result["answer"],
            "retrieved_contexts": result["retrieved_contexts"],
            "sources": result["sources"],
        })

    return samples, records


def run_ragas(samples: list, judge_llm, judge_embeddings):
    with warnings.catch_warnings():
        # These are ragas's own deprecation notices about its classic
        # metrics API (see module docstring) — not a bug in this script.
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=get_run_config(),
        )
    return result


def merge_scores(records: list[dict], result) -> list[dict]:
    """Join ragas's per-sample scores (result.to_pandas(), same row order
    the samples were submitted in) back onto our per-question records."""
    scores_df = result.to_pandas()
    merged = []
    for record, (_, row) in zip(records, scores_df.iterrows()):
        merged.append({**record, **{m: row.get(m) for m in METRIC_NAMES}})
    return merged


def _is_missing(value) -> bool:
    """True for None AND for float('nan') — a failed/timed-out judge call
    (see module docstring: RAGAS's executor records these per-sample rather
    than crashing the run) comes back as NaN, not None, and `x is not None`
    alone silently lets NaN poison every downstream sum()/mean()."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def print_summary_table(all_results: dict[str, list[dict]]) -> None:
    def mean(records: list[dict], metric: str, category: str | None = None) -> float:
        vals = [
            r[metric] for r in records
            if not _is_missing(r[metric]) and (category is None or r["category"] == category)
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    header = f"{'Metric':<20}" + "".join(f"{b:>18}" for b in BACKENDS)

    def print_block(title: str, category: str | None) -> None:
        print(f"\n{title}")
        print(header)
        for metric in METRIC_NAMES:
            row = f"{metric:<20}" + "".join(
                f"{mean(all_results[b], metric, category):>18.3f}" for b in BACKENDS
            )
            print(row)

    print("\n" + "=" * 60)
    print_block("Overall (mean across all questions)", None)

    categories = sorted({r["category"] for records in all_results.values() for r in records})
    for category in categories:
        print_block(f"--- {category} ---", category)

    # Missing scores get silently excluded from the means above (correctly —
    # see _is_missing()) rather than silently counted as 0, but "excluded"
    # still needs to be visible somewhere, or a few failed judge calls could
    # quietly shift a mean without anyone noticing.
    missing = [
        (backend, r["category"], r["question"], metric)
        for backend, records in all_results.items()
        for r in records
        for metric in METRIC_NAMES
        if _is_missing(r[metric])
    ]
    if missing:
        print(f"\n{len(missing)} metric score(s) missing (failed/timed-out judge calls, excluded from means above):")
        for backend, category, question, metric in missing:
            print(f"  [{backend}/{category}] {metric}: {question[:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run just the first eval question, both backends, and print raw output.",
    )
    args = parser.parse_args()

    full_eval_set = load_eval_set()
    eval_set = full_eval_set[:1] if args.dry_run else full_eval_set
    if args.dry_run:
        print(f"[dry run] Using 1/{len(full_eval_set)} eval questions.\n")

    judge_llm, judge_embeddings = get_judge_llm_and_embeddings()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for backend in BACKENDS:
        print(f"\n=== Generating answers: backend={backend} ===")
        samples, records = build_ragas_samples(eval_set, backend)

        print(f"=== Running RAGAS metrics: backend={backend} ===")
        result = run_ragas(samples, judge_llm, judge_embeddings)
        merged = merge_scores(records, result)
        all_results[backend] = merged

        suffix = "_dryrun" if args.dry_run else ""
        out_path = RESULTS_DIR / f"results_{backend}{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"Wrote {out_path}")

        if args.dry_run:
            print(f"\n--- Raw result for backend={backend} ---")
            print(json.dumps(merged[0], indent=2))

    if not args.dry_run:
        summary_path = RESULTS_DIR / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {summary_path}")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
