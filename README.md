# Last-Mile Delivery RAG Assistant

> A production-style Retrieval-Augmented Generation system for last-mile delivery operations — built as a structured learning project covering transformers, embeddings, RAG pipelines, fine-tuning, and LLM evaluation.

---

## Business context

Last-mile delivery operations generate large volumes of structured operational knowledge — routing rules, zone configurations, SLA policies, capacity guidelines, and fleet protocols. This knowledge is often locked in documents requiring manual lookup, creating friction for planners and engineers.

This project builds an internal-facing Q&A assistant that makes that knowledge instantly queryable in natural language.

> **Data note:** All documents use synthetic, fictitious data modelled on real logistics operations. No proprietary data is used.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE (run once)                          │
│                                                         │
│  Docs (MD/PDF) → Chunker → Embedding model → ChromaDB  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  QUERY PIPELINE (implemented, Phase 2)                  │
│                                                         │
│  User query → Contextualize (if multi-turn) → Retrieve  │
│  (similarity / MMR / HyDE) → Cross-encoder re-rank      │
│  → LLM (Ollama/Mistral-7B) → Answer + sources           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  EVALUATION LAYER                                       │
│                                                         │
│  RAGAS: faithfulness · answer relevancy · context recall│
└─────────────────────────────────────────────────────────┘
```

---

## Project phases

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Chunking strategies, embeddings, FAISS → ChromaDB | ✅ Done |
| **Phase 2** | Full RAG chain, HyDE, MMR, re-ranking, memory | ✅ Done |
| **Phase 3** | LoRA fine-tuning on a 4-bit quantized Mistral-7B (QLoRA-style) via MLX | ✅ Done |
| **Phase 4** | RAGAS evaluation — base vs. fine-tuned, 21 held-out questions | ✅ Done |
| **Phase 5** | Streamlit chat UI, Docker, deployment | ⬜ Planned |

---

## Production Considerations: Guardrails vs. Fine-Tuning

A question worth answering explicitly for any RAG system heading toward production is where security and compliance boundaries actually belong — because the honest answer is that they don't belong in the language model at all, no matter how it's prompted or fine-tuned.

**Access control has to be deterministic, and that means it belongs in retrieval, not generation.** If different users are only supposed to see certain zones, regions, or document classes, that restriction needs to be enforced as a metadata filter applied *before* similarity search runs — role/region tags on each chunk, checked against the authenticated user's identity, so that a restricted chunk is never even a retrieval candidate for a user who shouldn't see it. It is tempting to instead just tell the model in its system prompt "only answer using documents this user is allowed to see" and trust it to comply, or to assume that fine-tuning on an access-scoped dataset would bake the restriction in. Both are probabilistic controls layered on a system that's supposed to provide a hard guarantee — and probabilistic controls can be jailbroken, confused by clever phrasing, or simply get it wrong on some fraction of requests. A permissions boundary needs to fail closed every time, not most of the time, which is exactly the property a metadata filter has and an instruction to an LLM does not.

**Data governance — PII masking and pseudonymization — belongs even further upstream, at ingestion time.** Once sensitive data is embedded into the vector store, it's effectively been copied into a new representation that's much harder to audit, redact, or delete on request than the original document was. The right place to catch and mask that data is during the document-loading and chunking step, before anything is embedded — so the vector store, and every downstream retrieval and generation call, only ever operates on data that was already safe to expose in the first place. Trying to filter sensitive content back out at query time or in the model's output is solving the problem one step too late.

**A third, independent layer — input/output guardrails — exists specifically because the generator's own behavior isn't the thing you want to rely on for safety.** A lightweight moderation or classification check on the incoming question (catching prompt injection, out-of-scope requests, obvious abuse) and a second check on the model's output (catching policy violations, leaked instructions, hallucinated claims presented too confidently) gives you an enforcement point that sits outside the LLM and doesn't share its failure modes. If the generator has a bad day — a jailbreak, a weird edge case, a regression from a model update — the guardrail layer is what catches it, precisely because it isn't the same system making the same kind of probabilistic judgment call.

**Fine-tuning (Phase 3 of this project) has a real role to play, but it's a narrower one than any of the above.** It's well suited to shaping *how* the model communicates — consistent tone, house formatting conventions, fluency with domain-specific vocabulary and abbreviations — because those are properties where "usually right" is an acceptable bar. It is not well suited to enforcing *what* the model is allowed to say or see, because "usually right" is not an acceptable bar for a security or compliance boundary. Conflating the two — reaching for fine-tuning as a way to make a model "learn" access restrictions or compliance rules — is a common mistake, and one worth calling out explicitly rather than leaving implicit.

To be clear about scope: this project currently implements the RAG pipeline itself — retrieval, re-ranking, generation, and conversation memory — and nothing described in this section is built yet. Access-control filtering and a guardrails layer are noted here as the production-hardening this architecture is designed to support, not as features of this portfolio version.

---

## Tool stack

| Tool | Role | Status |
|---|---|---|
| `LangChain` | Pipeline orchestration, chains, memory | ✅ Implemented |
| `sentence-transformers` | Text → vector embeddings (`all-MiniLM-L6-v2`) | ✅ Implemented |
| `ChromaDB` | Persistent vector store with metadata filtering | ✅ Implemented |
| `FAISS` | In-memory vector search (Phase 1 baseline) | Superseded by ChromaDB (Phase 2) |
| `Ollama` | Run Mistral-7B locally, no API key needed | ✅ Implemented |
| `mlx-lm` | QLoRA-style fine-tuning (LoRA on a 4-bit quantized base) + local inference on Apple Silicon (Apple Silicon only — will not install on Intel Macs or non-Apple hardware) | ✅ Implemented |
| `RAGAS` | RAG evaluation framework — faithfulness, answer relevancy, context precision/recall | ✅ Implemented |
| `LangSmith` | Pipeline tracing and observability | ⬜ Planned |
| `Streamlit` | Chat UI with source attribution panel | ⬜ Planned |

---

## Repository structure

```
lastmile-delivery-rag/
├── data/
│   ├── raw/                         # synthetic source documents (.md)
│   ├── processed/
│   │   └── chunks.json              # chunked output from Phase 1
│   ├── chroma_db/                   # persisted Chroma vector store (gitignored, rebuild via build_vectorstore.py)
│   ├── finetune/                    # Phase 3: LoRA training data
│   │   ├── train.jsonl              # current train split (95 ex.) — confident-answer + refusal examples merged
│   │   ├── valid.jsonl              # current val split (17 ex.)
│   │   ├── train_v1.jsonl           # pre-refusal-fix split (79 ex., confident-answer only) — kept for reference
│   │   ├── valid_v1.jsonl           # pre-refusal-fix val split (15 ex.) — kept for reference
│   │   └── refusal_examples.json    # the 18 deliberate refusal examples merged into train/valid.jsonl
│   └── eval/                        # Phase 4: RAGAS evaluation set + results
│       ├── eval_set.json            # 21 held-out questions (15 single-doc, 3 cross-doc, 3 refusal) + reference answers
│       └── results/
│           ├── results_ollama.json          # per-question scores, base backend
│           ├── results_mlx-finetuned.json   # per-question scores, fine-tuned backend
│           ├── summary.json                 # both backends' per-question records combined
│           └── results_*_dryrun.json        # 1-question sanity-check runs, kept for reference
│
├── notebooks/
│   └── 01_data_prep.ipynb           # Phase 1: chunking + embedding experiments
│
├── src/
│   ├── build_vectorstore.py         # Phase 2: builds the persistent Chroma store from chunks.json
│   ├── rag_chain.py                 # Phase 2/3: retrieval + generation + memory, dual backend, REPL entry point
│   ├── generate_finetune_data.py    # Phase 3: builds train/valid.jsonl from chunks.json (RAFT-style + quality gates)
│   ├── generate_refusal_examples.py # Phase 3: builds the deliberate refusal examples
│   ├── train_lora.py                # Phase 3: LoRA fine-tuning via mlx-lm, versioned run config
│   ├── compare_models.py            # Phase 3: qualitative base-vs-fine-tuned side-by-side
│   ├── mlx_smoke_test.py            # Phase 3: base-model load/generate/memory sanity check
│   ├── generate_eval_set.py         # Phase 4: builds the held-out eval_set.json from data/raw/*.md
│   └── run_eval.py                  # Phase 4: RAGAS harness — scores both backends, prints the comparison table
│
├── adapters/                         # Phase 3: LoRA runs (.safetensors gitignored — see .gitignore; configs/logs kept)
│   ├── lastmile-lora/                # run 1: 316 iters (4 epochs), 94-example dataset — overfit past iter ~159
│   ├── lastmile-lora-earlystop/      # run 2: early-stopped at iter 160 on the same 94-example dataset
│   ├── lastmile-lora-v2/             # run 3: 400 iters on the merged 112-example (+refusal) dataset, full checkpoint history
│   └── lastmile-lora-v2-best/        # iter-340 checkpoint copied out of lastmile-lora-v2 — what rag_chain.py loads
│
├── requirements.txt
└── README.md
```

---

## What's implemented (Phase 2)

`src/rag_chain.py` is the full retrieval-augmented generation pipeline, runnable directly as an interactive chat REPL:

```bash
python src/rag_chain.py
```

It wraps the ChromaDB store built during ingestion and a locally-running Ollama/Mistral-7B model behind these pieces:

| Function | What it does |
|---|---|
| `get_retriever(k, search_type)` | Opens the persisted Chroma collection as a LangChain retriever. `search_type="similarity"` (default) returns plain top-k nearest neighbors; `search_type="mmr"` re-ranks for Maximal Marginal Relevance, trading a little relevance for diversity across chunks. |
| `hyde_retrieve(question, k)` | Hypothetical Document Embeddings — asks the LLM to draft a plausible policy-doc paragraph that would answer the question, then embeds and searches with *that* instead of the raw question. |
| `rerank_retrieve(question, k, fetch_k)` | Two-stage retrieval: a cheap similarity search pulls `fetch_k` candidates, then a `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder jointly scores each (question, chunk) pair and keeps the top `k`. |
| `contextualize_question(chat_history, question)` | Rewrites follow-up questions (e.g. "what about zone 2?") into standalone form using the last 2–3 conversation turns, so retrieval isn't searching on a pronoun. |
| `format_context(docs)` | Joins retrieved chunks into one prompt-ready string, tagging each with its source filename so the model can cite it. |
| `generate_answer(question, chat_history, k)` | Orchestrates a full turn: contextualize → retrieve → format → prompt → generate via `ChatOllama(model="mistral", temperature=0)`. Returns the answer, its source filenames, and the standalone question it retrieved with. |

**Retrieval techniques compared:**

- **Similarity (top-k)** — the baseline. Fast, but can return several near-duplicate chunks when the same fact is restated across documents.
- **MMR** — re-ranks the candidate pool for relevance *and* diversity; most useful when a question's full answer is scattered across multiple policy documents.
- **HyDE** — retrieves using an LLM-generated hypothetical answer instead of the raw question; most useful when a question is phrased very differently from how the source documents are written.
- **Cross-encoder re-ranking** — the two-stage retrieve-then-rerank pattern: a bi-encoder cheaply narrows the field to `fetch_k` candidates, then a cross-encoder (which jointly attends over the question and each candidate, rather than comparing precomputed vectors) picks the final top-`k`. This is the default retrieval strategy inside `generate_answer()` — it produced the most relevant top-k in side-by-side testing against similarity, MMR, and HyDE.

**Conversation memory** — multi-turn follow-ups are handled by rewriting each new question into a standalone form before retrieval, using recent chat history. The final answer is still generated strictly from that turn's retrieved chunks: chat history is never fed into the answer-generation prompt itself, so grounding stays strict — a conversational tone is fine, but the model can't repeat something it (or the user) said earlier as if it were a retrieved fact.

---

## What's implemented (Phase 3)

Phase 3 takes `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (an MLX-converted, 4-bit quantized build of Mistral, run locally on Apple Silicon via `mlx-lm`) and LoRA fine-tunes it — QLoRA-style, since the base model is already 4-bit quantized — on this project's own knowledge base, so the model's answer *style* — grounding, source citation, and refusal behavior — is shaped by examples drawn directly from `data/raw/`, rather than relying entirely on prompting.

| Script | What it does |
|---|---|
| `generate_finetune_data.py` | Builds the core training set: for each of the 55 Phase 1 chunks, an LLM (Ollama/Mistral) generates 2–3 plausible questions that chunk answers, then a second LLM call answers each question using *only* that chunk as context — the same `(context, question) → answer` shape `rag_chain.py` uses at inference time (a RAFT-style dataset, not bare Q&A), so fine-tuning reinforces reading and citing supplied context rather than answering from memorized facts. |
| `generate_refusal_examples.py` | Builds a second, smaller set of *deliberately* mismatched examples — a genuinely out-of-scope question (e.g. fuel surcharge policy, driver certification) paired with a randomly chosen, unrelated chunk — labeled with the RAG prompt's exact required refusal sentence. See "Refusal-wording dilution" below for why this exists. |
| `train_lora.py` | Runs the actual LoRA fine-tune via `mlx-lm`'s Python API, with every hyperparameter and the reasoning behind it committed in the file itself (rank-8 LoRA — QLoRA-style, since the base model is already 4-bit quantized — on the top 16 of 32 layers, prompt-masked loss so only the assistant's answer tokens are trained on, per-run configurable `--iters`/`--adapter-path`/`--save-every`). Logs the full train/val loss curve to `training_log.json` alongside each adapter. |
| `compare_models.py` | Runs a fixed set of test questions through the base model and the fine-tuned adapter side by side, using identical retrieved context for both, for a qualitative gut-check before RAGAS scoring (Phase 4). |

**Two quality gates in `generate_finetune_data.py`, and why:** a chunk that's mostly a markdown header with little body text doesn't give question-generation enough to work with, so thin chunks (under ~40 words of body text) are skipped before spending any LLM calls on them. Separately, even a substantial chunk can produce a question that drifts off-topic during generation — and because each question is generated *from* a specific chunk, that chunk is guaranteed by construction to answer it, so any refusal-shaped answer at that point is a signal of drift, not a legitimate "out of scope" case. Both accidental-refusal filtering and one regeneration retry are applied automatically (17 of 55 chunks skipped as too thin; 15 of the remaining question/answer pairs caught by the refusal filter, 12 recovered by retrying, 3 dropped), leaving 94 confident, correctly-grounded examples.

**Finding 1 — overfitting, and early stopping.** The first LoRA run (94 examples, 316 iterations ≈ 4 epochs) showed validation loss bottom out at iteration 159 (0.348) and then drift back up to 0.376 by the final iteration — an ~8% relative regression — while training loss kept falling the entire time. That's the textbook overfitting signature: past roughly 2 epochs, the adapter was still fitting the training set but no longer generalizing better. A second run, identical in every setting except stopping at iteration 160, reproduced that same minimum almost exactly (0.350) and landed there as both the best *and* final validation score — confirming the early-stopping point rather than just assuming it from the first curve.

**Finding 2 — refusal-wording dilution.** Comparing the early-stopped fine-tuned model against the base model on an out-of-scope question (`compare_models.py`) surfaced a subtler issue: both models correctly declined to answer, but the fine-tuned model's refusal had drifted slightly off the RAG prompt's exact required wording, while the base model used it verbatim. The root cause traced back to the quality gate described above — by design, it filtered out *every* accidental refusal from the training set, which meant the model never saw a single correctly-labeled "this genuinely isn't in the knowledge base" example during training, only confident, fully-answered ones. The fix, `generate_refusal_examples.py`, adds that missing signal back deliberately rather than accidentally: 18 out-of-scope questions, each paired with a randomly mismatched chunk, labeled with the literal required fallback sentence (pulled programmatically from the prompt template itself, not retyped by hand, so it can't drift out of sync). Merged with the 94 confident-answer examples, the dataset grew to 112 total (94 confident / 18 refusal — 16.1% refusal examples), re-split 95 train / 17 validation. Re-running `compare_models.py` against a model fine-tuned on this merged set confirmed the fix — including against a second, harder out-of-scope question designed so retrieval returns superficially on-topic chunks (mentioning "Zone 1," SLA figures) that still don't actually answer it.

**A nice side-effect worth calling out: the two training runs' validation curves look qualitatively different.** Run 1 (94 examples) shows a clean rise-then-fall — bottoms at iteration 159, then climbs — the sharp signature of a small, fairly narrow dataset saturating quickly. Run 3 (112 examples, 400 iterations) shows no such divergence: validation loss drifts gently downward through most of training and flattens into noisy oscillation from roughly iteration 120 onward, still near its best value at the final iteration (iteration 340 was treated as this run's practical optimum, since `--save-every` was set to match the evaluation interval specifically so a checkpoint exists at the minimum). A larger, more varied training set — even 18 additional examples — visibly pushed back the point where the model runs out of new signal to learn from.

**Dual generation backends.** `rag_chain.py`'s `generate_answer()` now takes a `backend` parameter: `"ollama"` (unchanged from Phase 2 — base Mistral via a local Ollama server) or `"mlx-finetuned"` (the LoRA adapter above, served via `mlx-lm`). These stay as two separate backends rather than one fused model because Ollama runs Mistral through a GGUF/llama.cpp runtime, while the LoRA adapter was trained against — and is stored as MLX safetensors against — the MLX build of the model; there's no lossless bridge between the two short of a real fuse-then-convert-to-GGUF pipeline. Keeping the adapter unfused also preserves exactly the thing this phase needed throughout: the ability to swap in a different checkpoint (the overfit run, the early-stopped run, or the final merged-dataset run) and compare it against the base model on demand, which a single merged model wouldn't support nearly as easily.

---

## Quickstart

### 1 · Clone and install

```bash
git clone https://github.com/evertonhsg/Last-Mile-Delivery-RAG-Assistant.git
cd Last-Mile-Delivery-RAG-Assistant

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 · Build the vector store

```bash
python src/build_vectorstore.py   # builds data/chroma_db/ (gitignored) from data/processed/chunks.json
```

### 3 · Start the local LLM (Ollama)

```bash
# Install from https://ollama.com, then:
ollama pull mistral
```

### 4 · Run the RAG assistant

```bash
python src/rag_chain.py
```

---

## Knowledge base documents (synthetic)

| Document | Contents |
|---|---|
| `zone_definitions.md` | Zone 1–4 classification, fleet types, SLAs per zone, reclassification policy |
| `sla_policies.md` | Service tiers (Priority/Express/Standard), targets, failed delivery handling |
| `routing_rules.md` | Route types, fleet allocation, VRPTW objective function, KPIs |
| `capacity_planning.md` | 13-week planning cycle, demand forecasting models, fleet sizing rules |
| `station_operations.md` | Station types, sortation process, end-of-day, escalation procedures |

---

## Evaluation results

RAGAS scoring of both `generate_answer()` backends (`ollama` = base Mistral, `mlx-finetuned` = the Phase 3 LoRA adapter) against 21 held-out questions — see `data/eval/eval_set.json` for the exact questions and reference answers, and `src/run_eval.py` for the harness.

**Overall (n=21)**

| Metric | ollama (base) | mlx-finetuned |
|---|---|---|
| Faithfulness | 0.835 | **0.880** |
| Answer relevancy | **0.706** | 0.695 |
| Context precision | 0.799 | 0.802 |
| Context recall | 0.816 | 0.816 |

**By category**

*single-doc (n=15)*

| Metric | ollama (base) | mlx-finetuned |
|---|---|---|
| Faithfulness | 0.825 | 0.832 |
| Answer relevancy | 0.812 | 0.792 |
| Context precision | 0.937 | 0.941 |
| Context recall | 1.000 | 1.000 |

*cross-doc (n=3)*

| Metric | ollama (base) | mlx-finetuned |
|---|---|---|
| Faithfulness | 0.717 | **1.000** |
| Answer relevancy | 0.885 | 0.903 |
| Context precision | 0.907 | 0.907 |
| Context recall | 0.833 | 0.833 |

*refusal (n=3)*

| Metric | ollama (base) | mlx-finetuned |
|---|---|---|
| Faithfulness | 1.000 | 1.000 |
| Answer relevancy | 0.000 | 0.000 |
| Context precision | 0.000 | 0.000 |
| Context recall | 0.000 | 0.000 |

### Methodology and caveats

**Held-out, not recycled.** All 21 questions are built directly from `data/raw/*.md` (see `src/generate_eval_set.py`) — never passed through `generate_finetune_data.py` or `generate_refusal_examples.py`, and not sampled from `data/finetune/*.jsonl`, which the LoRA adapter was both trained AND checkpoint-selected against. A model doing well on data shaped like its own training set would be a much weaker result than doing well here.

**The judge is a local 7B model, not GPT-4.** All four metrics are scored by `ChatOllama(model="mistral", temperature=0)` — the same model this project runs everywhere else, wrapped via RAGAS's LangChain wrapper, with no API key and no hosted call. Most RAGAS tutorials assume a frontier hosted judge (GPT-4/GPT-4o); a 7B local model gives noticeably noisier per-statement verdicts (we saw this directly during harness testing — 5 of 168 judge calls in this run outright failed with a timeout or a malformed structured-output parse, excluded from the means above rather than counted as 0). Treat the absolute scores as directional, not as calibrated against any external benchmark.

**Judge self-evaluation bias runs, if anything, against the fine-tuning result.** The judge (plain Ollama Mistral) is the literal same model and weights as the `ollama` backend being scored — a textbook self-evaluation setup, which LLM-as-judge research generally finds biases a judge toward rating its *own* phrasing and style more favorably. The `mlx-finetuned` backend, by contrast, is being judged by a genuinely different model (different weights from LoRA fine-tuning, different runtime — MLX vs. Ollama/GGUF). If this bias has any effect here, it should work *against* observing a fine-tuning improvement, not for it — which makes the faithfulness gap below more credible, not less.

**Sample sizes are small and uneven across categories — 15 single-doc, but only 3 each for cross-doc and refusal.** This cuts against reading the category breakdown as precise. The cross-doc faithfulness gap (0.717 → 1.000) is a full swing across just 3 questions — directionally consistent with RAFT-style grounded-answer training helping most on multi-document synthesis, but not something to treat as a precise effect size at n=3. Single-doc's larger n=15 makes its small answer-relevancy gap (0.812 → 0.792) more likely to reflect genuine (if minor) backend variation than pure noise — though at n=15 this still isn't a rigorous statistical claim in either direction.

**Refusal category's near-zero answer relevancy/context precision/context recall for BOTH backends is expected metric behavior, not a failure.** RAGAS's `answer_relevancy` explicitly scores non-committal answers like "I don't have enough information..." as low-relevance by design (it's in the metric's own documented examples), and `context_precision`/`context_recall` have nothing to score when the reference answer contains no factual claims to check retrieval against. The metric that *does* meaningfully validate refusal quality — **faithfulness = 1.000 for both backends** — confirms neither model hallucinated or contradicted the (irrelevant) retrieved context when correctly declining to answer.

**Headline finding:** fine-tuning improved overall faithfulness (0.835 → 0.880), concentrated in cross-document synthesis questions — exactly where RAFT-style training on grounded, context-only answers would be expected to help most. Retrieval-dependent metrics (context precision/recall) are within judge noise of each other across both backends, correctly — `_build_prompt()` is shared, so both backends retrieve from the identical call, and any difference reflects judge-scoring noise across separate runs, not a real retrieval difference.

---

## Concepts covered

| Concept | Reference | What you learn |
|---|---|---|
| Transformer embeddings | `01_data_prep` | Text → dense vectors; cosine similarity for semantic search |
| Chunking strategies | `01_data_prep` | Fixed vs recursive; chunk size/overlap trade-offs |
| Vector stores | `01_data_prep` | FAISS internals; ChromaDB persistence + metadata filtering |
| RAG architecture | `src/rag_chain.py` | End-to-end retrieval-augmented generation |
| HyDE & MMR | `src/rag_chain.py` | Advanced retrieval beyond naive top-k |
| LoRA fine-tuning (QLoRA-style) | `src/train_lora.py`, `src/generate_finetune_data.py` | Parameter-efficient fine-tuning on a 4-bit quantized base, locally via MLX; RAFT-style dataset construction, overfitting detection, early stopping |
| RAGAS | `src/run_eval.py`, `src/generate_eval_set.py` | Rigorous RAG quality measurement; faithfulness/relevancy/precision/recall, held-out eval set design, LLM-as-judge caveats |
| LangSmith | (not yet created) | Production observability for LLM apps |

---

## About

Built by **Everton Gomes** as a structured learning project covering transformers, embeddings, RAG pipelines, fine-tuning, and LLM evaluation. The business problem, architecture patterns, and evaluation methodology are modeled on real-world last-mile delivery and logistics operations, implemented here entirely with synthetic data — no proprietary or employer data is used anywhere in this project.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-evertonhsg-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/evertonhsg)
