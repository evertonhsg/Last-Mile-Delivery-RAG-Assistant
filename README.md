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
| **Phase 1** | Chunking strategies, embeddings, FAISS → ChromaDB | 🟠 In progress |
| **Phase 2** | Full RAG chain, HyDE, MMR, re-ranking, memory | ✅ Done |
| **Phase 3** | LoRA/QLoRA fine-tuning on Mistral-7B | ⬜ Planned |
| **Phase 4** | RAGAS evaluation, LangSmith tracing | ⬜ Planned |
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
| `FAISS` | In-memory vector search (Phase 1 baseline) | 🟠 In progress |
| `Ollama` | Run Mistral-7B locally, no API key needed | ✅ Implemented |
| `RAGAS` | RAG evaluation framework | ⬜ Planned |
| `LangSmith` | Pipeline tracing and observability | ⬜ Planned |
| `Streamlit` | Chat UI with source attribution panel | ⬜ Planned |

---

## Repository structure

```
lastmile-delivery-rag/
├── data/
│   ├── raw/                    # synthetic source documents (.md)
│   ├── processed/              # chunked text (chunks.json)
│   └── chroma_db/              # persisted Chroma vector store
│
├── notebooks/
│   ├── 01_data_prep.ipynb      # Phase 1: chunking + embedding experiments
│   ├── 02_rag_pipeline.ipynb   # Phase 2: retrieval + generation
│   ├── 03_finetuning.ipynb     # Phase 3: LoRA fine-tune (Colab)
│   └── 04_evaluation.ipynb     # Phase 4: RAGAS scoring
│
├── src/
│   ├── ingest.py               # document loading + chunking pipeline
│   ├── embeddings.py           # embedding model wrapper
│   ├── retriever.py            # vector store + retrieval logic
│   ├── chain.py                # RAG chain assembly
│   ├── memory.py               # conversation memory
│   └── rag_chain.py            # Phase 2: retrieval + generation + memory, REPL entry point
│
├── eval/
│   ├── eval_dataset.json       # 50 Q&A pairs for RAGAS
│   └── results/                # RAGAS run outputs
│
├── app/
│   └── streamlit_app.py        # chat interface
│
├── requirements.txt
└── Dockerfile
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

## Quickstart

### 1 · Clone and install

```bash
git clone https://github.com/evertonhsg/lastmile-delivery-rag
cd lastmile-delivery-rag

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

### 4 · Ingest the knowledge base

```bash
python src/ingest.py --data-dir data/raw --chroma-dir .chroma
```

### 5 · Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 6 · Run the RAG assistant

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

> *Populated after Phase 4 — RAGAS scoring on 50 synthetic Q&A pairs.*

| Metric | Score | Description |
|---|---|---|
| Faithfulness | — | Is the answer grounded in the retrieved context? |
| Answer relevancy | — | Does the answer actually address the question? |
| Context precision | — | Are the retrieved chunks relevant? |
| Context recall | — | Was the necessary context retrieved? |

---

## Concepts covered

| Concept | Notebook | What you learn |
|---|---|---|
| Transformer embeddings | `01_data_prep` | Text → dense vectors; cosine similarity for semantic search |
| Chunking strategies | `01_data_prep` | Fixed vs recursive; chunk size/overlap trade-offs |
| Vector stores | `01_data_prep` | FAISS internals; ChromaDB persistence + metadata filtering |
| RAG architecture | `02_rag_pipeline` | End-to-end retrieval-augmented generation |
| HyDE & MMR | `02_rag_pipeline` | Advanced retrieval beyond naive top-k |
| LoRA / QLoRA | `03_finetuning` | Parameter-efficient fine-tuning on free GPU |
| RAGAS | `04_evaluation` | Rigorous RAG quality measurement |
| LangSmith | `04_evaluation` | Production observability for LLM apps |

---

## About

Built by **Everton Gomes** as a structured learning project covering transformers, embeddings, RAG pipelines, fine-tuning, and LLM evaluation. The business problem, architecture patterns, and evaluation methodology are modeled on real-world last-mile delivery and logistics operations, implemented here entirely with synthetic data — no proprietary or employer data is used anywhere in this project.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-evertonhsg-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/evertonhsg)
