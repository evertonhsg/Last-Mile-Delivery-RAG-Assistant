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
│  QUERY PIPELINE (runtime)                               │
│                                                         │
│  User query → Embed → Retrieve top-k → Re-rank         │
│            → LLM (Mistral-7B) → Answer + sources       │
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
| **Phase 2** | Full RAG chain, HyDE, MMR, re-ranking, memory | ⬜ Planned |
| **Phase 3** | LoRA/QLoRA fine-tuning on Mistral-7B | ⬜ Planned |
| **Phase 4** | RAGAS evaluation, LangSmith tracing | ⬜ Planned |
| **Phase 5** | Streamlit chat UI, Docker, deployment | ⬜ Planned |

---

## Tool stack

| Tool | Role |
|---|---|
| `LangChain` | Pipeline orchestration, chains, memory |
| `sentence-transformers` | Text → vector embeddings (`all-MiniLM-L6-v2`) |
| `ChromaDB` | Persistent vector store with metadata filtering |
| `FAISS` | In-memory vector search (Phase 1 baseline) |
| `Ollama` | Run Mistral-7B locally, no API key needed |
| `RAGAS` | RAG evaluation framework |
| `LangSmith` | Pipeline tracing and observability |
| `Streamlit` | Chat UI with source attribution panel |

---

## Repository structure

```
fedex-lastmile-rag/
├── data/
│   ├── raw/                    # synthetic source documents (.md)
│   └── processed/              # chunked text (chunks.json)
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
│   └── memory.py               # conversation memory
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

## Quickstart

### 1 · Clone and install

```bash
git clone https://github.com/evertonhsg/fedex-lastmile-rag
cd fedex-lastmile-rag

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 · Start the local LLM (Ollama)

```bash
# Install from https://ollama.com, then:
ollama pull mistral
```

### 3 · Ingest the knowledge base

```bash
python src/ingest.py --data-dir data/raw --chroma-dir .chroma
```

### 4 · Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 5 · Run Phase 1 notebook

```bash
jupyter notebook notebooks/01_data_prep.ipynb
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

Built by **Everton Gomes**, Senior Engineer at FedEx Europe (Analytics, AI & BI Tech Lead), as part of a structured career development plan targeting AI deployment and technical program management roles.

The project maps directly to real production systems — the business problem, architecture patterns, and evaluation methodology reflect work across FedEx's Planning & Engineering division in Europe and Latin America.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-evertonhsg-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/evertonhsg)
