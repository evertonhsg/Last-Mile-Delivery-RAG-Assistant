# 🚚 Last-Mile Delivery RAG Assistant

<p align="center">
  <b>Production-style Retrieval-Augmented Generation system for logistics intelligence</b><br>
  Natural language Q&A over routing rules, SLAs, zones, and fleet operations
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/LangChain-Orchestration-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/ChromaDB-VectorDB-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Mistral-7B-green?style=flat-square" />
  <img src="https://img.shields.io/badge/RAGAS-Evaluation-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Streamlit-UI-pink?style=flat-square" />
</p>

---

## 🎥 Demo

<p align="center">
  <img src="docs/demo.gif" alt="Demo GIF" width="800"/>
</p>

> 💡 Replace with a real screen recording (Streamlit chat + retrieval sources)

---

## ✨ What This Project Does

This system allows users to ask:

* *“What is the SLA for rural delivery zones?”*
* *“How are delivery zones defined in Madrid?”*
* *“What happens if fleet capacity exceeds threshold?”*

…and get **grounded, source-backed answers** from internal documents.

---

## 🧠 Why This Matters

Logistics organizations rely on fragmented operational knowledge:

* Routing rules
* Zone configurations
* SLA policies
* Fleet constraints

⚠️ Traditionally buried in PDFs and manuals

✅ This project turns that into a **searchable intelligence layer powered by LLMs**

---

## 🏗️ Architecture Overview

<p align="center">
  <img src="docs/architecture.png" width="800"/>
</p>

```text
INGESTION
Docs → Chunking → Embeddings → Vector Store (ChromaDB)

QUERY
User → Embed → Retrieve → Re-rank → LLM → Answer + Sources

EVALUATION
RAGAS → Faithfulness · Relevancy · Precision
```

---

## ⚙️ Core Features

### 🔍 Retrieval-Augmented Generation

* Context-aware answers grounded in documents
* Eliminates hallucinations via source injection

### 🧠 Advanced Retrieval

* HyDE (Hypothetical Document Embeddings)
* MMR (Maximal Marginal Relevance)
* Cross-encoder re-ranking

### 💬 Conversational Memory

* Multi-turn chat support
* Context persistence across queries

### 📊 Evaluation (RAGAS)

* Faithfulness
* Answer relevancy
* Context precision
* Context recall

### 🧪 Experimentation-first Design

* Notebook-driven development
* Each phase fully reproducible

---

## 🧰 Tech Stack

| Layer         | Tools                                      |
| ------------- | ------------------------------------------ |
| Orchestration | LangChain, LangGraph                       |
| Embeddings    | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB     | ChromaDB                                   |
| LLM           | Mistral-7B via Ollama                      |
| UI            | Streamlit                                  |
| Evaluation    | RAGAS                                      |
| Observability | LangSmith                                  |

---

## 📸 UI Preview

<p align="center">
  <img src="docs/ui.png" width="800"/>
</p>

**Features:**

* Chat interface
* Source chunk panel
* Transparent retrieval

---

## 📁 Project Structure

```bash
fedex-lastmile-rag/
├── data/
├── notebooks/
├── src/
├── eval/
├── app/
├── Dockerfile
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & Setup

```bash
git clone https://github.com/evertonhsg/fedex-lastmile-rag
cd fedex-lastmile-rag

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 2. Start Local LLM

```bash
ollama pull mistral
ollama run mistral
```

---

### 3. Ingest Data

```bash
python src/ingest.py --data-dir data/raw --chroma-dir .chroma
```

---

### 4. Run App

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Evaluation Results

| Metric            | Score |
| ----------------- | ----- |
| Faithfulness      | —     |
| Answer Relevancy  | —     |
| Context Precision | —     |
| Context Recall    | —     |

---

## 🧪 Project Roadmap

* [x] RAG pipeline
* [x] Vector DB integration
* [x] Streamlit UI
* [x] Evaluation (RAGAS)
* [ ] Fine-tuned model (LoRA)
* [ ] Production deployment

---

## 📚 Key Learnings

* How embeddings enable semantic search
* Trade-offs in chunking strategies
* How RAG reduces hallucinations
* Evaluation beyond “looks correct”
* Observability in LLM pipelines

---

## ⚠️ Disclaimer

This project uses **synthetic data only**.
No proprietary FedEx data is included.

---

## 👤 Author

**Everton Gomes**
Senior Engineer — FedEx Europe
AI · Data · Systems Design

🔗 [https://www.linkedin.com/in/evertonhsg](https://www.linkedin.com/in/evertonhsg)

---

## ⭐ If You Like This Project

Give it a star ⭐ and feel free to fork!

---

## 🎯 How to Make This EVEN Better

To truly make it “top 1% GitHub”:

### Add these assets:

* `docs/demo.gif` → screen recording (use Kap / Loom)
* `docs/architecture.png` → draw.io or Excalidraw diagram
* `docs/ui.png` → screenshot of Streamlit app

### Optional upgrades:

* Live demo (Streamlit Cloud)
* Docker one-click run
* Benchmark comparison (baseline vs improved RAG)
