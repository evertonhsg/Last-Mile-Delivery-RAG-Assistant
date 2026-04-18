<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FedEx Last-Mile RAG Assistant — README</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0c0f;
    --bg2: #111418;
    --bg3: #181c22;
    --border: #1e2530;
    --border2: #2a3342;
    --text: #e8ecf0;
    --text2: #8a9ab0;
    --text3: #4a5a6e;
    --orange: #f97316;
    --orange-dim: #7c3912;
    --orange-glow: rgba(249,115,22,0.08);
    --teal: #14b8a6;
    --teal-dim: #0f766e;
    --blue: #3b82f6;
    --blue-dim: #1e3a5f;
    --purple: #a78bfa;
    --purple-dim: #3b2f6e;
    --green: #4ade80;
    --green-dim: #14532d;
    --yellow: #fbbf24;
    --red: #f87171;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.7;
    padding: 0;
  }

  /* HERO */
  .hero {
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 3.5rem 3rem 3rem;
    position: relative;
    overflow: hidden;
  }

  .hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -100px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(249,115,22,0.06) 0%, transparent 70%);
    pointer-events: none;
  }

  .hero-eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--orange);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: var(--orange);
  }

  h1 {
    font-family: var(--sans);
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1.1;
    color: var(--text);
    margin-bottom: 1.2rem;
    letter-spacing: -0.02em;
  }

  h1 span { color: var(--orange); }

  .hero-desc {
    font-size: 16px;
    color: var(--text2);
    max-width: 680px;
    line-height: 1.8;
    margin-bottom: 2rem;
  }

  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .badge {
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 4px;
    border: 1px solid;
    letter-spacing: 0.04em;
  }

  .badge-orange { border-color: var(--orange-dim); color: var(--orange); background: var(--orange-glow); }
  .badge-teal { border-color: var(--teal-dim); color: var(--teal); background: rgba(20,184,166,0.06); }
  .badge-blue { border-color: var(--blue-dim); color: var(--blue); background: rgba(59,130,246,0.06); }
  .badge-purple { border-color: var(--purple-dim); color: var(--purple); background: rgba(167,139,250,0.06); }
  .badge-green { border-color: var(--green-dim); color: var(--green); background: rgba(74,222,128,0.06); }

  /* LAYOUT */
  .content {
    max-width: 900px;
    margin: 0 auto;
    padding: 2.5rem 3rem;
  }

  /* SECTION HEADERS */
  .section-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text3);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }

  h2 {
    font-size: 1.25rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 1.25rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
  }

  h3 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    margin: 1.25rem 0 0.4rem;
  }

  section { margin-bottom: 3rem; }

  p { color: var(--text2); margin-bottom: 0.75rem; }

  /* ARCH DIAGRAM */
  .arch {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 2rem;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text2);
    line-height: 2;
    overflow-x: auto;
  }

  .arch .layer-label {
    font-size: 10px;
    color: var(--text3);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    margin-top: 1.2rem;
  }

  .arch .layer-label:first-child { margin-top: 0; }

  .arch .row {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
  }

  .arch .box {
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 5px 14px;
    background: var(--bg3);
    white-space: nowrap;
    font-size: 11px;
  }

  .arch .arrow {
    color: var(--text3);
    padding: 0 6px;
    font-size: 13px;
  }

  .arch .box.o { border-color: var(--orange-dim); color: var(--orange); background: var(--orange-glow); }
  .arch .box.t { border-color: var(--teal-dim); color: var(--teal); background: rgba(20,184,166,0.05); }
  .arch .box.b { border-color: var(--blue-dim); color: var(--blue); background: rgba(59,130,246,0.05); }
  .arch .box.p { border-color: var(--purple-dim); color: var(--purple); background: rgba(167,139,250,0.05); }
  .arch .box.g { border-color: var(--green-dim); color: var(--green); background: rgba(74,222,128,0.05); }

  /* PHASE CARDS */
  .phase-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin-bottom: 1.5rem;
  }

  .phase-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
  }

  .phase-card .phase-num {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text3);
    letter-spacing: 0.1em;
    margin-bottom: 6px;
  }

  .phase-card .phase-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 10px;
  }

  .phase-card ul {
    list-style: none;
    padding: 0;
  }

  .phase-card ul li {
    font-size: 12px;
    color: var(--text2);
    padding: 3px 0;
    display: flex;
    align-items: flex-start;
    gap: 7px;
  }

  .phase-card ul li::before {
    content: '›';
    color: var(--text3);
    flex-shrink: 0;
    margin-top: 1px;
  }

  /* TOOL TABLE */
  .tool-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .tool-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }

  .tool-icon {
    width: 32px; height: 32px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    flex-shrink: 0;
  }

  .tool-body .tool-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 2px;
  }

  .tool-body .tool-role {
    font-size: 11px;
    color: var(--text2);
    line-height: 1.5;
  }

  /* CODE BLOCKS */
  pre {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text2);
    overflow-x: auto;
    margin: 0.75rem 0 1rem;
    line-height: 1.8;
  }

  pre .comment { color: var(--text3); }
  pre .kw { color: var(--purple); }
  pre .str { color: var(--green); }
  pre .cmd { color: var(--orange); }
  pre .fn { color: var(--teal); }
  pre .num { color: var(--yellow); }

  code {
    font-family: var(--mono);
    font-size: 12px;
    background: var(--bg3);
    border: 1px solid var(--border);
    padding: 1px 6px;
    border-radius: 3px;
    color: var(--teal);
  }

  /* FOLDER TREE */
  .tree {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--orange-dim);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 2;
    color: var(--text2);
  }

  .tree .dir { color: var(--orange); }
  .tree .file { color: var(--text2); }
  .tree .comment-inline { color: var(--text3); }

  /* EVAL TABLE */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 0.75rem 0 1rem;
  }

  th {
    text-align: left;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text3);
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
  }

  td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text2);
    vertical-align: top;
  }

  tr:last-child td { border-bottom: none; }

  td:first-child { color: var(--text); font-weight: 500; }

  /* RESULT CARDS */
  .results-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 1rem 0;
  }

  .result-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }

  .result-card .metric {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 500;
    color: var(--orange);
    display: block;
  }

  .result-card .metric-label {
    font-size: 11px;
    color: var(--text3);
    margin-top: 4px;
    font-family: var(--mono);
    letter-spacing: 0.06em;
  }

  /* FOOTER */
  .footer {
    background: var(--bg2);
    border-top: 1px solid var(--border);
    padding: 2rem 3rem;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text3);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
  }

  .footer a { color: var(--orange); text-decoration: none; }

  /* HORIZONTAL RULE */
  hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

  /* CALLOUT */
  .callout {
    background: var(--orange-glow);
    border: 1px solid var(--orange-dim);
    border-left: 3px solid var(--orange);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 13px;
    color: var(--text2);
    margin: 1rem 0;
  }

  .callout strong { color: var(--orange); font-weight: 500; }

  @media (max-width: 700px) {
    .hero { padding: 2rem 1.5rem; }
    h1 { font-size: 1.8rem; }
    .content { padding: 2rem 1.5rem; }
    .tool-grid { grid-template-columns: 1fr; }
    .results-grid { grid-template-columns: repeat(2, 1fr); }
    .footer { padding: 1.5rem; }
  }
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="hero-eyebrow">portfolio project · logistics AI</div>
  <h1>Last-Mile Delivery<br><span>RAG Assistant</span></h1>
  <p class="hero-desc">
    A production-style Retrieval-Augmented Generation system that answers operational questions about last-mile delivery performance — zone definitions, routing policies, SLAs, and fleet capacity — using transformer embeddings, a vector store, and an LLM.
  </p>
  <div class="badges">
    <span class="badge badge-orange">Python 3.11</span>
    <span class="badge badge-teal">LangChain</span>
    <span class="badge badge-blue">ChromaDB</span>
    <span class="badge badge-purple">sentence-transformers</span>
    <span class="badge badge-blue">Mistral-7B / Ollama</span>
    <span class="badge badge-green">Streamlit</span>
    <span class="badge badge-orange">RAGAS</span>
  </div>
</div>

<!-- CONTENT -->
<div class="content">

  <!-- MOTIVATION -->
  <section>
    <div class="section-label">01 — context</div>
    <h2>Business motivation</h2>
    <p>
      Last-mile delivery operations generate a vast amount of structured and unstructured operational knowledge — routing rules, zone configurations, SLA policies, station-level capacity guidelines, and vendor fleet protocols. This knowledge is often locked in documents that require manual lookup, creating friction for planners, engineers, and analysts.
    </p>
    <p>
      This project builds an internal-facing Q&amp;A assistant that makes that knowledge instantly queryable in natural language. It mirrors a real system deployed at FedEx Europe to automate zone and route management, rebuilt from first principles as an open learning artifact.
    </p>
    <div class="callout">
      <strong>Note on data:</strong> All documents in this project use synthetic, fictitious data modelled on real logistics operations. No proprietary FedEx data is used or referenced.
    </div>
  </section>

  <!-- ARCHITECTURE -->
  <section>
    <div class="section-label">02 — design</div>
    <h2>System architecture</h2>

    <div class="arch">
      <div class="layer-label">ingestion pipeline</div>
      <div class="row">
        <div class="box o">PDF / Markdown docs</div>
        <div class="arrow">→</div>
        <div class="box t">Text chunker</div>
        <div class="arrow">→</div>
        <div class="box b">Embedding model</div>
        <div class="arrow">→</div>
        <div class="box p">ChromaDB vector store</div>
      </div>

      <div class="layer-label">query pipeline (runtime)</div>
      <div class="row">
        <div class="box o">User query</div>
        <div class="arrow">→</div>
        <div class="box b">Embed query</div>
        <div class="arrow">→</div>
        <div class="box p">Retrieve top-k chunks</div>
        <div class="arrow">→</div>
        <div class="box t">Re-rank</div>
        <div class="arrow">→</div>
        <div class="box g">LLM (Mistral-7B)</div>
        <div class="arrow">→</div>
        <div class="box o">Answer + sources</div>
      </div>

      <div class="layer-label">evaluation layer</div>
      <div class="row">
        <div class="box t">RAGAS eval set</div>
        <div class="arrow">→</div>
        <div class="box b">Faithfulness</div>
        <div class="arrow">·</div>
        <div class="box b">Answer relevancy</div>
        <div class="arrow">·</div>
        <div class="box b">Context precision</div>
      </div>
    </div>
  </section>

  <!-- PROJECT PHASES -->
  <section>
    <div class="section-label">03 — roadmap</div>
    <h2>Project phases</h2>

    <div class="phase-grid">
      <div class="phase-card">
        <div class="phase-num">PHASE 01</div>
        <div class="phase-name">Foundations &amp; data prep</div>
        <ul>
          <li>Synthetic knowledge base creation</li>
          <li>Chunking strategy experiments</li>
          <li>Embedding models &amp; vector stores</li>
          <li>FAISS → ChromaDB migration</li>
        </ul>
      </div>
      <div class="phase-card">
        <div class="phase-num">PHASE 02</div>
        <div class="phase-name">RAG pipeline</div>
        <ul>
          <li>Basic retrieval + LLM generation</li>
          <li>HyDE &amp; MMR retrieval</li>
          <li>Cross-encoder re-ranking</li>
          <li>Multi-turn conversation memory</li>
        </ul>
      </div>
      <div class="phase-card">
        <div class="phase-num">PHASE 03</div>
        <div class="phase-name">Fine-tuning (optional)</div>
        <ul>
          <li>Instruction dataset creation (JSONL)</li>
          <li>LoRA / QLoRA fine-tune on Colab</li>
          <li>Base vs fine-tuned comparison</li>
        </ul>
      </div>
      <div class="phase-card">
        <div class="phase-num">PHASE 04</div>
        <div class="phase-name">Evaluation &amp; observability</div>
        <ul>
          <li>RAGAS metric suite</li>
          <li>LangSmith tracing</li>
          <li>Latency &amp; token usage logging</li>
        </ul>
      </div>
      <div class="phase-card">
        <div class="phase-num">PHASE 05</div>
        <div class="phase-name">UI &amp; packaging</div>
        <ul>
          <li>Streamlit chat interface</li>
          <li>Source chunk display panel</li>
          <li>Docker packaging</li>
          <li>LinkedIn article write-up</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- TOOL STACK -->
  <section>
    <div class="section-label">04 — stack</div>
    <h2>Tool stack</h2>

    <div class="tool-grid">
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(249,115,22,0.08);color:var(--orange);border:1px solid var(--orange-dim);">LC</div>
        <div class="tool-body">
          <div class="tool-name">LangChain + LangGraph</div>
          <div class="tool-role">Orchestration framework for chains, retrieval, memory, and agentic workflows. The backbone of the pipeline.</div>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(20,184,166,0.08);color:var(--teal);border:1px solid var(--teal-dim);">ST</div>
        <div class="tool-body">
          <div class="tool-name">sentence-transformers</div>
          <div class="tool-role">Hugging Face library for dense vector embeddings. Primary model: <code style="font-size:10px">all-MiniLM-L6-v2</code> (fast, 384-dim).</div>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(59,130,246,0.08);color:var(--blue);border:1px solid var(--blue-dim);">CH</div>
        <div class="tool-body">
          <div class="tool-name">ChromaDB</div>
          <div class="tool-role">Lightweight local vector database. Persistent storage of document embeddings. No server needed to start.</div>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(167,139,250,0.08);color:var(--purple);border:1px solid var(--purple-dim);">OL</div>
        <div class="tool-body">
          <div class="tool-name">Ollama</div>
          <div class="tool-role">Run Mistral-7B or LLaMA3 locally. Free, no API key. Swap for OpenAI/Mistral API via a one-line config change.</div>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(251,191,36,0.08);color:var(--yellow);border:1px solid #78450a;">JN</div>
        <div class="tool-body">
          <div class="tool-name">Jupyter Notebooks</div>
          <div class="tool-role">Each phase has a dedicated notebook. Experiments are documented inline — readable as a learning journal and shareable on GitHub.</div>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(74,222,128,0.08);color:var(--green);border:1px solid var(--green-dim);">SL</div>
        <div class="tool-body">
          <div class="tool-name">Streamlit</div>
          <div class="tool-role">Chat UI for the final assistant. Displays answer + source chunks side by side. Deployable to Streamlit Cloud for free.</div>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(249,115,22,0.08);color:var(--orange);border:1px solid var(--orange-dim);">RG</div>
        <div class="tool-body">
          <div class="tool-name">RAGAS</div>
          <div class="tool-role">RAG evaluation framework. Measures faithfulness, answer relevancy, context precision, and context recall automatically.</div>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon" style="background:rgba(59,130,246,0.08);color:var(--blue);border:1px solid var(--blue-dim);">LS</div>
        <div class="tool-body">
          <div class="tool-name">LangSmith</div>
          <div class="tool-role">Tracing and observability. Inspect every retrieval call, prompt, and LLM response. Free tier is sufficient for this project.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- REPO STRUCTURE -->
  <section>
    <div class="section-label">05 — structure</div>
    <h2>Repository layout</h2>

    <div class="tree">
<span class="dir">fedex-lastmile-rag/</span>
├── <span class="dir">data/</span>
│   ├── <span class="dir">raw/</span>            <span class="comment-inline"># synthetic source documents (.md, .pdf)</span>
│   └── <span class="dir">processed/</span>       <span class="comment-inline"># chunked text ready for embedding</span>
│
├── <span class="dir">notebooks/</span>
│   ├── <span class="file">01_data_prep.ipynb</span>       <span class="comment-inline"># Phase 1: chunking + embedding experiments</span>
│   ├── <span class="file">02_rag_pipeline.ipynb</span>    <span class="comment-inline"># Phase 2: retrieval + generation</span>
│   ├── <span class="file">03_finetuning.ipynb</span>      <span class="comment-inline"># Phase 3: LoRA fine-tune (Colab)</span>
│   └── <span class="file">04_evaluation.ipynb</span>      <span class="comment-inline"># Phase 4: RAGAS scoring</span>
│
├── <span class="dir">src/</span>
│   ├── <span class="file">ingest.py</span>          <span class="comment-inline"># document loading + chunking pipeline</span>
│   ├── <span class="file">embeddings.py</span>      <span class="comment-inline"># embedding model wrapper</span>
│   ├── <span class="file">retriever.py</span>       <span class="comment-inline"># vector store + retrieval logic</span>
│   ├── <span class="file">chain.py</span>           <span class="comment-inline"># RAG chain assembly (LangChain)</span>
│   └── <span class="file">memory.py</span>          <span class="comment-inline"># conversation memory management</span>
│
├── <span class="dir">eval/</span>
│   ├── <span class="file">eval_dataset.json</span>  <span class="comment-inline"># 50 Q&amp;A pairs for RAGAS</span>
│   └── <span class="file">results/</span>           <span class="comment-inline"># RAGAS run outputs</span>
│
├── <span class="dir">app/</span>
│   └── <span class="file">streamlit_app.py</span>   <span class="comment-inline"># chat UI</span>
│
├── <span class="file">requirements.txt</span>
├── <span class="file">Dockerfile</span>
└── <span class="file">README.md</span>
    </div>
  </section>

  <!-- QUICKSTART -->
  <section>
    <div class="section-label">06 — setup</div>
    <h2>Quickstart</h2>

    <h3>1 · Clone and install</h3>
    <pre><span class="comment"># clone the repo</span>
<span class="cmd">git clone</span> https://github.com/evertonhsg/fedex-lastmile-rag
<span class="cmd">cd</span> fedex-lastmile-rag

<span class="comment"># create a virtual environment</span>
<span class="cmd">python -m venv</span> .venv
<span class="cmd">source</span> .venv/bin/activate   <span class="comment"># Windows: .venv\Scripts\activate</span>

<span class="comment"># install dependencies</span>
<span class="cmd">pip install</span> -r requirements.txt</pre>

    <h3>2 · Start the local LLM (Ollama)</h3>
    <pre><span class="comment"># install Ollama from https://ollama.com, then pull a model</span>
<span class="cmd">ollama pull</span> mistral

<span class="comment"># confirm it's running</span>
<span class="cmd">ollama run</span> mistral <span class="str">"Hello, can you summarize a delivery SLA?"</span></pre>

    <h3>3 · Ingest the knowledge base</h3>
    <pre><span class="comment"># chunks docs, embeds them, and persists to ChromaDB</span>
<span class="cmd">python</span> src/ingest.py --data-dir data/raw --chroma-dir .chroma</pre>

    <h3>4 · Run the Streamlit app</h3>
    <pre><span class="cmd">streamlit run</span> app/streamlit_app.py</pre>

    <h3>5 · Run RAGAS evaluation</h3>
    <pre><span class="cmd">jupyter nbconvert</span> --to notebook --execute notebooks/04_evaluation.ipynb</pre>
  </section>

  <!-- EVAL RESULTS -->
  <section>
    <div class="section-label">07 — results</div>
    <h2>Evaluation results</h2>
    <p>RAGAS scores on 50 synthetic Q&amp;A pairs. Results will be updated as the pipeline improves across phases.</p>

    <div class="results-grid">
      <div class="result-card">
        <span class="metric">—</span>
        <div class="metric-label">Faithfulness</div>
      </div>
      <div class="result-card">
        <span class="metric">—</span>
        <div class="metric-label">Answer relevancy</div>
      </div>
      <div class="result-card">
        <span class="metric">—</span>
        <div class="metric-label">Context precision</div>
      </div>
      <div class="result-card">
        <span class="metric">—</span>
        <div class="metric-label">Context recall</div>
      </div>
    </div>
    <p style="font-size:12px;font-family:var(--mono);color:var(--text3);">metrics populated after Phase 4 · scores range 0.0 – 1.0</p>
  </section>

  <!-- LEARNING LOG TABLE -->
  <section>
    <div class="section-label">08 — learnings</div>
    <h2>Key concepts covered</h2>
    <table>
      <thead>
        <tr>
          <th>Concept</th>
          <th>Where it appears</th>
          <th>What you learn</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Transformer embeddings</td>
          <td>Phase 1 — notebook 01</td>
          <td>How text becomes dense vectors; why semantic similarity works for retrieval</td>
        </tr>
        <tr>
          <td>Chunking strategies</td>
          <td>Phase 1 — notebook 01</td>
          <td>Fixed vs recursive vs semantic chunking; effect of overlap on retrieval quality</td>
        </tr>
        <tr>
          <td>Vector stores</td>
          <td>Phase 1 — notebook 01</td>
          <td>FAISS (in-memory) vs ChromaDB (persistent); HNSW index internals</td>
        </tr>
        <tr>
          <td>RAG architecture</td>
          <td>Phase 2 — notebook 02</td>
          <td>Retrieval-augmented generation end-to-end; context window injection patterns</td>
        </tr>
        <tr>
          <td>HyDE &amp; MMR</td>
          <td>Phase 2 — notebook 02</td>
          <td>Improving retrieval beyond naive top-k cosine similarity</td>
        </tr>
        <tr>
          <td>LoRA / QLoRA</td>
          <td>Phase 3 — notebook 03</td>
          <td>Parameter-efficient fine-tuning; training on free GPU (Google Colab)</td>
        </tr>
        <tr>
          <td>RAGAS evaluation</td>
          <td>Phase 4 — notebook 04</td>
          <td>How to rigorously measure RAG quality beyond "does it seem right"</td>
        </tr>
        <tr>
          <td>LangSmith tracing</td>
          <td>Phase 4 — notebook 04</td>
          <td>Production observability patterns for LLM applications</td>
        </tr>
      </tbody>
    </table>
  </section>

  <!-- ABOUT -->
  <section>
    <div class="section-label">09 — author</div>
    <h2>About this project</h2>
    <p>
      Built by <strong style="color:var(--text);font-weight:500;">Everton Gomes</strong>, Senior Engineer at FedEx Europe (Analytics, AI &amp; BI Tech Lead), as part of a structured learning plan targeting roles in AI deployment and technical program management at companies like Mistral AI, Google, and Amazon.
    </p>
    <p>
      The project intentionally maps to real production systems — the business problem, architecture patterns, and evaluation methodology all reflect work done in FedEx's Planning &amp; Engineering division across Europe and Latin America.
    </p>
    <p style="font-size:13px;color:var(--text3);">
      Connect on LinkedIn · <a href="https://www.linkedin.com/in/evertonhsg" style="color:var(--orange);text-decoration:none;">linkedin.com/in/evertonhsg</a>
    </p>
  </section>

</div>

<!-- FOOTER -->
<div class="footer">
  <span>fedex-lastmile-rag · everton gomes · 2025</span>
  <span>python 3.11 · langchain · chromadb · mistral-7b · ragas</span>
</div>

</body>
</html>
