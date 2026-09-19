"""
Phase 2 — Full RAG pipeline (retrieve + prompt + generate) for the Last-Mile
Delivery RAG Assistant, with multi-turn conversation memory.

Given a question, this retrieves relevant chunks from the Chroma store built
by build_vectorstore.py, assembles a grounded prompt, and sends it to a local
Ollama model to get a final answer with cited sources. Retrieval uses
rerank_retrieve() (similarity search + cross-encoder re-ranking), which
tested as the strongest of the retrieval variants explored earlier in
Phase 2. Follow-up questions are rewritten into standalone form via
contextualize_question() before retrieval, so pronouns and implicit
references ("what about zone 2?") resolve to something the vector store can
actually search on.

Requires Ollama running locally with the "mistral" model pulled:
    ollama pull mistral

Run directly for an interactive chat loop (type "exit" to quit):
    python src/rag_chain.py
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

# Must match build_vectorstore.py exactly — same directory, collection name,
# and embedding model, otherwise we'd either find nothing or (worse) get
# vectors that were embedded by a different model and aren't comparable.
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "lastmile_delivery_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_vectorstore() -> Chroma:
    """Open the persisted Chroma collection (shared by every retrieval path).

    Factored out of get_retriever() because hyde_retrieve() also needs direct
    vectorstore access (for similarity_search with a custom query string)
    rather than the narrower `.invoke(query)` interface a retriever exposes.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_retriever(k: int = 4, search_type: str = "similarity") -> VectorStoreRetriever:
    """Load the existing Chroma collection and expose it as a retriever.

    A *retriever* is a thin LangChain wrapper around a vector store that
    standardizes retrieval behind one method: given a query string, return a
    list of relevant `Document` objects. Wrapping the vector store this way
    means later code (and any LangChain chain we build) doesn't need to know
    or care that the underlying store happens to be Chroma — it could be
    swapped for FAISS, Pinecone, etc. without changing the retrieval call site.

    Note: this does NOT re-embed or rebuild anything. `Chroma(...)` here just
    opens the collection that build_vectorstore.py already persisted to disk.

    search_type:
      - "similarity" (default): plain top-k nearest neighbors by embedding
        distance. Fast and usually fine, but if several chunks are near-
        duplicates of each other (e.g. the SLA number is restated in both
        sla_policies.md and station_operations.md), top-k can hand the LLM
        4 chunks that all say almost the same thing instead of covering the
        question from multiple angles.
      - "mmr": Maximal Marginal Relevance re-ranks for *relevance + diversity*
        instead of relevance alone. Most useful here when a question could be
        answered more completely by pulling from several different policy
        documents (e.g. "how are Zone 1 deliveries handled?" touches SLA
        commitments, routing rules, AND station operating hours) — MMR is
        more likely to surface one chunk from each rather than 4 near-
        duplicates from the single closest document.
    """
    vectorstore = _get_vectorstore()

    if search_type == "mmr":
        search_kwargs = {
            # fetch_k: how many candidates to pull by similarity BEFORE the
            # MMR re-ranking step runs. Needs to be larger than k so there's
            # actually a diverse pool to choose from — fetch_k=k would just
            # give back plain top-k with extra overhead.
            "fetch_k": k * 3,
            "k": k,
            # lambda_mult: 1.0 = pure relevance (behaves like similarity
            # search), 0.0 = pure diversity (ignores relevance almost
            # entirely). 0.5 is a balanced middle ground — still favors
            # relevant chunks but actively penalizes picking one that's
            # redundant with a chunk already selected.
            "lambda_mult": 0.5,
        }
    else:
        # search_kwargs={"k": k} controls how many chunks come back per
        # query. Too few and the LLM may lack context; too many and
        # irrelevant chunks dilute the prompt (and cost more tokens).
        search_kwargs = {"k": k}

    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def format_context(docs: list[Document]) -> str:
    """Join retrieved documents into a single context string for the prompt.

    Each chunk is prefixed with its source filename so the LLM can see, in
    the context itself, which document each piece of text came from — that's
    what makes it possible to ask the model to cite its sources later.
    """
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n".join(blocks)


# ── Prompt template ──────────────────────────────────────────────────────────
# ChatPromptTemplate builds a list of chat messages (system/human/...) from a
# template with placeholders. We keep this separate from the retrieval and
# generation code so the prompt wording can be tuned on its own — it's the
# single place that defines "how the model is instructed to behave."
#
# {context} and {question} are filled in at call time via .invoke({...}).
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about "
            "last-mile delivery operations.\n"
            "Answer ONLY using the information in the provided context — do "
            "not use outside knowledge.\n"
            "Always cite the source filename(s) you used to answer, e.g. "
            "(Source: sla_policies.md).\n"
            "If the context does not contain enough information to answer "
            "the question, respond exactly with: \"I don't have enough "
            "information in the knowledge base to answer that.\"\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# ── HyDE prompt template ─────────────────────────────────────────────────────
# HyDE = Hypothetical Document Embeddings. The trick: embedding models judge
# similarity by how text is *phrased*, not just its topic. A short, informally
# worded question ("What's the SLA for Zone 1?") sits further away in
# embedding space from a formally worded policy paragraph ("Priority
# Overnight commits to delivery by 10:30 the next business day...") than two
# formally worded passages sit from each other — even when both are "about"
# the same thing. So instead of embedding the raw question, we first ask the
# LLM to hallucinate a plausible-sounding policy excerpt that WOULD answer it,
# written in the same register as our actual source docs, and embed *that*.
# It doesn't need to be factually correct — it only needs to be phrased like
# the real documents so its embedding lands close to theirs.
#
# This tends to matter most for terse or jargon-light questions (a user
# typing "hazmat rules?" instead of quoting policy language) or questions
# that mix vocabulary across documents in a way no single chunk mirrors
# closely — cases where the wording gap, not the topic, is why plain
# similarity search misses the right chunk.
HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write excerpts from internal last-mile delivery policy "
            "and operations documents.\n"
            "Given a question, write ONE short paragraph that would answer "
            "it, written exactly as if it were copied directly out of an "
            "internal policy or operations manual — confident, factual, "
            "specific (include plausible numbers, thresholds, or timeframes "
            "where relevant).\n"
            "Do not mention that this is hypothetical, an example, or a "
            "guess. Do not add commentary before or after the paragraph.",
        ),
        ("human", "{question}"),
    ]
)


def hyde_retrieve(question: str, k: int = 4) -> list[Document]:
    """Retrieve chunks using HyDE instead of embedding the raw question.

    Steps:
      1. Ask the LLM to write a hypothetical policy-doc paragraph that would
         answer the question (see HYDE_PROMPT comment above for why).
      2. Embed that paragraph (not the original question) and run similarity
         search against the Chroma store with it as the query.

    Returns the retrieved Documents directly (not a retriever) since this is
    meant for side-by-side comparison against get_retriever()'s output.
    """
    llm = ChatOllama(model="mistral", temperature=0)
    hyde_messages = HYDE_PROMPT.invoke({"question": question})
    hypothetical_doc = llm.invoke(hyde_messages).content

    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search(hypothetical_doc, k=k)


# ── Cross-encoder re-ranking ─────────────────────────────────────────────────
# Why a *second* model instead of just trusting the vector search?
#
# The embeddings used by get_retriever()/_get_vectorstore() are a BI-encoder:
# the question and each document chunk are embedded SEPARATELY, in complete
# isolation from each other, and compared afterward with a cheap similarity
# calculation (cosine distance) between two fixed vectors. That's exactly why
# it can search a whole corpus of thousands of chunks in milliseconds — the
# documents' vectors are precomputed once, ahead of time, and the query is the
# only thing embedded at request time.
#
# A CROSS-encoder instead feeds the (question, document) pair into the model
# TOGETHER, as one input, so every token of the question can directly attend
# to every token of the document (and vice versa) inside the model itself.
# That joint attention is what makes it much more precise at judging "does
# this chunk actually answer this question" — but it also means there is no
# such thing as precomputing a document's score ahead of time, since the
# score only exists relative to one specific question. Scoring is O(pairs),
# and each pair is a full forward pass through a transformer, so running it
# over an entire corpus would be far too slow for interactive use.
#
# The standard fix is the two-stage "retrieve-then-rerank" pattern used
# below: use the fast, imprecise bi-encoder to cheaply narrow thousands of
# chunks down to a small candidate pool (fetch_k), then spend the cross-
# encoder's precision only on that small pool to pick the best k. We get the
# bi-encoder's speed AND the cross-encoder's accuracy, on the part of the
# pipeline (final ranking) where accuracy matters most.
#
# Note: the first time this runs, sentence-transformers will download the
# "cross-encoder/ms-marco-MiniLM-L-6-v2" model (~80MB) from Hugging Face and
# cache it locally — later runs load it from disk.
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """Lazily load and cache the cross-encoder so repeated calls in the same
    process (e.g. looping over test questions in __main__) don't re-load the
    model from disk every time."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank_retrieve(question: str, k: int = 4, fetch_k: int = 15) -> list[Document]:
    """Retrieve via cheap similarity search, then re-rank with a cross-encoder.

    Stage 1 (broad, cheap): plain bi-encoder similarity search pulls fetch_k
    candidates — a wide enough net to likely contain the right chunks, even
    though their similarity-search ORDER isn't fully trustworthy.

    Stage 2 (narrow, precise): the cross-encoder scores each of those
    fetch_k candidates against the actual question, jointly, and we keep only
    the top k by that score. This is where a chunk that "sounds similar" but
    doesn't actually answer the question gets caught and demoted, and where
    a chunk that phrases things awkwardly but is directly on-point gets
    promoted.

    Each returned Document has its cross-encoder score attached as
    metadata["rerank_score"], purely so callers (like our __main__ comparison
    below) can print/inspect the ranking signal — it plays no further role in
    the RAG pipeline itself.
    """
    # Stage 1: cheap, broad candidate pool via the existing similarity path.
    candidates = get_retriever(k=fetch_k, search_type="similarity").invoke(question)

    # Stage 2: score every (question, chunk) pair jointly. predict() takes a
    # list of (query, text) tuples and returns one relevance score per pair —
    # higher means more relevant. Note this is fetch_k forward passes through
    # a transformer, done up front here, which is exactly the cost the
    # retrieve-then-rerank pattern exists to contain (fetch_k=15, not the
    # whole corpus).
    cross_encoder = _get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = cross_encoder.predict(pairs)

    # Attach scores as metadata, then sort candidates by score (descending)
    # and keep only the top k.
    for doc, score in zip(candidates, scores):
        doc.metadata["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda doc: doc.metadata["rerank_score"], reverse=True)
    return reranked[:k]


# ── Conversation-memory (query contextualization) ───────────────────────────
# In a single-turn RAG system, the question you retrieve with and the
# question you generate with are the same string. Multi-turn breaks that:
# a follow-up like "what about zone 2?" or "does that apply to Standard
# service too?" is only meaningful in light of what was just asked — on its
# own, embedded and searched against the vector store, it has almost no
# useful signal (there's no chunk in the KB that's "about" the word "that").
#
# The fix is to rewrite the follow-up into a fully standalone question
# BEFORE retrieval — e.g. "what about zone 2?" after a Zone 1 SLA question
# becomes "What is the SLA for Zone 2 priority deliveries?" — using the LLM
# and a short slice of recent chat history. That rewritten question is what
# actually goes into the vector store.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the recent chat history and a latest user question, "
            "rewrite the latest question as a standalone question that can "
            "be fully understood WITHOUT reading the chat history. Resolve "
            "pronouns, implicit references, and follow-up phrasing (e.g. "
            "'what about zone 2?', 'does that include weekends?') into "
            "explicit terms.\n"
            "Do NOT answer the question. Return ONLY the rewritten "
            "standalone question — no preamble, quotes, or explanation. If "
            "the latest question is already standalone, return it unchanged."
            "\n\nRecent chat history:\n{chat_history}",
        ),
        ("human", "{question}"),
    ]
)


def contextualize_question(chat_history: list[tuple[str, str]], question: str) -> str:
    """Rewrite `question` into a standalone form using recent chat history.

    chat_history is a list of (question, answer) tuples, oldest first.

    If chat_history is empty, the question is already standalone by
    definition (there's nothing prior for it to depend on) — we skip the LLM
    call entirely rather than spend a generation call confirming that.
    """
    if not chat_history:
        return question

    # Only the last 2-3 turns are used: distant history is rarely needed to
    # resolve what the LATEST question is referring to, and keeping this
    # rewrite prompt short keeps the extra LLM call fast.
    recent_turns = chat_history[-3:]
    history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent_turns)

    llm = ChatOllama(model="mistral", temperature=0)
    messages = CONTEXTUALIZE_PROMPT.invoke({"chat_history": history_text, "question": question})
    response = llm.invoke(messages)

    return response.content.strip()


def generate_answer(question: str, chat_history: list[tuple[str, str]] | None = None, k: int = 4) -> dict:
    """Run the full RAG pipeline: contextualize, retrieve, format, prompt, generate.

    chat_history is an optional list of (question, answer) tuples from
    earlier in the conversation, oldest first. Defaults to [] (a fresh
    conversation) when not provided.

    Returns a dict with:
      - "answer": the model's text response
      - "sources": sorted, de-duplicated list of source filenames the
        retrieved chunks came from (not necessarily all of which the model
        actually cited — this is "what was available", useful for debugging
        even when the model's own inline citations are incomplete)
      - "standalone_question": what contextualize_question() rewrote the
        question to — surfaced so the rewrite is visible/debuggable instead
        of happening invisibly inside the pipeline
    """
    if chat_history is None:
        chat_history = []

    # 1. Contextualize — resolve the raw question against recent history into
    #    something retrieval can act on independently of the conversation.
    standalone_question = contextualize_question(chat_history, question)

    # 2. Retrieve — using the STANDALONE question, not the raw one. The
    #    vector store and cross-encoder have no memory of the conversation;
    #    they only ever see one string. If we searched with "what about
    #    zone 2?" verbatim, retrieval would have nothing meaningful to match
    #    against. rerank_retrieve() (similarity + cross-encoder re-rank) is
    #    used here since it was the strongest-performing variant when we
    #    compared retrieval strategies earlier in Phase 2.
    retrieved_docs = rerank_retrieve(standalone_question, k=k)

    # 3. Format — turn those Documents into one context string
    context = format_context(retrieved_docs)

    # 4. Fill the template — using the ORIGINAL question, not the standalone
    #    one, as what the LLM sees as "the human's message". This keeps the
    #    reply feeling like a natural continuation of the conversation (the
    #    model responds to what the user actually typed) rather than an
    #    answer to a robotically-expanded restatement of it. This is safe
    #    specifically BECAUSE retrieval already resolved the reference —
    #    the right chunks are already in `context` by the time generation
    #    runs, so the model doesn't need the expanded phrasing to know what
    #    "that" or "zone 2" means.
    messages = RAG_PROMPT.invoke({"context": context, "question": question})

    # 5. Generate — send the filled prompt to a local Ollama model.
    #    temperature=0 makes the model always pick its highest-probability
    #    next token instead of sampling randomly. For RAG specifically we
    #    want the model to *report* what's in the retrieved context, not be
    #    creative — a higher temperature would let it phrase things more
    #    "freely" at the cost of being less faithful to the source text and
    #    less repeatable (same question could get different answers), which
    #    makes wrong answers harder to catch and debug.
    #
    #    Note chat_history is NOT passed into this prompt at all. Grounding
    #    must stay strict even in a multi-turn setting: a friendly,
    #    conversational TONE is fine and expected, but the actual facts in
    #    the answer must come only from `context` (this turn's retrieved
    #    chunks). If prior chat history were fed into this prompt as
    #    additional "context", the model could latch onto something it (or
    #    the user) said earlier — possibly wrong, outdated, or from a
    #    different question's retrieval — and repeat it as fact. Chat
    #    history's only job in this pipeline is upstream, in
    #    contextualize_question(), to figure out WHAT to retrieve — it never
    #    gets to influence WHAT THE ANSWER SAYS.
    llm = ChatOllama(model="mistral", temperature=0)
    response = llm.invoke(messages)

    # 6. Collect sources — pull the filename each retrieved chunk came from,
    #    de-duplicate (multiple chunks can share a source file), and sort
    #    for stable, readable output.
    sources = sorted({doc.metadata.get("source", "unknown") for doc in retrieved_docs})

    return {
        "answer": response.content,
        "sources": sources,
        "standalone_question": standalone_question,
    }


if __name__ == "__main__":
    print("Last-Mile Delivery RAG Assistant (type 'exit' to quit)\n")

    # chat_history accumulates as the conversation goes — each completed
    # turn is appended at the end of the loop body, so the NEXT call to
    # generate_answer() can use it to contextualize a follow-up question.
    chat_history: list[tuple[str, str]] = []

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        result = generate_answer(question, chat_history=chat_history)

        # Printing the standalone question makes the rewrite step visible —
        # useful for sanity-checking that follow-ups are being resolved
        # correctly instead of trusting it silently.
        if result["standalone_question"] != question:
            print(f"  (interpreted as: {result['standalone_question']!r})")

        print(f"\nAssistant: {result['answer']}")
        print("Sources:")
        for source in result["sources"]:
            print(f"  - {source}")
        print()

        chat_history.append((question, result["answer"]))
