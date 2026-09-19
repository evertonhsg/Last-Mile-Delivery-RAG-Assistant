"""
Phase 2 — Build a persistent Chroma vector store from the chunks produced in Phase 1.

Phase 1 wrote data/processed/chunks.json (a plain list of dicts: text + metadata).
This script's job is narrow: turn that JSON back into LangChain `Document` objects,
embed each one, and persist the result to disk with `langchain_chroma.Chroma` so
later phases (retrieval, RAG chain, UI) can just open the collection and query it
without re-embedding anything.

Run directly to (re)build the store and try a test query:
    python src/build_vectorstore.py
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Paths ───────────────────────────────────────────────────────────────────
CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_DIR = Path("data/chroma_db")            # where Chroma persists its files
COLLECTION_NAME = "lastmile_delivery_docs"

# Same embedding model used in Phase 1 — keeping it identical matters because
# vectors from different models aren't comparable to each other.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks_as_documents(chunks_path: Path) -> list[Document]:
    """Read chunks.json and wrap each chunk as a LangChain Document.

    A Document has two parts:
      - page_content: the actual chunk text (what gets embedded)
      - metadata: everything else (source file, chunk index) so we can later
        tell the user *where* an answer came from
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        documents.append(doc)

    return documents


def build_vectorstore() -> Chroma:
    """Embed all chunks and persist them to a Chroma collection on disk."""
    documents = load_chunks_as_documents(CHUNKS_PATH)
    print(f"Loaded {len(documents)} chunks from {CHUNKS_PATH}")

    # HuggingFaceEmbeddings wraps a sentence-transformers model behind
    # LangChain's standard `Embeddings` interface, so any vector store class
    # (Chroma, FAISS, ...) can call it the same way.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Chroma.from_documents does three things in one call:
    #   1. Embeds every document's page_content
    #   2. Writes vectors + text + metadata into a collection
    #   3. Persists everything to CHROMA_DIR so it survives after this process exits
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Persisted collection '{COLLECTION_NAME}' to {CHROMA_DIR}")
    print(f"Collection count: {vectorstore._collection.count()}")

    return vectorstore


if __name__ == "__main__":
    vectorstore = build_vectorstore()

    # Quick sanity check: does similarity search actually return relevant,
    # correctly-attributed chunks?
    test_query = "What is the SLA for Zone 1?"
    print(f"\nTest query: {test_query!r}")

    results = vectorstore.similarity_search(test_query, k=3)
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source")
        chunk_index = doc.metadata.get("chunk_index")
        print(f"\n--- Result {i} (source={source}, chunk_index={chunk_index}) ---")
        print(doc.page_content[:300])
