"""
test_rag.py — Phase 1 smoke test

Creates a synthetic text document, ingests it into ChromaDB,
then runs several queries to verify retrieval is working.

Usage:
    python test_rag.py
"""

from pathlib import Path
import document_processor as dp

# ── 1. Write a sample document ────────────────────────────────────────────────

SAMPLE_FILE = "sample_doc.txt"

SAMPLE_TEXT = """\
Retrieval-Augmented Generation (RAG) is an AI framework that improves large
language model (LLM) outputs by grounding them in an external knowledge base.

How RAG works
=============
1. The user submits a natural-language query.
2. The retriever embeds the query and performs semantic search over a vector
   database to find the most relevant document chunks.
3. The retrieved chunks are injected into the LLM prompt as additional context.
4. The LLM generates a factual, grounded answer based on that context.

RAG reduces hallucinations because the model can cite retrieved passages
instead of relying solely on its parametric (training-time) memory.

ChromaDB
========
ChromaDB is an open-source embedding database that stores document vectors
locally on disk.  It uses an HNSW (Hierarchical Navigable Small World) index
for fast approximate nearest-neighbour (ANN) search.  No internet connection
or API key is required — everything runs in-process.

Sentence-Transformers
=====================
Sentence-transformers provide lightweight, CPU-friendly embedding models.
The all-MiniLM-L6-v2 model maps text to a 384-dimensional vector space and
runs entirely offline.  It is a good default for English-language semantic
search tasks and can process thousands of sentences per second on a CPU.

LangChain
=========
LangChain is a framework for building LLM-powered applications.  It provides
standardised interfaces for document loaders, text splitters, embedding models,
vector stores, and LLM chains — making it easy to swap out individual
components without rewriting the rest of the pipeline.

Text Splitting Strategy
=======================
RecursiveCharacterTextSplitter breaks documents into overlapping chunks.
It tries to split on paragraph breaks first (\\n\\n), then sentence breaks (\\n),
then word boundaries, before resorting to hard character cuts.  An overlap of
200 characters ensures that sentences spanning a chunk boundary are present in
both neighbouring chunks, preventing context loss during retrieval.
"""

Path(SAMPLE_FILE).write_text(SAMPLE_TEXT, encoding="utf-8")
print(f"[1/3] Created sample document: {SAMPLE_FILE}")

# ── 2. Ingest ─────────────────────────────────────────────────────────────────

print("\n[2/3] Ingesting into ChromaDB …")
print("-" * 50)
store = dp.ingest([SAMPLE_FILE])

# To test with a real PDF, replace the list above, e.g.:
#   store = dp.ingest(["my_paper.pdf", "notes.txt"])

# ── 3. Query ──────────────────────────────────────────────────────────────────

print("\n[3/3] Running test queries …")
print("=" * 50)

TEST_QUERIES = [
    "How does RAG reduce hallucinations?",
    "What embedding model is used and what are its dimensions?",
    "How does ChromaDB store and search vectors?",
    "What is the text splitting strategy and why is overlap important?",
]

for question in TEST_QUERIES:
    print(f"\nQ: {question}")
    results = dp.query(question, vector_store=store, k=2)
    for rank, doc in enumerate(results, start=1):
        source  = doc.metadata.get("source", "n/a")
        snippet = doc.page_content[:220].replace("\n", " ").strip()
        print(f"  [{rank}] source={source!r}")
        print(f"       {snippet} …")

print("\n" + "=" * 50)
print("Phase 1 complete.")
print(f"ChromaDB is persisted at: {dp.CHROMA_PERSIST_DIR}/")
print()
print("Next steps:")
print("  - Call dp.load_store() in future sessions to skip re-ingestion")
print("  - Phase 2: wrap dp.query() in a LangChain RAG chain with an LLM")
