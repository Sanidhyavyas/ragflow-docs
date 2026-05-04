"""
document_processor.py — Phase 1: Document Ingestion and Vector Storage

Pipeline:
  File (PDF/TXT)
      → raw LangChain Documents  (loader)
      → overlapping text chunks  (splitter)
      → 384-d float vectors      (sentence-transformer embedding)
      → persisted ChromaDB       (vector store)
      → top-k similarity search  (query)
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = "./chroma_db"          # ChromaDB writes here automatically
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE         = 1000                   # characters per chunk
CHUNK_OVERLAP      = 200                    # overlap between adjacent chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformer embedding model.

    all-MiniLM-L6-v2:
      - 384-dimensional output vectors
      - Runs entirely on CPU — no GPU or API key needed
      - ~22 MB download (cached by HuggingFace after first use)
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
    )


def _load_document(file_path: str) -> List[Document]:
    """
    Dispatch to the correct LangChain loader based on file extension.

    PyPDFLoader  — parses each PDF page as a separate Document object;
                   metadata includes {"source": path, "page": 0-indexed int}
    TextLoader   — reads the whole file as a single Document object;
                   metadata includes {"source": path}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: .pdf, .txt"
        )

    return loader.load()


# ── Public API ────────────────────────────────────────────────────────────────

def ingest(
    file_paths: List[str],
    collection_name: str = "documents",
) -> Chroma:
    """
    Load, split, embed, and persist one or more PDF/TXT files.

    Steps
    -----
    1. Load  — each file becomes a list of LangChain `Document` objects.
               PDFs yield one Document per page; TXT files yield one Document.

    2. Split — RecursiveCharacterTextSplitter tries to break on paragraphs
               (\\n\\n), then sentences (\\n), then words before resorting to
               hard character cuts.  chunk_size=1000 / overlap=200 is a solid
               default for most document RAG use cases.

    3. Embed — HuggingFaceEmbeddings calls the sentence-transformer locally
               and converts each chunk into a 384-d float vector.

    4. Store — Chroma.from_documents() stores (text + vector + metadata) in
               a persistent SQLite-backed directory under CHROMA_PERSIST_DIR.
               No manual `.persist()` call needed (ChromaDB ≥ 0.4 auto-saves).

    Parameters
    ----------
    file_paths      : List of absolute or relative paths to .pdf / .txt files.
    collection_name : ChromaDB collection to write into.  Calling ingest()
                      twice with the same collection appends (may duplicate).

    Returns
    -------
    The live Chroma vector-store instance (ready for immediate querying).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,   # adds {"start_index": N} to each chunk's metadata
    )

    all_chunks: List[Document] = []
    for fp in file_paths:
        raw_docs = _load_document(fp)
        chunks   = splitter.split_documents(raw_docs)
        print(f"  {Path(fp).name}: {len(raw_docs)} page(s) → {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No content found — check that your files are non-empty.")

    print(f"\nEmbedding {len(all_chunks)} chunk(s) with {EMBEDDING_MODEL} …")
    embeddings   = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents        = all_chunks,
        embedding        = embeddings,
        collection_name  = collection_name,
        persist_directory= CHROMA_PERSIST_DIR,
    )

    print(f"Persisted {len(all_chunks)} chunk(s) → '{CHROMA_PERSIST_DIR}/'")
    return vector_store


def load_store(collection_name: str = "documents") -> Chroma:
    """
    Re-attach to an existing ChromaDB collection without re-ingesting.

    Use this after the first ingest() run so you can query across sessions
    without paying the embedding cost again.
    """
    if not Path(CHROMA_PERSIST_DIR).exists():
        raise FileNotFoundError(
            f"No ChromaDB found at '{CHROMA_PERSIST_DIR}'. Run ingest() first."
        )
    embeddings = _get_embeddings()
    return Chroma(
        collection_name  = collection_name,
        embedding_function = embeddings,
        persist_directory  = CHROMA_PERSIST_DIR,
    )


def query(
    question: str,
    vector_store: Optional[Chroma] = None,
    k: int = 4,
    collection_name: str = "documents",
) -> List[Document]:
    """
    Semantic similarity search: return the top-k most relevant chunks.

    Internally ChromaDB computes cosine similarity between the question
    embedding and every stored chunk vector, then returns the k nearest
    neighbours via an HNSW index (fast approximate search).

    Parameters
    ----------
    question      : Natural-language query string.
    vector_store  : Live Chroma instance from ingest() or load_store().
                    If None, loads from disk automatically.
    k             : Number of chunks to return (default 4).
    collection_name: Which ChromaDB collection to search.

    Returns
    -------
    List of Document objects ordered by relevance (most similar first).
    Each Document has:
      .page_content  — the raw text of the chunk
      .metadata      — {"source": path, "page": N, "start_index": N, …}
    """
    if vector_store is None:
        vector_store = load_store(collection_name)

    return vector_store.similarity_search(question, k=k)
