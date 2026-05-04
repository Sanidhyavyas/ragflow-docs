"""
document_processor.py — Phase 1 + 3: Document Ingestion and Vector Storage

Pipeline:
  File (PDF / scanned PDF / image / TXT)
      → raw LangChain Documents  (auto-routed loader)
      → overlapping text chunks  (splitter)
      → 384-d float vectors      (sentence-transformer embedding)
      → persisted ChromaDB       (vector store)
      → top-k similarity search  (query)

Supported file types
--------------------
  .pdf   — text-based PDFs via PyPDFLoader; scanned PDFs auto-detected and
            routed to OCR via image_loader.load_scanned_pdf()
  .txt   — plain text via TextLoader
  .jpg / .jpeg / .png — image OCR via image_loader.load_image()
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import image_loader

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
    Auto-detect file type and route to the correct loader.

    Routing logic
    -------------
    .txt              → TextLoader  (plain UTF-8 text)
    .jpg/.jpeg/.png   → image_loader.load_image()  (Tesseract OCR)
    .pdf              → heuristic check via image_loader.is_likely_scanned_pdf()
                          True  → image_loader.load_scanned_pdf()  (OCR)
                          False → PyPDFLoader  (native text extraction)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return TextLoader(str(path), encoding="utf-8").load()

    if suffix in image_loader.IMAGE_EXTENSIONS:
        print(f"      Detected image file — using OCR loader")
        return image_loader.load_image(str(path))

    if suffix == ".pdf":
        if image_loader.is_likely_scanned_pdf(str(path)):
            print(f"      Detected scanned PDF — using OCR loader")
            return image_loader.load_scanned_pdf(str(path))
        else:
            print(f"      Detected text-based PDF — using PyPDF loader")
            return PyPDFLoader(str(path)).load()

    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        f"Supported: .pdf, .txt, .jpg, .jpeg, .png"
    )


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
        add_start_index=True,
    )

    all_chunks: List[Document] = []
    for fp in file_paths:
        path = Path(fp)
        print(f"[1/4] Loading   : {path.name}")
        try:
            raw_docs = _load_document(fp)
        except FileNotFoundError:
            print(f"      ERROR: File not found — {fp}")
            continue
        except ValueError as e:
            print(f"      ERROR: {e}")
            continue
        except Exception as e:
            print(f"      ERROR: Failed to load '{path.name}' — {e}")
            continue
        print(f"      OK     {len(raw_docs)} page(s) extracted")

        print(f"[2/4] Chunking  : splitting into ~{CHUNK_SIZE}-char chunks (overlap {CHUNK_OVERLAP})")
        chunks = splitter.split_documents(raw_docs)
        print(f"      OK     {len(chunks)} chunk(s) created")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "No content was extracted from any of the provided files. "
            "Check that the files exist, are non-empty, and are PDF or TXT."
        )

    print(f"\n[3/4] Embedding : loading model '{EMBEDDING_MODEL}' (first run downloads ~22 MB)")
    embeddings = _get_embeddings()
    print(f"      OK     model ready — embedding {len(all_chunks)} chunk(s) …")

    print(f"[4/4] Persisting: writing vectors to '{CHROMA_PERSIST_DIR}/'")
    vector_store = Chroma.from_documents(
        documents        = all_chunks,
        embedding        = embeddings,
        collection_name  = collection_name,
        persist_directory= CHROMA_PERSIST_DIR,
    )
    print(f"      OK     {len(all_chunks)} chunk(s) persisted to '{CHROMA_PERSIST_DIR}/'")
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
    vector_store  : Live Chroma instance from ingest(python rag_chain.py) or load_store().
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


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Ingest PDF/TXT files into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python document_processor.py --file report.pdf\n"
            "  python document_processor.py --file a.pdf b.txt --collection mycol"
        ),
    )
    parser.add_argument(
        "--file", "-f",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more PDF or TXT files to ingest",
    )
    parser.add_argument(
        "--collection", "-c",
        default="documents",
        help="ChromaDB collection name (default: documents)",
    )
    args = parser.parse_args()

    print("=" * 56)
    print("  Document Processor — Ingestion Pipeline")
    print(f"  Files      : {', '.join(args.file)}")
    print(f"  Collection : {args.collection}")
    print(f"  ChromaDB   : {CHROMA_PERSIST_DIR}/")
    print("=" * 56)
    print()

    try:
        store = ingest(args.file, collection_name=args.collection)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\nVerifying retrieval …")
    results = store.similarity_search("summary", k=1)
    print(f"  Retrieved {len(results)} chunk(s) — ChromaDB is working.")
    print("\nDone. You can now run: python rag_chain.py")
