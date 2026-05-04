"""
rag_chain.py — Phase 2: LLM Query Layer

Full RAG pipeline:
  User query
      → top-k ChromaDB chunks        (document_processor.query)
      → structured prompt             (context injection + system rules)
      → Claude API call               (anthropic SDK)
      → cited answer                  ([Source: filename, chunk N] format)

Usage
-----
  # Programmatic
  from rag_chain import ask
  print(ask("What is RAG?"))

  # Interactive CLI
  python rag_chain.py
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # loads ANTHROPIC_API_KEY (and others) from .env into os.environ

from anthropic import Anthropic, APIError, AuthenticationError
from langchain_core.documents import Document

import document_processor as dp

# ── Configuration ─────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K        = 5
MAX_TOKENS   = 1024

# ── System prompt ──────────────────────────────────────────────────────────────
#
# Design choices:
#
# 1. "ONLY using information from the CONTEXT CHUNKS"
#    Hard constraint that prevents the model from drifting into parametric
#    (training-time) knowledge.  Without this, Claude may confidently answer
#    from its own memory even when the context is silent on the topic.
#
# 2. Mandatory citation format [Source: <file>, chunk <N>]
#    Ties every claim back to a retrievable chunk.  Users can verify answers
#    and engineers can debug retrieval quality by checking which chunks fired.
#
# 3. Explicit "I don't have enough information" fallback
#    Giving the model a scripted escape route is the most reliable way to
#    suppress hallucination.  If the model knows it is allowed to say it
#    doesn't know, it will do so rather than confabulate.
#
# 4. System vs. user message split
#    Rules go in the system prompt (evaluated once, high priority).
#    Context + question go in the user message (varies per request).
#    This is more token-efficient than embedding rules in every user turn.

SYSTEM_PROMPT = """\
You are a precise document-intelligence assistant.

Rules you MUST follow:
1. Answer ONLY using information from the CONTEXT CHUNKS provided in the user message.
2. Cite every factual claim using the format  [Source: <filename>, chunk <N>].
3. If the context does not contain sufficient information to answer, respond with exactly:
   "I don't have enough information in the provided documents to answer this question."
   Do NOT speculate, infer beyond what is written, or use outside knowledge.
4. If multiple chunks support a point, cite all relevant ones.
5. Be concise and structured. Use bullet points for multi-part answers.\
"""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_user_message(question: str, chunks: list[Document]) -> str:
    """
    Inject retrieved chunks as numbered, labelled context blocks.

    Format rationale:
    - Explicit "CONTEXT CHUNKS" header makes the document boundary clear.
    - Numbering gives the model an unambiguous reference for citations.
    - Source + page metadata is embedded per-chunk so Claude can cite accurately.
    - The question is placed AFTER the context so the model reads evidence first,
      which measurably improves faithfulness on factual tasks.
    """
    blocks = []
    for i, doc in enumerate(chunks, start=1):
        source   = doc.metadata.get("source", "unknown")
        page     = doc.metadata.get("page", "")
        page_str = f", page {page}" if page != "" else ""
        blocks.append(
            f"[Chunk {i} | Source: {source}{page_str}]\n{doc.page_content.strip()}"
        )

    context_section = "\n\n".join(blocks)
    divider         = "─" * 60

    return (
        f"CONTEXT CHUNKS:\n"
        f"{divider}\n"
        f"{context_section}\n"
        f"{divider}\n\n"
        f"QUESTION: {question}"
    )


# ── Core RAG function ──────────────────────────────────────────────────────────

def ask(
    question: str,
    *,
    api_key: Optional[str] = None,
    vector_store=None,
    k: int = TOP_K,
    collection_name: str = "documents",
    verbose: bool = False,
) -> str:
    """
    Full RAG pipeline: retrieve → prompt → generate → return cited answer.

    Parameters
    ----------
    question        : Natural-language question.
    api_key         : Anthropic API key.  Falls back to ANTHROPIC_API_KEY env var.
    vector_store    : Live Chroma instance.  If None, loads from disk automatically.
    k               : Number of chunks to retrieve (default 5).
    collection_name : ChromaDB collection to search.
    verbose         : Print retrieved chunks before the LLM answer.

    Returns
    -------
    String answer with inline [Source: ...] citations, or a "not enough
    information" message if the retrieved context is empty or off-topic.

    Raises
    ------
    EnvironmentError  : Missing or invalid API key.
    RuntimeError      : Anthropic API / network error.
    """
    # ── Step 1: Retrieve ───────────────────────────────────────────────────────
    chunks = dp.query(question, vector_store=vector_store, k=k,
                      collection_name=collection_name)

    if not chunks:
        return (
            "No relevant documents were found in the knowledge base. "
            "Please ingest documents first using document_processor.ingest()."
        )

    if verbose:
        print(f"\n── Retrieved {len(chunks)} chunk(s) ──")
        for i, doc in enumerate(chunks, 1):
            snippet = doc.page_content[:90].replace("\n", " ")
            print(f"  [{i}] {doc.metadata.get('source', '?')} | {snippet} …")
        print()

    # ── Step 2: Build prompt ───────────────────────────────────────────────────
    user_message = _build_user_message(question, chunks)

    # ── Step 3: Resolve API key ────────────────────────────────────────────────
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "Anthropic API key not found.\n"
            "  Option A: set the environment variable  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Option B: pass  api_key='sk-ant-...'  to ask()"
        )

    # ── Step 4: Call Claude ────────────────────────────────────────────────────
    client = Anthropic(api_key=resolved_key)

    try:
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = MAX_TOKENS,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )
    except AuthenticationError:
        raise EnvironmentError(
            "Anthropic API key rejected. Verify your ANTHROPIC_API_KEY value."
        )
    except APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")

    return response.content[0].text


# ── CLI interface ──────────────────────────────────────────────────────────────

def _cli() -> None:
    """
    Interactive REPL for testing RAG queries end-to-end.
    Loads ChromaDB once at startup to avoid re-embedding on every query.

    Commands:
      quit / exit  — stop the session
      verbose      — toggle chunk display (default off)
    """
    print("=" * 60)
    print("  Multi-Modal RAG — Phase 2 CLI")
    print(f"  Model  : {CLAUDE_MODEL}")
    print(f"  Top-K  : {TOP_K} chunks per query")
    print("  Cmds   : 'verbose' to toggle chunk view | 'quit' to exit")
    print("=" * 60)

    # Pre-load vector store once so every query is fast
    try:
        store = dp.load_store()
        print("ChromaDB loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    verbose = False

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue

        lower = question.lower()

        if lower in ("quit", "exit"):
            print("Goodbye.")
            break

        if lower == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}\n")
            continue

        try:
            answer = ask(question, vector_store=store, verbose=verbose)
            print(f"\nAssistant: {answer}\n")
        except EnvironmentError as e:
            print(f"\nCONFIG ERROR: {e}\n")
            break
        except RuntimeError as e:
            print(f"\nAPI ERROR: {e}\n")


if __name__ == "__main__":
    _cli()
