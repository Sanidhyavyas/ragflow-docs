"""
streamlit_app.py — Phase 4: Multi-Modal RAG Chat Interface

Features
--------
- Sidebar file uploader  : PDF, PNG, JPG, TXT → auto-ingested into ChromaDB
- Chat UI                : st.chat_message() with full message history
- Cited answers          : source citations from rag_chain.ask()
- Raw chunks toggle      : expandable view of retrieved context chunks
- Ingested files list    : live sidebar list of what's in the knowledge base
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env

import document_processor as dp
import rag_chain

# ── Constants ──────────────────────────────────────────────────────────────────

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Modal RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Session state bootstrap ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "chunks": list|None}
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    # Try to reconnect to an existing ChromaDB on disk
    try:
        st.session_state.vector_store = dp.load_store()
    except FileNotFoundError:
        st.session_state.vector_store = None

if "ingested_files" not in st.session_state:
    # Recover file list from ChromaDB metadata on startup
    ingested: list[str] = []
    if st.session_state.vector_store is not None:
        try:
            result = st.session_state.vector_store._collection.get(
                include=["metadatas"]
            )
            seen: set[str] = set()
            for meta in result.get("metadatas", []):
                src = meta.get("source", "")
                name = Path(src).name
                if name and name not in seen:
                    seen.add(name)
                    ingested.append(name)
        except Exception:
            pass
    st.session_state.ingested_files = ingested

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📁 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload files to ingest",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, PNG, JPG, JPEG, TXT",
    )

    top_k = st.slider(
        "Chunks to retrieve (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of context chunks passed to the LLM",
    )

    ingest_btn = st.button(
        "⬆️ Ingest Files",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_btn and uploaded_files:
        new_paths: list[str] = []
        new_names: list[str] = []

        for uf in uploaded_files:
            dest = UPLOAD_DIR / uf.name
            dest.write_bytes(uf.getbuffer())  # overwrite to keep files fresh
            if uf.name not in st.session_state.ingested_files:
                new_paths.append(str(dest))
                new_names.append(uf.name)

        if not new_paths:
            st.info("All uploaded files are already in the knowledge base.")
        else:
            with st.spinner(f"Ingesting {len(new_paths)} file(s)…"):
                try:
                    vs = dp.ingest(new_paths)
                    st.session_state.vector_store = vs
                    st.session_state.ingested_files.extend(new_names)
                    st.success(f"✅ {len(new_paths)} file(s) ingested.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.divider()

    st.subheader("📚 Knowledge Base")
    if st.session_state.ingested_files:
        for fname in st.session_state.ingested_files:
            st.markdown(f"- `{fname}`")
    else:
        st.caption("No files ingested yet. Upload files above.")

    st.divider()
    api_key_status = "✅ Loaded" if os.environ.get("GROQ_API_KEY") else "❌ Missing"
    st.caption(f"GROQ_API_KEY: {api_key_status}")
    st.caption(f"Model: `{rag_chain.GROQ_MODEL}`")

# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("💬 Multi-Modal RAG Chat")

if not st.session_state.ingested_files:
    st.info(
        "Upload and ingest documents using the sidebar to start chatting.",
        icon="👈",
    )

# Render existing message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("🔍 View raw retrieved chunks"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    source   = chunk.metadata.get("source", "unknown")
                    page     = chunk.metadata.get("page", "")
                    page_str = f" · page {page}" if page != "" else ""
                    st.markdown(
                        f"**Chunk {i}** — `{Path(source).name}{page_str}`"
                    )
                    st.text(chunk.page_content.strip())
                    if i < len(msg["chunks"]):
                        st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    # Append and render user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "chunks": None}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and render assistant response
    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            answer = (
                "No documents are loaded yet. "
                "Please upload and ingest files using the sidebar first."
            )
            chunks = []
            st.markdown(answer)
        else:
            with st.spinner("Retrieving and reasoning…"):
                try:
                    # Retrieve chunks (for the raw-chunks toggle)
                    chunks = dp.query(
                        prompt,
                        vector_store=st.session_state.vector_store,
                        k=top_k,
                    )
                    # Full RAG: retrieve → prompt → Groq LLM → cited answer
                    answer = rag_chain.ask(
                        prompt,
                        vector_store=st.session_state.vector_store,
                        k=top_k,
                    )
                except EnvironmentError as exc:
                    answer = f"**API Key Error:** {exc}"
                    chunks = []
                except RuntimeError as exc:
                    answer = f"**LLM Error:** {exc}"
                    chunks = []
                except Exception as exc:
                    answer = f"**Unexpected error:** {exc}"
                    chunks = []

            st.markdown(answer)

            if chunks:
                with st.expander("🔍 View raw retrieved chunks"):
                    for i, chunk in enumerate(chunks, 1):
                        source   = chunk.metadata.get("source", "unknown")
                        page     = chunk.metadata.get("page", "")
                        page_str = f" · page {page}" if page != "" else ""
                        st.markdown(
                            f"**Chunk {i}** — `{Path(source).name}{page_str}`"
                        )
                        st.text(chunk.page_content.strip())
                        if i < len(chunks):
                            st.divider()

    # Persist to session history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )
