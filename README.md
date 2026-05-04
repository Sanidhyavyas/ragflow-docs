# ragflow-docs

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange?style=flat)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=flat)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

A **Multi-Modal Retrieval-Augmented Generation (RAG)** system for document intelligence. Upload PDFs, scanned documents, images, or plain text — ask natural-language questions and get precise, cited answers powered by a free LLaMA 3.3 70B model via the Groq API.

Built in four focused phases: document ingestion, RAG chain, OCR support, and a Streamlit web UI.

---

## Demo

> _Screenshot placeholder — add a screenshot of the Streamlit UI here._
>
> ![Demo Screenshot](assets/demo.png)

---

## Features

- **Multi-format ingestion** — PDF (text-based and scanned), PNG, JPG, JPEG, and TXT files
- **OCR pipeline** — Tesseract + pdf2image extracts text from images and scanned PDFs automatically
- **Semantic search** — sentence-transformers embeddings stored and queried in a local ChromaDB instance
- **Cited answers** — every LLM response includes `[Source: filename, chunk N]` citations
- **Streamlit chat UI** — upload files, ingest into the knowledge base, and chat in the browser
- **Raw chunk viewer** — expandable toggle to inspect the exact retrieved context passed to the LLM
- **CLI mode** — run queries directly from the terminal without the UI
- **Fully local embeddings** — `all-MiniLM-L6-v2` runs on CPU, no embedding API key needed
- **Persistent knowledge base** — ChromaDB persists to disk; ingested files survive restarts

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Document loading & splitting | LangChain (`PyPDFLoader`, `TextLoader`, `RecursiveCharacterTextSplitter`) |
| Vector database | ChromaDB (local, SQLite-backed) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d, CPU) |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| OCR | Tesseract + `pytesseract` + `pdf2image` + Pillow |
| Web UI | Streamlit |
| Environment | `python-dotenv` |

---

## Project Structure

```
ragflow-docs/
├── streamlit_app.py        # Phase 4 — Streamlit web UI
├── rag_chain.py            # Phase 2 — Groq LLM + retrieval + citations
├── document_processor.py   # Phase 1 — Ingestion pipeline (load → chunk → embed → store)
├── image_loader.py         # Phase 3 — Tesseract OCR for images and scanned PDFs
├── test_rag.py             # Integration tests for the RAG pipeline
├── test_ocr.py             # Tests for the OCR loader
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── chroma_db/              # Persisted vector store (auto-created on first ingest)
└── uploads/                # Uploaded files saved here by the Streamlit UI
```

---

## Installation

### Prerequisites

**1. Python 3.11**

Download from [python.org](https://www.python.org/downloads/).

**2. Tesseract OCR** _(required for image and scanned PDF support)_

- Download the Windows installer from the [UB Mannheim release page](https://github.com/UB-Mannheim/tesseract/wiki)
- Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Verify: `tesseract --version`

**3. Poppler** _(required for PDF-to-image conversion)_

- Download the latest release from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
- Extract and add the `bin/` folder to your system PATH, **or** set `POPPLER_PATH` in your `.env`
- Verify: `pdftoppm -v`

**4. Groq API Key** _(free)_

Sign up at [console.groq.com](https://console.groq.com) and generate an API key.

---

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Sanidhyavyas/ragflow-docs.git
cd ragflow-docs

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# Then edit .env with your values (see section below)
```

---

## Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```env
# Required — get a free key at https://console.groq.com
GROQ_API_KEY=gsk_...

# Optional — set if Tesseract is not on PATH
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Optional — set if Poppler bin/ is not on PATH
POPPLER_PATH=C:\poppler\Library\bin
```

> **Never commit your `.env` file.** It is already listed in `.gitignore`.

---

## Usage

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. Use the **sidebar** to upload PDF, PNG, JPG, or TXT files
2. Click **Ingest Files** to embed and store them in ChromaDB
3. Type a question in the chat box and press Enter
4. The answer appears with `[Source: filename, chunk N]` citations
5. Expand **View raw retrieved chunks** to inspect the context

### CLI Mode

```python
from document_processor import ingest
from rag_chain import ask

# Ingest documents
ingest(["report.pdf", "diagram.png", "notes.txt"])

# Ask a question
answer = ask("What are the main findings in the report?")
print(answer)
```

Or run the interactive REPL:

```bash
python rag_chain.py
```

### Ingest from the command line

```bash
python document_processor.py ingest report.pdf diagram.png notes.txt
```

---

## How It Works

```
User query
    │
    ▼
ChromaDB similarity search   ←   Uploaded files
    │                              │
    │                        load → chunk → embed → store
    ▼
Top-K context chunks
    │
    ▼
Groq LLM (LLaMA 3.3 70B)
    │
    ▼
Cited answer [Source: file, chunk N]
```

1. **Ingest** — files are loaded (OCR if needed), split into 1000-character overlapping chunks, embedded with `all-MiniLM-L6-v2`, and persisted in ChromaDB.
2. **Retrieve** — the user's question is embedded and the top-K most similar chunks are fetched via cosine similarity.
3. **Generate** — chunks are injected into a structured prompt with strict citation rules and sent to the Groq API.
4. **Cite** — the model is constrained to answer only from the provided context and cite every claim.

---

## Roadmap

- [ ] Multi-collection support (separate knowledge bases per project)
- [ ] Hybrid search — BM25 keyword search combined with semantic search
- [ ] Re-ranking with a cross-encoder for higher retrieval precision
- [ ] Conversation memory — multi-turn chat with history-aware retrieval
- [ ] Docker containerization for one-command deployment
- [ ] Support for DOCX, PPTX, and HTML document formats
- [ ] Evaluation dashboard — precision@K and faithfulness scoring

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with LangChain · ChromaDB · Groq · Streamlit</p>
