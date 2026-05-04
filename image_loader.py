"""
image_loader.py — Phase 3: OCR-based Document Loading

Handles two input types that PyPDFLoader cannot read:

  1. Scanned PDFs  — each page is a raster image with no embedded text layer.
                     pdf2image converts each page to a PIL image, then
                     pytesseract runs Tesseract OCR to extract text.

  2. Image files   — .jpg / .jpeg / .png loaded directly with PIL, then OCR'd.

Both return a list of LangChain Document objects (same contract as PyPDFLoader)
so document_processor.py can treat them identically.

Tesseract installation (Windows — required before using this module)
--------------------------------------------------------------------
  1. Download the installer from:
       https://github.com/UB-Mannheim/tesseract/wiki
       (choose "tesseract-ocr-w64-setup-*.exe" for 64-bit Windows)

  2. Run the installer.  Default path:
       C:\\Program Files\\Tesseract-OCR\\tesseract.exe

  3. Add Tesseract to PATH, OR set the path explicitly in your .env:
       TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe

  4. Verify installation:
       tesseract --version

Poppler installation (Windows — required for pdf2image)
-------------------------------------------------------
  1. Download from:
       https://github.com/oschwartz10612/poppler-windows/releases
       (choose the latest "Release-*.zip")

  2. Extract to a folder, e.g.:
       C:\\poppler\\Library\\bin

  3. Add that folder to PATH, OR set it in your .env:
       POPPLER_PATH=C:\\poppler\\Library\\bin

  4. Verify:
       pdftoppm -v
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# ── Optional Tesseract path from .env ─────────────────────────────────────────
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
_POPPLER_PATH  = os.environ.get("POPPLER_PATH") or None

# ── Lazy imports (so import errors are clear at call-time, not at module load) ─
try:
    import pytesseract
    if _TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
except ImportError as _e:
    raise ImportError(
        "pytesseract is not installed.  Run:  pip install pytesseract"
    ) from _e

try:
    from PIL import Image
except ImportError as _e:
    raise ImportError(
        "Pillow is not installed.  Run:  pip install Pillow"
    ) from _e

try:
    from pdf2image import convert_from_path
except ImportError as _e:
    raise ImportError(
        "pdf2image is not installed.  Run:  pip install pdf2image"
    ) from _e

from langchain_core.documents import Document


# ── Supported extensions ───────────────────────────────────────────────────────

IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png"}
SCANNED_PDF_EXT   = ".pdf"   # used only when routed here explicitly


# ── Core OCR helpers ───────────────────────────────────────────────────────────

def _ocr_image(image: "Image.Image") -> str:
    """Run Tesseract on a single PIL image and return extracted text."""
    text = pytesseract.image_to_string(image, lang="eng")
    return text.strip()


def _ocr_pdf_pages(file_path: str) -> List[str]:
    """
    Convert each PDF page to an image and OCR it.

    Returns a list of strings, one per page (empty string if a page is blank).

    pdf2image uses Poppler's pdftoppm under the hood.  DPI=300 gives a good
    balance between accuracy and speed for typical letter/A4 documents.
    """
    kwargs = {"dpi": 300}
    if _POPPLER_PATH:
        kwargs["poppler_path"] = _POPPLER_PATH

    images = convert_from_path(file_path, **kwargs)
    return [_ocr_image(img) for img in images]


# ── Public API ─────────────────────────────────────────────────────────────────

def load_scanned_pdf(file_path: str) -> List[Document]:
    """
    OCR a scanned PDF and return one Document per page.

    Metadata mirrors PyPDFLoader:
      {"source": file_path, "page": 0-indexed int, "loader": "ocr"}

    Parameters
    ----------
    file_path : Absolute or relative path to a PDF file.

    Returns
    -------
    List of Document objects, one per page with extracted text as page_content.
    Pages that produce no text are skipped.

    Raises
    ------
    FileNotFoundError : file_path does not exist.
    RuntimeError      : Tesseract or Poppler not found / OCR failed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"      OCR-ing scanned PDF (DPI=300) — this may take a moment …")
    try:
        page_texts = _ocr_pdf_pages(str(path))
    except Exception as e:
        raise RuntimeError(
            f"OCR failed for '{path.name}'.\n"
            f"  Make sure Tesseract and Poppler are installed.\n"
            f"  Details: {e}"
        ) from e

    docs = []
    for i, text in enumerate(page_texts):
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": str(path), "page": i, "loader": "ocr"},
        ))

    return docs


def load_image(file_path: str) -> List[Document]:
    """
    OCR a single image file (.jpg / .jpeg / .png) and return one Document.

    Metadata:
      {"source": file_path, "page": 0, "loader": "ocr"}

    Parameters
    ----------
    file_path : Absolute or relative path to an image file.

    Returns
    -------
    A list containing a single Document with OCR'd text.

    Raises
    ------
    FileNotFoundError : file_path does not exist.
    ValueError        : Unsupported image format.
    RuntimeError      : Tesseract not found / OCR failed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {', '.join(IMAGE_EXTENSIONS)}"
        )

    print(f"      OCR-ing image: {path.name} …")
    try:
        image = Image.open(str(path))
        text  = _ocr_image(image)
    except Exception as e:
        raise RuntimeError(
            f"OCR failed for '{path.name}'.\n"
            f"  Make sure Tesseract is installed and TESSERACT_CMD is set if needed.\n"
            f"  Details: {e}"
        ) from e

    if not text:
        return []

    return [Document(
        page_content=text,
        metadata={"source": str(path), "page": 0, "loader": "ocr"},
    )]


def is_likely_scanned_pdf(file_path: str, sample_pages: int = 3) -> bool:
    """
    Heuristic: try to extract text with pypdf from the first N pages.
    If all sampled pages have fewer than 20 characters, treat the PDF as scanned.

    Used by document_processor._load_document() to auto-detect which loader to use.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_to_check = min(sample_pages, len(reader.pages))
        for i in range(pages_to_check):
            text = reader.pages[i].extract_text() or ""
            if len(text.strip()) >= 20:
                return False   # found readable text → not scanned
        return True            # all sampled pages were empty → treat as scanned
    except Exception:
        return False           # if in doubt, fall back to regular PDF loader
