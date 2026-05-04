"""
test_ocr.py — Phase 3 smoke test

Creates a synthetic PNG image with printed text using Pillow,
runs it through the full OCR → chunk → embed → ChromaDB pipeline,
then queries to verify retrieval works end-to-end.

Usage:
    python test_ocr.py
"""

import sys
from pathlib import Path

# ── 1. Generate a sample PNG with text ────────────────────────────────────────

SAMPLE_IMAGE = "sample_ocr_test.png"

def create_sample_image() -> None:
    """Draw white text on a black background using Pillow (no Tesseract needed here)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow not installed.  Run: pip install Pillow")
        sys.exit(1)

    width, height = 800, 400
    img  = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use default bitmap font — always available, no extra files needed
    text = (
        "OCR Test Document\n\n"
        "This image was generated to test the Phase 3 OCR pipeline.\n\n"
        "Tesseract should extract this text correctly.\n\n"
        "The RAG system can then answer questions about scanned documents.\n\n"
        "Sample fact: The Eiffel Tower is located in Paris, France."
    )

    draw.multiline_text((40, 40), text, fill=(0, 0, 0), spacing=8)
    img.save(SAMPLE_IMAGE)
    print(f"[1/4] Created sample image: {SAMPLE_IMAGE}")


# ── 2. Verify Tesseract is reachable ──────────────────────────────────────────

def check_tesseract() -> None:
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"[2/4] Tesseract found — version {version}")
    except Exception as e:
        print(
            "\nERROR: Tesseract not found or not configured.\n"
            "  1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  2. Install it (default: C:\\Program Files\\Tesseract-OCR\\)\n"
            "  3. Add to PATH or set TESSERACT_CMD in your .env file:\n"
            "       TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
            f"\nDetails: {e}"
        )
        sys.exit(1)


# ── 3. OCR the image ──────────────────────────────────────────────────────────

def ocr_image() -> None:
    from image_loader import load_image
    docs = load_image(SAMPLE_IMAGE)
    if not docs:
        print("ERROR: OCR returned no text. Check that the image is readable.")
        sys.exit(1)
    print(f"[3/4] OCR extracted {len(docs)} document(s)")
    snippet = docs[0].page_content[:120].replace("\n", " ")
    print(f"      Preview: {snippet} …")


# ── 4. Full pipeline: ingest → query ──────────────────────────────────────────

def ingest_and_query() -> None:
    import document_processor as dp

    print("[4/4] Ingesting image into ChromaDB …")
    store = dp.ingest([SAMPLE_IMAGE], collection_name="ocr_test")

    print("\nQuerying: 'Where is the Eiffel Tower?'")
    results = dp.query(
        "Where is the Eiffel Tower?",
        vector_store=store,
        k=1,
        collection_name="ocr_test",
    )

    if results:
        print(f"  Answer chunk: {results[0].page_content[:150].strip()}")
        print("\nPhase 3 OCR pipeline is working correctly.")
    else:
        print("  No results returned — check embedding and ChromaDB setup.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  Phase 3 OCR Smoke Test")
    print("=" * 56)
    print()
    create_sample_image()
    check_tesseract()
    ocr_image()
    ingest_and_query()
