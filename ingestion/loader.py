import fitz  # PyMuPDF
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data" / "papers"


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF and extracts text page by page.
    Returns a list of dicts — one per page — with text + metadata.
    
    Why page by page?
    - Preserves page number for citation (important in legal domain)
    - Lets us skip blank/header-only pages
    - Easier to debug when something goes wrong
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip pages with no meaningful content
        # Legal PDFs often have blank pages or cover pages
        if len(text.strip()) < 50:
            continue

        pages.append({
            "text": text.strip(),
            "source": Path(pdf_path).name,
            "page": page_num + 1,  # 1-indexed, humans don't say "page 0"
        })

    doc.close()
    return pages


def load_all_pdfs(pdf_dir: str) -> list[dict]:
    """
    Loads every PDF in a directory.
    Returns a flat list of all pages across all documents.
    """
    pdf_dir = Path(pdf_dir)
    print(f"Looking for PDFs in: {pdf_dir.resolve()}")
    print(f"Exists: {pdf_dir.exists()}")
    print(f"Files found: {len(list(pdf_dir.glob('*.pdf')))}")
    all_pages = []

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        try:
            pages = extract_text_from_pdf(str(pdf_file))
            all_pages.extend(pages)
            print(f"  → {len(pages)} pages extracted")
        except Exception as e:
            # Don't crash the whole pipeline if one PDF is corrupted
            print(f"  → FAILED: {e} — skipping")

    print(f"\nTotal pages loaded: {len(all_pages)}")
    return all_pages


# Quick sanity check — run this file directly to test
if __name__ == "__main__":
    pages = load_all_pdfs(DATA_DIR)
    
    # Print first page of first doc so you can visually verify
    if pages:
        print("\n--- Sample Output ---")
        print(f"Source: {pages[0]['source']}")
        print(f"Page:   {pages[0]['page']}")
        print(f"Text preview:\n{pages[0]['text'][:300]}")