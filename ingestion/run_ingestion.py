import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import sys

from ingestion.loader import load_all_pdfs
from ingestion.chunker import chunk_pages

# ── Setup ─────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Anchor everything to project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "papers"   
OUTPUT_FILE = DATA_DIR / "chunks.json"


def run():
    # ── Validation ─────────────────────────────────────
    logger.info(f"Looking for PDFs in: {PDF_DIR}")

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {PDF_DIR}")

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    logger.info(f"Found {len(pdf_files)} PDF files")

    # ── Step 1: Load ───────────────────────────────────
    logger.info("=== Step 1: Loading PDFs ===")
    pages = load_all_pdfs(PDF_DIR)

    if not pages:
        raise ValueError("No pages extracted — check PDF parsing")

    # ── Step 2: Chunk ──────────────────────────────────
    logger.info("=== Step 2: Chunking ===")
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("Chunking failed — no chunks produced")

    # ── Step 3: Save ───────────────────────────────────
    logger.info(f"=== Step 3: Saving chunks to {OUTPUT_FILE} ===")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. {len(chunks)} chunks saved.")

    # ── Debug Sample ───────────────────────────────────
    logger.info("--- Sample Chunk ---")
    logger.info(json.dumps(chunks[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()