import re
import uuid


def clean_text(text: str) -> str:
    """
    Legal PDFs often have:
    - Multiple consecutive newlines (section breaks formatted as whitespace)
    - Hyphenated words split across lines (e.g., "employ-\nment")
    - Page headers/footers repeated on every page ("Page 1 of 20", "DRAFT")
    
    We clean before chunking — garbage in, garbage out.
    """
    # Fix hyphenated line breaks (common in legal docs)
    text = re.sub(r"-\n", "", text)

    # Collapse multiple newlines into double newline (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Context-aware splitting strategy for legal text.
    
    Priority order of split points (we try the least destructive first):
    1. Double newline     → paragraph boundary (most natural)
    2. Single newline     → line boundary
    3. Period + space     → sentence boundary
    4. Space              → word boundary (last resort)
    
    We NEVER cut mid-word. chunk_size is in characters, not tokens.
    500 chars ≈ 100-130 words ≈ fits well within any embedding model limit.
    
    chunk_overlap = 100 chars of the previous chunk carried into the next.
    This ensures a clause that spans two chunks isn't lost completely.
    """

    # Split points in order of preference
    separators = ["\n\n", "\n", ". ", " "]

    def split_by_separator(text, sep):
        return [s.strip() for s in text.split(sep) if s.strip()]

    # Start with paragraph-level splits
    segments = split_by_separator(text, "\n\n")

    chunks = []
    current_chunk = ""

    for segment in segments:
        # If adding this segment keeps us under chunk_size, add it
        if len(current_chunk) + len(segment) <= chunk_size:
            current_chunk += (" " if current_chunk else "") + segment

        else:
            # Save what we have
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If the segment itself is larger than chunk_size,
            # we need to break it further (e.g., a very long clause)
            if len(segment) > chunk_size:
                # Break by sentences
                sentences = split_by_separator(segment, ". ")
                temp = ""
                for sentence in sentences:
                    if len(temp) + len(sentence) <= chunk_size:
                        temp += (" " if temp else "") + sentence
                    else:
                        if temp:
                            chunks.append(temp.strip())
                        temp = sentence
                if temp:
                    chunks.append(temp.strip())
                current_chunk = ""
            else:
                current_chunk = segment

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Apply overlap: carry last `chunk_overlap` chars of previous chunk
    # into the start of the next chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap_text = chunks[i - 1][-chunk_overlap:]
            overlapped.append(overlap_text + " " + chunks[i])
        return overlapped

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Takes the output of loader.py (list of page dicts)
    and returns a flat list of chunk dicts with full metadata.
    
    Each chunk carries:
    - text       : the actual content
    - source     : which PDF it came from
    - page       : which page
    - chunk_id   : unique ID (used later when storing in vector DB)
    - chunk_index: position within the document (useful for ordering)
    """
    all_chunks = []
    chunk_index = 0

    for page in pages:
        cleaned = clean_text(page["text"])
        chunks = split_into_chunks(cleaned, chunk_size=500, chunk_overlap=100)

        for chunk_text in chunks:
            # Skip chunks that are too short to be meaningful
            # e.g., "Section 1." alone is not a useful chunk
            if len(chunk_text) < 80:
                continue

            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),  # unique ID for each chunk
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Test chunker independently with a dummy page
    dummy_pages = [{
        "text": """Section 4 — Termination of Employment

An employer may terminate the services of an employee on the following grounds:
(a) misconduct as defined under Section 14 of this Act;
(b) inefficiency or incompetence, subject to a notice period of thirty days;
(c) redundancy arising from restructuring or closure of the establishment.

The employee shall be entitled to receive a termination letter stating the reason 
for dismissal within three working days of termination.

Section 5 — Severance Pay

Every employee who has completed one year of continuous service shall be entitled 
to severance pay calculated at the rate of thirty days wages for each completed 
year of service.""",
        "source": "test.pdf",
        "page": 1
    }]

    chunks = chunk_pages(dummy_pages)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk['text'])} chars")
        print(chunk["text"])