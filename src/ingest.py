"""
PDF ingestion pipeline.

Reads every PDF in /docs, splits into overlapping character chunks,
embeds with sentence-transformers (locally, no API cost), and persists
to ChromaDB. Incremental: chunks already in the collection are skipped,
so re-running after a partial failure picks up where it left off.

Run with:  uv run python -m src.ingest
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def get_collection() -> chromadb.Collection:
    """Return (or create) the persisted ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) for every page in the PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _chunk_id(source_file: str, page: int, chunk_index: int, text: str) -> str:
    """Stable, unique ID for a chunk so we can skip duplicates on re-run."""
    payload = f"{source_file}::{page}::{chunk_index}::{text[:64]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def ingest_pdf(pdf_path: Path, collection: chromadb.Collection) -> int:
    """Ingest one PDF. Returns the number of new chunks added."""
    source_file = pdf_path.name
    pages = _extract_pages(pdf_path)

    ids, documents, metadatas = [], [], []
    for page_num, page_text in pages:
        for chunk_idx, chunk in enumerate(_chunk_text(page_text)):
            chunk_id = _chunk_id(source_file, page_num, chunk_idx, chunk)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source_file": source_file, "page": page_num})

    if not ids:
        return 0

    # Skip IDs already in the collection (incremental ingest).
    existing = set(collection.get(ids=ids)["ids"])
    new_ids = [i for i in ids if i not in existing]
    if not new_ids:
        print(f"  {source_file}: all {len(ids)} chunks already indexed, skipping.")
        return 0

    # Filter parallel lists to only new chunks.
    idx_map = {id_: pos for pos, id_ in enumerate(ids)}
    new_docs = [documents[idx_map[i]] for i in new_ids]
    new_meta = [metadatas[idx_map[i]] for i in new_ids]

    collection.add(ids=new_ids, documents=new_docs, metadatas=new_meta)
    print(f"  {source_file}: added {len(new_ids)} new chunks (skipped {len(existing)} existing).")
    return len(new_ids)


def main() -> None:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}. Add documents and re-run.")
        return

    print(f"Found {len(pdfs)} PDF(s) in {DOCS_DIR}")
    collection = get_collection()

    total_new = 0
    for pdf_path in pdfs:
        print(f"Processing {pdf_path.name} ...")
        total_new += ingest_pdf(pdf_path, collection)

    total_in_db = collection.count()
    print(f"\nDone. {total_new} new chunks added. {total_in_db} total chunks in collection '{CHROMA_COLLECTION}'.")


if __name__ == "__main__":
    main()
