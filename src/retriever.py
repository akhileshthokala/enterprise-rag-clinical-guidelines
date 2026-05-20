"""
Retrieval layer.

Wraps ChromaDB so the API and eval script share the same retrieval behavior.
Returns chunks plus their distance score so the caller can decide how to use them.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import TOP_K
from src.ingest import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page: int
    distance: float  # lower = more similar (cosine distance)


def retrieve(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    """Return the top-k most relevant chunks for a query, ranked by similarity."""
    collection = get_collection()
    result = collection.query(query_texts=[query], n_results=k)

    # Chroma returns parallel lists; index 0 corresponds to our single input query.
    chunks: list[RetrievedChunk] = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(
            RetrievedChunk(
                text=text,
                source_file=meta["source_file"],
                page=int(meta["page"]),
                distance=float(dist),
            )
        )
    return chunks
