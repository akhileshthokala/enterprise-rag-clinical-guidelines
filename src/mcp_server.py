"""MCP tool server for policy-document retrieval.

Run with:
    uv run python -m src.mcp_server

This exposes the RAG retriever as LLM-callable tools over MCP stdio. The tools
return grounded snippets and citations; a host model can decide when to call
them and how to synthesize the final answer.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.config import TOP_K
from src.retriever import RetrievedChunk, retrieve


mcp = FastMCP("enterprise-rag-clinical-guidelines")


@mcp.tool()
def search_policy_docs(question: str, k: int = TOP_K) -> dict:
    """Retrieve grounded policy excerpts for a clinical-policy question."""
    chunks = retrieve(question, k=k)
    return {
        "question": question,
        "sources": [_chunk_to_source(chunk) for chunk in chunks],
        "context": _format_context(chunks),
    }


@mcp.tool()
def retrieve_citations(question: str, k: int = TOP_K) -> list[dict]:
    """Return file/page citations for the most relevant policy chunks."""
    return [_chunk_to_source(chunk) for chunk in retrieve(question, k=k)]


def _chunk_to_source(chunk: RetrievedChunk) -> dict:
    return {
        "source_file": chunk.source_file,
        "page": chunk.page,
        "distance": round(chunk.distance, 4),
        "preview": chunk.text[:500],
    }


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {index}] source: {chunk.source_file}, page {chunk.page}\n{chunk.text}"
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()
