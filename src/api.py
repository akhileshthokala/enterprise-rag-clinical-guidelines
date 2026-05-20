"""
FastAPI app exposing the RAG query endpoint.

  POST /query   { "question": "..." }  -> { "answer": "...", "sources": [...] }
  GET  /health                          -> { "status": "ok" }

The endpoint:
  1. Retrieves top-k chunks from ChromaDB for the question.
  2. Stuffs them into a Claude prompt with explicit grounding instructions
     so the model is told NOT to invent facts beyond the provided context.
  3. Returns Claude's answer plus the source citations (file + page).

Run with:  uv run uvicorn src.api:app --reload
"""
from __future__ import annotations

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, TOP_K
from src.retriever import RetrievedChunk, retrieve

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY missing. Copy .env.example to .env and fill it in.")

client = Anthropic(api_key=ANTHROPIC_API_KEY)
app = FastAPI(title="Enterprise RAG Pipeline", version="0.1.0")


SYSTEM_PROMPT = """You are a clinical policy assistant for a healthcare payer's care management team.

You answer questions using ONLY the policy excerpts provided in the user's message.

Rules:
- If the excerpts don't contain enough information to answer, say so explicitly. Do not guess.
- Cite the source file and page for every factual claim, like: (mbpm_ch07_home_health.pdf, p. 12).
- Keep answers concise and direct. Bullets are fine when listing requirements.
- Never fabricate policy details that aren't in the excerpts."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered block Claude can reference."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i}] source: {c.source_file}, page {c.page}\n{c.text}"
        )
    return "\n\n".join(parts)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural-language question about the policy corpus")
    k: int = Field(default=TOP_K, ge=1, le=20, description="Number of chunks to retrieve")


class SourceCitation(BaseModel):
    source_file: str
    page: int
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": ANTHROPIC_MODEL}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    chunks = retrieve(req.question, k=req.k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No documents indexed. Run `uv run python -m src.ingest` first.")

    context_block = format_context(chunks)
    user_message = (
        f"POLICY EXCERPTS:\n\n{context_block}\n\n"
        f"QUESTION: {req.question}\n\n"
        "Answer the question using only the excerpts above. Cite sources inline."
    )

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    # Claude responses are a list of content blocks. For a plain text answer there's just one.
    answer_text = "".join(block.text for block in response.content if block.type == "text").strip()

    return QueryResponse(
        answer=answer_text,
        sources=[
            SourceCitation(source_file=c.source_file, page=c.page, distance=c.distance)
            for c in chunks
        ],
    )
