# Enterprise RAG Pipeline

A working RAG (Retrieval-Augmented Generation) system that lets a healthcare payer's care management team query internal clinical policy documents using Claude.

The corpus in this demo is the CMS Medicare Benefit Policy Manual (publicly available), but the pipeline is corpus-agnostic — drop any PDFs into `/docs` and re-run ingest.

## Resume Claim Mapping

This repo backs the enterprise AI assistant claim:

- Semantic retrieval over policy PDFs using local embeddings and ChromaDB.
- Source-grounded responses with file/page citations and refusal-oriented prompting.
- MCP tools in `src/mcp_server.py` so an MCP-capable LLM host can call retrieval and citation tools.
- A repeatable portfolio benchmark in `eval/retrieval_benchmark.py` that validates a 40%+ retrieval-speed improvement target against transparent manual-lookup fixtures.

See [docs/mcp-and-benchmark.md](docs/mcp-and-benchmark.md) for the MCP tool surface and benchmark notes.

## What Was Just Added

- Added an MCP stdio server that exposes retrieval and citation lookup as LLM-callable tools.
- Added a transparent portfolio benchmark for the 40%+ knowledge-retrieval speed target.
- Added docs that distinguish local portfolio evidence from future production telemetry.
- Added a native GCP validation plan for a Cloud Run / Cloud Storage / Secret Manager smoke test.

## Architecture

```
                  ┌─────────────────────────────────────────┐
                  │            User question                │
                  │   "Who qualifies for home health?"      │
                  └────────────────┬────────────────────────┘
                                   │ POST /query
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │           FastAPI endpoint              │
                  │              (src/api.py)               │
                  └────────────────┬────────────────────────┘
                                   │
                ┌──────────────────┴───────────────────┐
                ▼                                      ▼
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │  Retriever               │         │  Claude (sonnet-4-6)     │
   │  (src/retriever.py)      │         │  - grounded system prompt│
   │                          │         │  - excerpts as context   │
   │  ChromaDB query →        │         │  - cites file + page     │
   │  top-k chunks            │         │                          │
   └──────────────┬───────────┘         └──────────────┬───────────┘
                  │                                    │
                  │           ┌────────────────────────┘
                  ▼           ▼
            ┌──────────────────────────┐
            │  JSON response:          │
            │   { answer, sources[] }  │
            └──────────────────────────┘

  Ingest side (run once, then on every doc change):

       /docs/*.pdf  ──▶  pypdf  ──▶  chunker  ──▶  sentence-transformers
                                                          │
                                                          ▼
                                                   ChromaDB (persisted on disk)
```

The same retriever is also exposed through `src/mcp_server.py` as MCP tools for LLM tool-calling demos.

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| LLM | Claude Sonnet 4.6 | Best quality/cost for grounded answering |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally, zero API cost, fast |
| Vector store | ChromaDB (local persistent) | No infra to set up; swap for pgvector/Pinecone in production |
| API | FastAPI + Uvicorn | Standard Python REST framework |
| PDF parsing | pypdf | Pure-Python, no system deps |
| Package mgmt | uv | Fast, modern Python package manager |

## Setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Set your Anthropic API key
cp .env.example .env
# then edit .env and paste your key

# 3. Build the vector index from the PDFs in /docs
uv run python -m src.ingest

# 4. Start the API
uv run uvicorn src.api:app --reload
```

## Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What makes a patient eligible for Medicare hospice coverage?"}'
```

Response shape:

```json
{
  "answer": "A patient qualifies for Medicare hospice when... (mbpm_ch09_hospice.pdf, p. 4)",
  "sources": [
    { "source_file": "mbpm_ch09_hospice.pdf", "page": 4, "distance": 0.31 },
    ...
  ]
}
```

## Run the eval

```bash
uv run python -m eval.run_eval
```

The eval runs 5 representative questions and reports:
- **Retrieval hit-rate**: did the expected source PDF appear in the top-k chunks?
- **Response hit-rate**: did Claude's answer mention the key terms we'd expect from a correct answer?

Exit code is non-zero on any failure, so this can drop into CI.

## Run the Knowledge-Retrieval Benchmark

```bash
uv run python -m eval.retrieval_benchmark
```

The benchmark fixture compares manual policy lookup estimates with a RAG-assisted workflow and passes when the average improvement is at least 40%. It is intentionally transparent portfolio evidence, not production user telemetry.

## Run the MCP Tool Server

```bash
uv sync --extra mcp
uv run --extra mcp python -m src.mcp_server
```

This exposes `search_policy_docs` and `retrieve_citations` over MCP stdio for a host model to call during answer generation.

## Next Step: Native GCP Validation

The next milestone is to deploy the API path to a sandbox GCP project with Cloud Run and Secret Manager while keeping the document set public-safe.

See [docs/native-gcp-validation.md](docs/native-gcp-validation.md) for the validation plan.

## Design decisions worth calling out

1. **Local embeddings, not Voyage/OpenAI.** Keeps the demo self-contained and zero-cost beyond Anthropic. Swap `EMBEDDING_MODEL` in `.env` for a different sentence-transformers model, or replace `SentenceTransformerEmbeddingFunction` in `src/ingest.py` for a hosted embedding API.

2. **Character-based chunking.** Simple and predictable. The brief calls for a portfolio piece, not a research project — production code would use a recursive splitter that respects sentence boundaries.

3. **Cosine similarity.** Standard for normalized text embeddings. Configured via `metadata={"hnsw:space": "cosine"}` on the collection.

4. **Grounded system prompt.** The prompt explicitly tells Claude to refuse rather than fabricate when the excerpts don't contain the answer. This is the single most important thing in a healthcare-adjacent RAG system.

5. **Citations in the response.** Every answer carries the source file + page numbers used for retrieval, so a human reviewer can verify the answer without re-running the query.

## Repo layout

```
enterprise-rag-pipeline/
├── docs/                # Sample PDFs (CMS Medicare Benefit Policy Manual chapters)
│   └── sources.md       # Where each PDF came from
├── src/
│   ├── config.py        # Env-driven config
│   ├── ingest.py        # PDF → chunks → embeddings → Chroma
│   ├── retriever.py     # Chroma query wrapper
│   ├── mcp_server.py    # MCP tools for retrieval + citation lookup
│   └── api.py           # FastAPI app
├── eval/
│   ├── questions.py     # 5 question/expected-source pairs
│   ├── retrieval_benchmark.py       # 40%+ knowledge retrieval benchmark
│   ├── retrieval_benchmark_cases.json
│   └── run_eval.py      # Eval runner with pass/fail summary
├── logs/                # (gitignored) runtime logs
├── chroma_db/           # (gitignored) persisted vector store
├── pyproject.toml       # uv-managed dependencies
├── .env.example         # Copy to .env and fill in
└── README.md
```

## Future work

- Native GCP smoke test on Cloud Run with Secret Manager.
- LLM-as-judge evals for response quality (current eval only checks keyword presence).
- Streaming responses from the API for better UX on long answers.
- Reranking step (e.g. cross-encoder) between retrieval and generation.
- Hybrid search (BM25 + dense) for better recall on rare terms.
- Async ingest with progress reporting for larger corpora.
- Docker compose for a single `docker compose up` demo.
