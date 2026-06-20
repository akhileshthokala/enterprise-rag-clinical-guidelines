# MCP and Retrieval Benchmark

This repo supports a resume claim about an enterprise RAG assistant using MCP, semantic retrieval, a vector database, and LLM tool calling.

## MCP Tool Surface

`src/mcp_server.py` exposes the Chroma-backed retriever as MCP tools:

- `search_policy_docs(question, k)`: returns retrieved excerpts, citations, and a model-ready context block.
- `retrieve_citations(question, k)`: returns file/page citations for audit and answer-grounding checks.

Run it with:

```bash
uv run --extra mcp python -m src.mcp_server
```

An MCP-capable host model can call these tools when it needs policy evidence, then synthesize the final answer from the returned excerpts.

## Benchmark

`eval/retrieval_benchmark.py` is a deterministic portfolio benchmark comparing a manual policy-review baseline with a RAG-assisted workflow for representative knowledge-retrieval tasks.

```bash
uv run python -m eval.retrieval_benchmark
```

The benchmark passes when the average improvement is at least 40%. The included fixture is intentionally transparent portfolio evidence and should be replaced with real analyst timing data or app telemetry in a production deployment.
