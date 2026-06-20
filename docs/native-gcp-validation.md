# Native GCP Validation Plan

This repo is local-first today. The next validation step is to run the same RAG workflow in a sandbox Google Cloud project while preserving source-grounded answers and citation quality.

Use only public documents or approved synthetic/internal-safe documents. Do not upload employer/customer confidential files to a personal cloud project.

## Target Architecture

- Cloud Storage for source PDFs.
- Cloud Run for the FastAPI query service.
- ChromaDB remains local inside the container for the first smoke test, then can be swapped for Vertex AI Vector Search or AlloyDB/pgvector in a later iteration.
- Secret Manager for `ANTHROPIC_API_KEY`.
- Cloud Logging for query/error traces without storing sensitive prompt content.

## Smoke-Test Path

1. Build the local vector index from the public CMS PDFs.
2. Deploy the API container to Cloud Run with the persisted index included or mounted through a controlled build artifact.
3. Configure `ANTHROPIC_API_KEY` through Secret Manager, not a checked-in file.
4. Run the five eval questions from `eval/questions.py` against the Cloud Run URL.
5. Run `eval/retrieval_benchmark.py` locally as the portfolio benchmark baseline.
6. Verify responses include source file/page citations.

## Acceptance Criteria

- `/health` returns successfully on Cloud Run.
- `/query` returns grounded answers for the five eval prompts.
- The expected source PDF appears in retrieved citations for each eval prompt.
- No secrets, document contents, or customer data are written to git or public logs.
- Any future claim about managed vector search is backed by an implemented Vertex AI Vector Search or equivalent adapter.
