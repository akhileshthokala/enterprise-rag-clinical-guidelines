"""
Eval runner.

For each question:
  1. Retrieves top-k chunks.
  2. Checks whether the expected source file appears in those chunks (retrieval hit).
  3. Calls Claude to generate an answer using the same prompt the API uses.
  4. Checks whether the answer mentions any of the expected substrings (response smoke check).

Prints a per-question and overall pass/fail summary.

Run with:  uv run python -m eval.run_eval
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure we can import from src/ when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import Anthropic  # noqa: E402

from eval.questions import EVAL_QUESTIONS  # noqa: E402
from src.api import SYSTEM_PROMPT, format_context  # noqa: E402
from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, TOP_K  # noqa: E402
from src.retriever import retrieve  # noqa: E402


def evaluate_one(q: dict) -> dict:
    """Run one question through retrieve + generate and score it."""
    chunks = retrieve(q["question"], k=TOP_K)
    retrieved_files = {c.source_file for c in chunks}
    retrieval_hit = q["expected_source"] in retrieved_files

    # Generate answer the same way the API does.
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    user_message = (
        f"POLICY EXCERPTS:\n\n{format_context(chunks)}\n\n"
        f"QUESTION: {q['question']}\n\n"
        "Answer the question using only the excerpts above. Cite sources inline."
    )
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = "".join(b.text for b in response.content if b.type == "text").strip()

    answer_lower = answer.lower()
    matches = [s for s in q["answer_must_contain"] if s.lower() in answer_lower]
    response_hit = len(matches) > 0

    return {
        "question": q["question"],
        "expected_source": q["expected_source"],
        "retrieved_files": sorted(retrieved_files),
        "retrieval_hit": retrieval_hit,
        "response_hit": response_hit,
        "matched_keywords": matches,
        "answer": answer,
    }


def main() -> int:
    print(f"Running {len(EVAL_QUESTIONS)} eval questions against {ANTHROPIC_MODEL}\n")
    results = []
    for i, q in enumerate(EVAL_QUESTIONS, start=1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {q['question']}")
        r = evaluate_one(q)
        results.append(r)
        ret_mark = "PASS" if r["retrieval_hit"] else "FAIL"
        resp_mark = "PASS" if r["response_hit"] else "FAIL"
        print(f"    retrieval: {ret_mark}  (expected {r['expected_source']}, got {r['retrieved_files']})")
        print(f"    response:  {resp_mark}  (matched keywords: {r['matched_keywords']})")
        print(f"    answer:    {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}")
        print()

    retrieval_pass = sum(1 for r in results if r["retrieval_hit"])
    response_pass = sum(1 for r in results if r["response_hit"])
    total = len(results)

    print("=" * 60)
    print(f"Retrieval hit-rate: {retrieval_pass}/{total}  ({100*retrieval_pass/total:.0f}%)")
    print(f"Response hit-rate:  {response_pass}/{total}  ({100*response_pass/total:.0f}%)")
    print("=" * 60)

    # Exit non-zero if anything failed — useful for CI later.
    return 0 if (retrieval_pass == total and response_pass == total) else 1


if __name__ == "__main__":
    sys.exit(main())
