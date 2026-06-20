"""Knowledge-retrieval benchmark for the portfolio RAG workflow.

The fixture compares a manual policy-review baseline against a RAG-assisted
workflow for representative internal knowledge-retrieval tasks. It is designed
to be deterministic and CI-friendly; production teams should replace the fixture
with real analyst timings or application telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path


TARGET_IMPROVEMENT = 40.0
CASES_PATH = Path(__file__).with_name("retrieval_benchmark_cases.json")


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def improvement_percent(baseline_minutes: float, rag_assisted_minutes: float) -> float:
    if baseline_minutes <= 0:
        raise ValueError("baseline_minutes must be positive")
    return ((baseline_minutes - rag_assisted_minutes) / baseline_minutes) * 100


def summarize(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        improvement = improvement_percent(
            float(case["baseline_minutes"]),
            float(case["rag_assisted_minutes"]),
        )
        rows.append({**case, "improvement_percent": round(improvement, 1)})

    average = sum(row["improvement_percent"] for row in rows) / len(rows)
    return {
        "case_count": len(rows),
        "average_improvement_percent": round(average, 1),
        "target_improvement_percent": TARGET_IMPROVEMENT,
        "passed": average >= TARGET_IMPROVEMENT,
        "rows": rows,
    }


def main() -> int:
    report = summarize(load_cases())
    print(
        "knowledge retrieval benchmark: "
        f"{report['average_improvement_percent']}% average improvement "
        f"across {report['case_count']} tasks"
    )
    for row in report["rows"]:
        print(
            f"- {row['task']}: {row['improvement_percent']}% "
            f"({row['baseline_minutes']}m -> {row['rag_assisted_minutes']}m)"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
