"""
Eval question set.

Each question has:
  - question: what the user asks
  - expected_source: the PDF that should appear in the retrieved chunks for the
    answer to be grounded correctly (retrieval hit-rate check).
  - answer_must_contain: at least one of these substrings (case-insensitive)
    should appear in Claude's answer (response quality smoke check).

This is a deliberately simple eval. A production setup would use LLM-as-judge
or labeled reference answers — see README "Future work" section.
"""

EVAL_QUESTIONS = [
    {
        "question": "What is required for a patient to be eligible for Medicare home health services?",
        "expected_source": "mbpm_ch07_home_health.pdf",
        "answer_must_contain": ["homebound", "physician", "plan of care"],
    },
    {
        "question": "What conditions make a patient eligible for Medicare hospice coverage?",
        "expected_source": "mbpm_ch09_hospice.pdf",
        "answer_must_contain": ["terminally ill", "six months", "life expectancy"],
    },
    {
        "question": "What inpatient hospital services does Medicare Part A cover?",
        "expected_source": "mbpm_ch01_inpatient_hospital.pdf",
        "answer_must_contain": ["semi-private", "nursing", "drugs"],
    },
    {
        "question": "Are routine physical examinations covered by Medicare?",
        "expected_source": "mbpm_ch16_general_exclusions.pdf",
        "answer_must_contain": ["not covered", "exclusion", "routine"],
    },
    {
        "question": "What does Medicare cover for outpatient diagnostic laboratory services?",
        "expected_source": "mbpm_ch15_covered_medical_services.pdf",
        "answer_must_contain": ["diagnostic", "laboratory", "physician"],
    },
]
