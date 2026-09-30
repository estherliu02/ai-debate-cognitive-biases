"""Prompt for generating factual attention checks from detective content plans."""

from __future__ import annotations

import json


def _format_evidence_bank(evidence_bank: list[dict]) -> str:
    lines = []
    for item in evidence_bank:
        lines.append(f"[E{item['index']}] {item['text']}")
    return "\n".join(lines)


def build_content_attention_question_prompt(
    *,
    content_plan: dict,
    evidence_bank: list[dict],
    validation_feedback: str | None = None,
) -> str:
    feedback_block = ""
    if validation_feedback:
        feedback_block = f"""

Previous attempt failed validation:
{validation_feedback}
Regenerate a corrected attention-check question.
"""

    return f"""
You are adding one factual attention-check question for a detective debate adjudication page.

Use ONLY the already-created content plan and the shared case evidence below.
The question should verify that a participant read the full dialogue.

Generation constraints:
- The question text must be exactly: "What happened in the story?"
- Test only the broad main event of the story.
- Do not test person names, times, places, clue details, or later-round details.
- Base the correct answer only on the main event directly supported by the content plan / case evidence.
- Do not depend on which side is stronger, which side is biased, or whether the target accusation is true.
- Do not ask who is guilty/responsible, who made the stronger case, or which interpretation is correct.
- Avoid interpretive words such as "most likely", "responsible", "guilty", "stronger", "convincing", "biased", or "decisive".
- Provide exactly 4 short event-summary options.
- correct_answer must exactly match one option string.
- source_evidence_ids must contain at least one evidence id formatted like "E12".
- The correct answer must be directly supported by the cited evidence ids.
- Distractors should be plausible in form but clearly not supported by the cited evidence.
- Avoid ambiguous wording.

Forbidden words/phrases in the question:
responsible, guilty, stronger, convincing, biased, decisive, side A, side B

Content plan JSON:
{json.dumps(content_plan, ensure_ascii=False, indent=2)}

Shared Evidence Bank:
{_format_evidence_bank(evidence_bank)}
{feedback_block}
Return valid JSON only with this exact structure:
{{
  "content_attention_question": {{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "correct_answer": "...",
    "source_evidence_ids": ["E..."]
  }}
}}
""".strip()
