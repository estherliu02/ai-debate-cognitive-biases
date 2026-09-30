from __future__ import annotations

import re
from typing import Any

from schemas.plan_schema import ContentAttentionQuestion
from utils.detective_claims import normalize_wrongdoing_event

CONTENT_ATTENTION_QUESTION_TEXT = "What happened in the story?"
_QUESTION_FORBIDDEN_RE = re.compile(
    r"\b(responsible|guilty|stronger|convincing|biased|decisive|side\s+[ab]|who)\b",
    flags=re.I,
)


def _clean_event_object(text: str) -> str:
    value = " ".join(text.split()).strip(" .?!,;:")
    value = re.sub(r"\s+to\s+.+$", "", value, flags=re.I).strip()
    return value or "the central item"


def _sentence_subject(text: str) -> str:
    value = _clean_event_object(text)
    if value.lower().startswith(("the ", "a ", "an ", "someone", "eddie", "ben", "eleanor")):
        return value[:1].upper() + value[1:]
    return f"The {value}"


def _passive_summary(text: str, participle: str) -> str:
    subject = _sentence_subject(text)
    head = subject.rsplit(" ", 1)[-1].strip(" .?!,;:").lower()
    be = "were" if head.endswith("s") and not head.endswith("ss") else "was"
    return f"{subject} {be} {participle}."


def broad_event_summary(wrongdoing_event: str | None) -> str:
    event = normalize_wrongdoing_event(wrongdoing_event)
    lowered = event.lower()
    verb_patterns = [
        ("stealing ", "stolen"),
        ("hiding ", "hidden"),
        ("damaging ", "damaged"),
        ("denting ", "damaged"),
        ("scratching ", "scratched"),
        ("slashing ", "slashed"),
        ("abducting ", "abducted"),
        ("kidnapping ", "abducted"),
        ("poisoning ", "poisoned"),
        ("blackmailing ", "blackmailed"),
        ("robbing ", "robbed"),
        ("sabotaging ", "sabotaged"),
    ]
    for prefix, participle in verb_patterns:
        if lowered.startswith(prefix):
            return _passive_summary(event[len(prefix):], participle)
    if lowered.startswith("causing "):
        return f"{_sentence_subject(event[len('causing '):])} happened."
    if lowered.startswith(("killing ", "murdering ")):
        return f"{_sentence_subject(event.split(' ', 1)[1])} was killed."
    return f"{event[:1].upper() + event[1:]} happened."


def build_default_content_attention_question(
    wrongdoing_event: str | None,
    evidence_bank: list[dict[str, Any]],
) -> ContentAttentionQuestion:
    correct = broad_event_summary(wrongdoing_event)
    distractor_pool = [
        "A planned meeting was canceled.",
        "A missing item was returned.",
        "A witness wrote a confession.",
        "A vehicle was repaired.",
        "A trophy was awarded.",
        "A package was delivered.",
    ]
    options: list[str] = [correct]
    for option in distractor_pool:
        if option != correct and option not in options:
            options.append(option)
        if len(options) == 4:
            break
    source_id = "E1"
    for item in evidence_bank:
        if "index" in item:
            source_id = f"E{item['index']}"
            break
    return ContentAttentionQuestion(
        question=CONTENT_ATTENTION_QUESTION_TEXT,
        options=options,
        correct_answer=correct,
        source_evidence_ids=[source_id],
    )


def validate_content_attention_question_payload(
    question_payload: Any,
    *,
    evidence_bank: list[dict[str, Any]] | None = None,
    context: str = "content_attention_question",
) -> ContentAttentionQuestion:
    if not isinstance(question_payload, dict):
        raise ValueError(f"{context} is missing.")

    attention_question = ContentAttentionQuestion(**question_payload)
    if attention_question.question != CONTENT_ATTENTION_QUESTION_TEXT:
        raise ValueError(
            f"{context}.question must be exactly {CONTENT_ATTENTION_QUESTION_TEXT!r}."
        )
    if _QUESTION_FORBIDDEN_RE.search(attention_question.question):
        raise ValueError(f"{context}.question asks about a forbidden judgment or detail.")

    options = attention_question.options
    if len(options) != 4 or any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError(f"{context}.options must contain exactly 4 non-empty strings.")
    if len(set(options)) != 4:
        raise ValueError(f"{context}.options must be unique.")
    if any(len(option.split()) > 12 for option in options):
        raise ValueError(f"{context}.options must be short event summaries.")
    if attention_question.correct_answer not in options:
        raise ValueError(f"{context}.correct_answer must exactly match one option.")
    if not attention_question.source_evidence_ids:
        raise ValueError(f"{context}.source_evidence_ids must contain at least one id.")

    valid_evidence_ids = {
        f"E{item['index']}"
        for item in (evidence_bank or [])
        if isinstance(item, dict) and "index" in item
    }
    if valid_evidence_ids:
        unknown_ids = [
            evidence_id
            for evidence_id in attention_question.source_evidence_ids
            if evidence_id not in valid_evidence_ids
        ]
        if unknown_ids:
            raise ValueError(
                f"{context}.source_evidence_ids contains unknown ids: "
                + ", ".join(unknown_ids)
            )

    return attention_question
