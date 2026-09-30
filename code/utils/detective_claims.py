from __future__ import annotations

import json
import re

from pydantic import BaseModel


_REASONING_DETAIL_RE = re.compile(
    r"\b(?:because|since|as shown|as evidenced|evidence|clue|proves?|shows?|suggests?|indicates?|motive|wanted|in order to|so that)\b",
    flags=re.I,
)
_CITATION_RE = re.compile(r"\[(?:E\d+|\d+)\]|\bE\d+\b")
_BANNED_MALFORMED_RE = re.compile(
    r"\bWho\s+(?:causing|hiding|stealing|taking|scratching)\b|"
    r"\bresponsible\s+for\s+causing\b|"
    r"\blikely\s+(?:causing|hiding|stealing|taking|scratching)\b",
    flags=re.I,
)


class SuspectClaimRealization(BaseModel):
    accusation_clause: str
    guilt_conclusion: str
    innocence_conclusion: str


class ClaimRealizationBundle(BaseModel):
    neutral_event_description: str
    suspect_claims: dict[str, SuspectClaimRealization]

def normalize_wrongdoing_event(wrongdoing_event: str | None) -> str:
    event = " ".join(str(wrongdoing_event or "").split()).strip()
    return event or "the wrongdoing"


def format_responsibility_question(
    wrongdoing_event: str | None,
    first_suspect: str,
    second_suspect: str,
) -> str:
    event = normalize_wrongdoing_event(wrongdoing_event)
    return f"Who was responsible for {event}: {first_suspect} or {second_suspect}?"


def format_responsibility_claim(suspect: str, wrongdoing_event: str | None) -> str:
    event = normalize_wrongdoing_event(wrongdoing_event)
    return f"{suspect} was responsible for {event}."


def format_likely_responsibility_claim(suspect: str, wrongdoing_event: str | None) -> str:
    event = normalize_wrongdoing_event(wrongdoing_event)
    return f"{suspect} was likely responsible for {event}."


def format_unlikely_responsibility_claim(suspect: str, wrongdoing_event: str | None) -> str:
    event = normalize_wrongdoing_event(wrongdoing_event)
    return f"{suspect} was unlikely to be responsible for {event}."


def build_claim_realization_prompt(
    *,
    public_story_text: str,
    neutral_wrongdoing_event: str,
    culprit_name: str,
    rival_name: str,
    validation_feedback: str | None = None,
) -> str:
    feedback = (
        f"\nPrevious attempt failed validation: {validation_feedback}\n"
        if validation_feedback
        else ""
    )
    return f"""
You are generating canonical claim wording for a two-suspect detective debate.

Use only this public story text, the neutral wrongdoing event, and the two suspect
names. Do not use or infer from any official solution. Do not add evidence,
motive, method, argument, reasoning, or unsupported story detail.

Public story text:
{public_story_text}

Neutral wrongdoing event:
{neutral_wrongdoing_event}

Suspects:
- {culprit_name}
- {rival_name}

Task:
- Express the same underlying accusation naturally for each suspect.
- Recognize reflexive or role-sensitive relationships. For example, when the
  suspect is also the missing person, ordinary wording may be "staged their own
  disappearance" rather than "{culprit_name} caused {culprit_name}'s disappearance."
- Avoid awkward forms such as "Who causing", "responsible for causing",
  "{{person}} caused {{same person's}} disappearance", and "{{person}} likely causing".
- Preserve the exact suspect identities and the original event.
- Keep each field concise ordinary English, not professional jargon.
- Return JSON only.

Required JSON shape:
{{
  "neutral_event_description": "short neutral event phrase with no responsible actor",
  "suspect_claims": {{
    "{culprit_name}": {{
      "accusation_clause": "...",
      "guilt_conclusion": "...",
      "innocence_conclusion": "..."
    }},
    "{rival_name}": {{
      "accusation_clause": "...",
      "guilt_conclusion": "...",
      "innocence_conclusion": "..."
    }}
  }}
}}
{feedback}
""".strip()


def _name_key(name: str) -> str:
    return " ".join(str(name or "").split()).casefold()


def _contains_name(text: str, name: str) -> bool:
    text_key = _name_key(text)
    name_key = _name_key(name)
    return bool(name_key and name_key in text_key)


def _validate_no_malformed(text: str, *, suspect: str | None = None) -> None:
    if _BANNED_MALFORMED_RE.search(text):
        raise ValueError(f"Malformed claim wording is banned: {text!r}")
    if suspect:
        escaped = re.escape(suspect)
        possessive = rf"(?:{escaped}'s|{escaped}'|their own|his own|her own)"
        if re.search(rf"\b{escaped}\b.+\bcaus(?:ed|ing)\s+{possessive}\s+disappearance\b", text, flags=re.I):
            raise ValueError(f"Self-referential disappearance must be realized naturally for {suspect!r}.")


def _validate_no_reasoning_detail(text: str) -> None:
    if _CITATION_RE.search(text):
        raise ValueError(f"Claim wording must not contain citations: {text!r}")
    if _REASONING_DETAIL_RE.search(text):
        raise ValueError(f"Claim wording must not contain evidence or reasoning detail: {text!r}")


def validate_claim_realization_payload(
    payload: dict,
    *,
    culprit_name: str,
    rival_name: str,
    neutral_wrongdoing_event: str,
) -> ClaimRealizationBundle:
    bundle = ClaimRealizationBundle.model_validate(payload)
    expected_names = {culprit_name, rival_name}
    actual_names = set(bundle.suspect_claims)
    if actual_names != expected_names:
        raise ValueError(
            "claim_realization_bundle.suspect_claims must contain exactly "
            f"{sorted(expected_names)!r}; got {sorted(actual_names)!r}."
        )

    event = " ".join(bundle.neutral_event_description.split()).strip()
    if not event:
        raise ValueError("neutral_event_description must be non-empty.")
    _validate_no_malformed(event)
    _validate_no_reasoning_detail(event)
    if re.search(r"\b(?:responsible|likely|unlikely|guilty|innocent)\b", event, flags=re.I):
        raise ValueError("neutral_event_description must not contain responsibility or verdict wording.")
    for suspect in expected_names:
        escaped = re.escape(suspect)
        if re.match(rf"^\s*{escaped}\s+\w+ed\b", event, flags=re.I):
            raise ValueError("neutral_event_description must not name a suspect as the responsible actor.")

    neutral_source = normalize_wrongdoing_event(neutral_wrongdoing_event)
    if not any(token in event.casefold() for token in re.findall(r"[A-Za-z0-9']{4,}", neutral_source.casefold())):
        raise ValueError("neutral_event_description must preserve the original event.")

    normalized_claims: dict[str, SuspectClaimRealization] = {}
    for suspect in expected_names:
        claim = bundle.suspect_claims[suspect]
        fields = claim.model_dump()
        for field, value in fields.items():
            text = " ".join(str(value or "").split()).strip()
            if not text:
                raise ValueError(f"{suspect}.{field} must be non-empty.")
            if not _contains_name(text, suspect):
                raise ValueError(f"{suspect}.{field} must name the correct suspect.")
            _validate_no_malformed(text, suspect=suspect)
            _validate_no_reasoning_detail(text)
            fields[field] = text
        normalized_claims[suspect] = SuspectClaimRealization(**fields)

    return ClaimRealizationBundle(
        neutral_event_description=event,
        suspect_claims=normalized_claims,
    )


def claim_realization_bundle_to_dict(bundle: ClaimRealizationBundle) -> dict:
    return bundle.model_dump()


def claim_for_suspect(bundle: dict | ClaimRealizationBundle | None, suspect: str) -> SuspectClaimRealization | None:
    if bundle is None:
        return None
    parsed = bundle if isinstance(bundle, ClaimRealizationBundle) else ClaimRealizationBundle.model_validate(bundle)
    claim = parsed.suspect_claims.get(suspect)
    if claim is not None:
        return claim
    target = _name_key(suspect)
    for name, candidate in parsed.suspect_claims.items():
        if _name_key(name) == target:
            return candidate
    return None


def accusation_clause_for_suspect(bundle: dict | ClaimRealizationBundle | None, suspect: str, wrongdoing_event: str | None) -> str:
    claim = claim_for_suspect(bundle, suspect)
    if claim is not None:
        return claim.accusation_clause
    return format_responsibility_claim(suspect, wrongdoing_event).rstrip(".")


def guilt_conclusion_for_suspect(bundle: dict | ClaimRealizationBundle | None, suspect: str, wrongdoing_event: str | None) -> str:
    claim = claim_for_suspect(bundle, suspect)
    if claim is not None:
        return claim.guilt_conclusion
    return format_likely_responsibility_claim(suspect, wrongdoing_event)


def innocence_conclusion_for_suspect(bundle: dict | ClaimRealizationBundle | None, suspect: str, wrongdoing_event: str | None) -> str:
    claim = claim_for_suspect(bundle, suspect)
    if claim is not None:
        return claim.innocence_conclusion
    return format_unlikely_responsibility_claim(suspect, wrongdoing_event)


def neutral_event_description_from_bundle(bundle: dict | ClaimRealizationBundle | None, wrongdoing_event: str | None) -> str:
    if bundle is None:
        return normalize_wrongdoing_event(wrongdoing_event)
    parsed = bundle if isinstance(bundle, ClaimRealizationBundle) else ClaimRealizationBundle.model_validate(bundle)
    return normalize_wrongdoing_event(parsed.neutral_event_description)


def format_motion_from_claim_bundle(
    bundle: dict | ClaimRealizationBundle | None,
    *,
    wrongdoing_event: str | None,
    first_suspect: str,
    second_suspect: str,
) -> str:
    event = neutral_event_description_from_bundle(bundle, wrongdoing_event)
    return format_responsibility_question(event, first_suspect, second_suspect)


def deterministic_claim_realization_bundle(
    *,
    neutral_wrongdoing_event: str,
    culprit_name: str,
    rival_name: str,
) -> ClaimRealizationBundle:
    """Legacy deterministic fallback for tests and old local callers without a client."""
    event = normalize_wrongdoing_event(neutral_wrongdoing_event)
    causing_match = re.match(r"^causing\s+(.+)$", event, flags=re.I)

    def fallback_claim(name: str) -> dict:
        if causing_match:
            object_text = causing_match.group(1).strip()
            clause = f"{name} caused {object_text}"
            return {
                "accusation_clause": clause,
                "guilt_conclusion": f"{name} likely caused {object_text}.",
                "innocence_conclusion": f"{name} was unlikely to have caused {object_text}.",
            }
        return {
            "accusation_clause": format_responsibility_claim(name, event).rstrip("."),
            "guilt_conclusion": format_likely_responsibility_claim(name, event),
            "innocence_conclusion": format_unlikely_responsibility_claim(name, event),
        }

    payload = {
        "neutral_event_description": event,
        "suspect_claims": {
            culprit_name: fallback_claim(culprit_name),
            rival_name: fallback_claim(rival_name),
        },
    }
    return validate_claim_realization_payload(
        json.loads(json.dumps(payload)),
        culprit_name=culprit_name,
        rival_name=rival_name,
        neutral_wrongdoing_event=event,
    )
