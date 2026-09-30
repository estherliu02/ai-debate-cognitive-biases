"""Validation helpers for static pairwise detective reasoning.

Reason generation now happens during case conversion. The debate-generation
pipeline must not call an LLM to create or repair reasoning findings.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from utils.fact_units import observable_fact_units

REASONING_FINDING_VERSION = "pairwise_suspect_reasons_v5"
REASON_DIRECTIONS = ("incriminating", "exculpatory")
SUSPECT_DIRECTION_SLOT_KEYS = (
    "culprit_incriminating",
    "culprit_exculpatory",
    "rival_incriminating",
    "rival_exculpatory",
)


class ReasoningFindingError(ValueError):
    """Raised when a static reasoning payload is not usable."""


def evidence_lookup(evidence_bank: list[dict]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for item in evidence_bank or []:
        try:
            evidence_id = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            lookup[evidence_id] = text
    return lookup


def reason_id_for(suspect_name: str, direction: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", suspect_name.lower()).strip("_")
    return f"{slug}_{direction}"


def additional_reason_id_for(suspect_name: str, direction: str) -> str:
    return f"{reason_id_for(suspect_name, direction)}_additional"


def _normalize_name(name: object) -> str:
    return " ".join(str(name or "").split()).casefold()


def _normalize_fact_units(raw_units: object) -> list[str]:
    if isinstance(raw_units, list):
        return observable_fact_units([
            " ".join(str(item).split())
            for item in raw_units
            if str(item).strip()
        ])
    return []


def _statement_similarity_key(statement: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9']+", statement.casefold())
        if token
        not in {
            "a", "an", "and", "are", "as", "at", "be", "because", "being",
            "by", "can", "could", "for", "from", "had", "has", "have", "he",
            "her", "him", "his", "if", "in", "into", "is", "it", "its", "may",
            "might", "not", "of", "on", "or", "she", "so", "suggest", "suggests",
            "that", "the", "their", "them", "this", "to", "was", "were", "which",
            "who", "with", "would",
        }
    ]
    return " ".join(tokens)


def _statements_are_duplicate_or_minor_variant(left: str, right: str) -> bool:
    left_key = _statement_similarity_key(left)
    right_key = _statement_similarity_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence_ratio = SequenceMatcher(None, left_key, right_key).ratio()
    return sequence_ratio >= 0.9 or (overlap >= 0.86 and sequence_ratio >= 0.78)


def _normalize_reason_statements(raw_statements: object, *, suspect_name: str, direction: str, label: str) -> list[dict]:
    if not isinstance(raw_statements, list) or not raw_statements:
        raise ReasoningFindingError(f"{suspect_name}.{direction}.{label} reason_statements must be a non-empty list.")
    if len(raw_statements) < 2:
        raise ReasoningFindingError(
            f"{suspect_name}.{direction}.{label} reason_statements must contain at least two equally strong, distinct interpretations."
        )
    seen_ids: set[str] = set()
    statements: list[dict] = []
    for raw_entry in raw_statements:
        if not isinstance(raw_entry, dict):
            raise ReasoningFindingError(f"{suspect_name}.{direction}.{label} reason_statements entries must be objects.")
        interpretation_id = str(raw_entry.get("interpretation_id") or "").strip()
        if not interpretation_id:
            raise ReasoningFindingError(f"{suspect_name}.{direction}.{label} interpretation_id must be non-empty.")
        if interpretation_id in seen_ids:
            raise ReasoningFindingError(
                f"{suspect_name}.{direction}.{label} duplicate interpretation_id {interpretation_id!r}."
            )
        seen_ids.add(interpretation_id)
        statement = " ".join(str(raw_entry.get("statement") or "").split())
        if not statement:
            raise ReasoningFindingError(
                f"{suspect_name}.{direction}.{label} reason_statements[{interpretation_id}] statement must be non-empty."
            )
        for previous in statements:
            if _statements_are_duplicate_or_minor_variant(previous["statement"], statement):
                raise ReasoningFindingError(
                    f"{suspect_name}.{direction}.{label} contains duplicate or near-duplicate statements."
                )
        statements.append({
            "interpretation_id": interpretation_id,
            "statement": statement,
        })
    return statements


def _normalize_reason_payload(
    raw_reason: dict,
    *,
    suspect_name: str,
    direction: str,
    evidence_text_by_id: dict[int, str],
    additional: bool = False,
) -> dict:
    if "reason_statement" in raw_reason:
        label = "additional_reason" if additional else "primary"
        raise ReasoningFindingError(
            f"{suspect_name}.{direction}.{label} uses obsolete scalar reason_statement; "
            "expected v5 reason_statements interpretation list."
        )
    raw_ids = raw_reason.get("evidence_ids")
    if raw_ids is None:
        raw_ids = raw_reason.get("evidence_indices")
    if not isinstance(raw_ids, list) or not raw_ids:
        label = "additional_reason" if additional else "primary"
        raise ReasoningFindingError(f"{suspect_name}.{direction}.{label} evidence_ids must be a non-empty list.")
    evidence_ids: list[int] = []
    for raw_id in raw_ids:
        try:
            evidence_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ReasoningFindingError(f"{suspect_name}.{direction} has non-integer evidence id {raw_id!r}.") from exc
        if evidence_id not in evidence_text_by_id:
            raise ReasoningFindingError(f"{suspect_name}.{direction} references unknown evidence id E{evidence_id}.")
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)

    label = "additional_reason" if additional else "primary"
    reason_statements = _normalize_reason_statements(
        raw_reason.get("reason_statements"),
        suspect_name=suspect_name,
        direction=direction,
        label=label,
    )
    fact_units = _normalize_fact_units(raw_reason.get("fact_units"))
    if not fact_units:
        raise ReasoningFindingError(f"{suspect_name}.{direction}.{label} fact_units must be non-empty.")
    evidence_text = [evidence_text_by_id[evidence_id] for evidence_id in evidence_ids]
    default_reason_id = (
        additional_reason_id_for(suspect_name, direction)
        if additional
        else reason_id_for(suspect_name, direction)
    )
    return {
        "reason_id": str(raw_reason.get("reason_id") or default_reason_id),
        "evidence_ids": evidence_ids,
        "evidence_text": evidence_text,
        "fact_units": fact_units,
        "reason_statements": reason_statements,
    }


def _reason_signature(reason: dict) -> tuple[tuple[int, ...], tuple[str, ...], str]:
    canonical_statement = ""
    reason_statements = reason.get("reason_statements")
    if isinstance(reason_statements, list) and reason_statements:
        canonical_statement = str(reason_statements[0].get("statement") or "")
    return (
        tuple(reason.get("evidence_ids") or []),
        tuple(str(fact).casefold() for fact in reason.get("fact_units") or []),
        " ".join(canonical_statement.casefold().split()),
    )


def _validate_distinct_reasons(primary: dict, additional: dict, *, suspect_name: str, direction: str) -> None:
    primary_ids = set(primary.get("evidence_ids") or [])
    additional_ids = set(additional.get("evidence_ids") or [])
    primary_facts = {str(fact).casefold() for fact in primary.get("fact_units") or []}
    additional_facts = {str(fact).casefold() for fact in additional.get("fact_units") or []}
    if _reason_signature(primary) == _reason_signature(additional):
        raise ReasoningFindingError(f"{suspect_name}.{direction}.additional_reason duplicates the primary reason.")
    if (primary_ids & additional_ids) and (primary_facts & additional_facts):
        raise ReasoningFindingError(
            f"{suspect_name}.{direction}.additional_reason must use different core facts "
            "or a different diagnostic evidence cluster."
        )


def interpretation_dimensions(reasoning_finding: dict) -> tuple[int, int]:
    """Return (P, A) after validating uniform interpretation counts.

    P is the number of primary-reason interpretations per suspect-direction slot.
    A is the number of additional-reason interpretations per slot.
    """
    suspects = reasoning_finding.get("suspects")
    if not isinstance(suspects, list) or len(suspects) != 2:
        raise ReasoningFindingError("Static reasoning must contain exactly two suspects before interpretation sizing.")

    primary_counts: list[tuple[str, int]] = []
    additional_counts: list[tuple[str, int]] = []
    for suspect_index, entry in enumerate(suspects):
        suspect_name = str(entry.get("suspect_name") or f"suspect[{suspect_index}]").strip()
        for direction in REASON_DIRECTIONS:
            reason = entry.get(direction)
            if not isinstance(reason, dict):
                raise ReasoningFindingError(f"{suspect_name} missing {direction} reason.")
            primary_statements = reason.get("reason_statements")
            if not isinstance(primary_statements, list) or not primary_statements:
                raise ReasoningFindingError(f"{suspect_name}.{direction}.primary reason_statements must be non-empty.")
            additional = reason.get("additional_reason")
            if not isinstance(additional, dict):
                raise ReasoningFindingError(f"{suspect_name}.{direction} missing additional_reason.")
            additional_statements = additional.get("reason_statements")
            if not isinstance(additional_statements, list) or not additional_statements:
                raise ReasoningFindingError(f"{suspect_name}.{direction}.additional_reason reason_statements must be non-empty.")
            primary_counts.append((f"{suspect_name}.{direction}.primary", len(primary_statements)))
            additional_counts.append((f"{suspect_name}.{direction}.additional_reason", len(additional_statements)))

    primary_count_values = {count for _label, count in primary_counts}
    additional_count_values = {count for _label, count in additional_counts}
    if len(primary_count_values) != 1:
        raise ReasoningFindingError(
            "Primary interpretation counts must match across all four suspect-direction slots: "
            + ", ".join(f"{label}={count}" for label, count in primary_counts)
        )
    if len(additional_count_values) != 1:
        raise ReasoningFindingError(
            "Additional interpretation counts must match across all four suspect-direction slots: "
            + ", ".join(f"{label}={count}" for label, count in additional_counts)
        )
    return next(iter(primary_count_values)), next(iter(additional_count_values))


def standard_interpretation_set_ids(reasoning_finding: dict) -> list[str]:
    primary_count, additional_count = interpretation_dimensions(reasoning_finding)
    return [
        *(f"primary-{idx}" for idx in range(1, primary_count + 1)),
        *(f"additional-{idx}" for idx in range(1, additional_count + 1)),
    ]


def verbosity_interpretation_set_ids(reasoning_finding: dict) -> list[str]:
    primary_count, additional_count = interpretation_dimensions(reasoning_finding)
    return [
        f"primary-{primary_idx}__additional-{additional_idx}"
        for primary_idx in range(1, primary_count + 1)
        for additional_idx in range(1, additional_count + 1)
    ]


def validate_reasoning_finding(payload: dict, *, topic_cfg: dict) -> dict:
    """Validate a converted-case pairwise reasoning object.

    The valid schema contains exactly the culprit and rival. Missing or old
    four-suspect reasoning data is a conversion error, not a runtime generation
    opportunity.
    """
    if payload.get("reasoning_finding_version") != REASONING_FINDING_VERSION:
        raise ReasoningFindingError(
            "Static reasoning_finding has an old or missing schema version; regenerate static reasons."
        )
    evidence_text_by_id = evidence_lookup(topic_cfg.get("evidence_bank") or [])
    culprit = str(topic_cfg.get("correct_answer") or payload.get("correct_answer") or "").strip()
    rival = str(topic_cfg.get("rival_suspect") or payload.get("rival_suspect") or "").strip()
    if not culprit or not rival:
        raise ReasoningFindingError("Static reasoning requires both correct_answer and rival_suspect.")
    if _normalize_name(culprit) == _normalize_name(rival):
        raise ReasoningFindingError("Static reasoning culprit and rival must be distinct.")

    suspects = payload.get("suspects")
    if not isinstance(suspects, list) or len(suspects) != 2:
        raise ReasoningFindingError("Static reasoning must contain exactly two suspects: culprit and rival.")

    by_key: dict[str, dict] = {}
    for raw_entry in suspects:
        if not isinstance(raw_entry, dict):
            raise ReasoningFindingError("suspects entries must be objects.")
        suspect_name = str(raw_entry.get("suspect_name") or "").strip()
        if not suspect_name:
            raise ReasoningFindingError("suspect entry missing suspect_name.")
        key = _normalize_name(suspect_name)
        if key in by_key:
            raise ReasoningFindingError(f"duplicate suspect in static reasoning: {suspect_name}.")
        by_key[key] = raw_entry

    expected_keys = {_normalize_name(culprit), _normalize_name(rival)}
    if set(by_key) != expected_keys:
        raise ReasoningFindingError("Static reasoning must contain exactly the selected culprit and rival.")

    normalized_suspects: list[dict] = []
    for suspect_name in (culprit, rival):
        raw_entry = by_key[_normalize_name(suspect_name)]
        normalized_entry = {
            "suspect_name": suspect_name,
            "is_correct_answer": _normalize_name(suspect_name) == _normalize_name(culprit),
        }
        for direction in REASON_DIRECTIONS:
            raw_reason = raw_entry.get(direction)
            if not isinstance(raw_reason, dict):
                raise ReasoningFindingError(f"{suspect_name} missing {direction} reason.")
            normalized_entry[direction] = _normalize_reason_payload(
                raw_reason,
                suspect_name=suspect_name,
                direction=direction,
                evidence_text_by_id=evidence_text_by_id,
            )
            additional_reason = raw_reason.get("additional_reason")
            if not isinstance(additional_reason, dict):
                raise ReasoningFindingError(f"{suspect_name}.{direction} missing additional_reason.")
            normalized_entry[direction]["additional_reason"] = _normalize_reason_payload(
                additional_reason,
                suspect_name=suspect_name,
                direction=direction,
                evidence_text_by_id=evidence_text_by_id,
                additional=True,
            )
            _validate_distinct_reasons(
                normalized_entry[direction],
                normalized_entry[direction]["additional_reason"],
                suspect_name=suspect_name,
                direction=direction,
            )
        normalized_suspects.append(normalized_entry)

    normalized = {
        "reasoning_finding_version": REASONING_FINDING_VERSION,
        "case_id": topic_cfg.get("case_id") or payload.get("case_id"),
        "correct_answer": culprit,
        "rival_suspect": rival,
        "suspects": normalized_suspects,
    }
    primary_count, additional_count = interpretation_dimensions(normalized)
    normalized["interpretation_dimensions"] = {
        "primary": primary_count,
        "additional": additional_count,
    }
    return normalized
