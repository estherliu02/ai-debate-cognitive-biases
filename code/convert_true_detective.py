#!/usr/bin/env python3
"""Convert the four experiment cases from the True Detective CSV into debate JSON files.

Source:     data/true-detective/data/data.zip (detective-puzzles.csv), not included;
            git clone https://github.com/MaksymDel/true-detective data/true-detective
Accusation results: data/5-minute-mystery/accusation_results.json
Output directory: data/detective_cases/
Each file: data/detective_cases/{case_id}.json
Index:      data/detective_cases/_index.json

The suspect evidence analysis calls an LLM through OpenRouter and needs
OPENROUTER_API_KEY. Existing case files with a complete analysis are reused.

Run from repo root:
    python code/convert_true_detective.py
"""

from __future__ import annotations

import csv
import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

_CODE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_ROOT.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

_DEFAULT_SOURCE_PATH = _REPO_ROOT / "data" / "true-detective" / "data" / "data.zip"
_CASES_OUT_DIR = _REPO_ROOT / "data" / "detective_cases"

try:
    import pysbd
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    pysbd = None

from configs.models import MODEL_CONFIGS
from core.openrouter_client import OpenRouterClient
from utils.detective_claims import format_responsibility_claim
from utils.fact_units import observable_fact_units


CASE_SPECS = [
    ("The Locker Incident", "our-quarterback-is-missing"),
    ("The Diamond Necklace", "the-diamond-necklace"),
    ("The Missing Briefcase", "the-missing-briefcase"),
    ("The Missing Trophy", "the-mystery-of-the-leprechaun-s-trophy"),
]

_ACCUSATION_RESULTS_PATH = _REPO_ROOT / "data" / "5-minute-mystery" / "accusation_results.json"
_PAIRWISE_REASONING_FINDING_VERSION = "pairwise_suspect_reasons_v5"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start_char: int | None = None
    end_char: int | None = None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalize_case_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.netloc:
        host = parsed.netloc.casefold()
        path = unquote(parsed.path)
    else:
        host = ""
        path = unquote(raw.split("?", 1)[0].split("#", 1)[0])
    path = re.sub(r"/results/?$", "", path, flags=re.IGNORECASE).rstrip("/")
    return f"{host}{path}".casefold() if host else path.casefold()


def parse_suspects(answer_options: str) -> list[dict]:
    """'(a) Chris Henderson; (b) Dave Perkins' → [{"key": "a", "name": "Chris Henderson"}, ...]"""
    suspects = []
    for part in answer_options.split(";"):
        part = part.strip()
        m = re.match(r"\((\w+)\)\s+(.+)", part)
        if m:
            suspects.append({"key": m.group(1), "name": m.group(2).strip()})
    return suspects


def parse_answer_name(answer: str) -> str:
    """'(a) Chris Henderson' → 'Chris Henderson'"""
    m = re.match(r"\(\w+\)\s+(.+)", answer.strip())
    return m.group(1).strip() if m else answer.strip()


def _name_key(name: str) -> str:
    return " ".join(parse_answer_name(str(name or "")).split()).casefold()


def _canonical_option_name(name: str, suspects: list[dict]) -> str | None:
    wanted = _name_key(name)
    for suspect in suspects:
        if _name_key(suspect.get("name", "")) == wanted:
            return suspect["name"]
    return None


def _make_sentence_segmenter() -> pysbd.Segmenter:
    if pysbd is None:
        class _FallbackSegmenter:
            def segment(self, text: str) -> list[str]:
                return [
                    part.strip()
                    for part in re.split(r"(?<=[.!?])\s+", text)
                    if part.strip()
                ]

        return _FallbackSegmenter()
    try:
        return pysbd.Segmenter(language="en", clean=False, char_span=True)
    except TypeError:
        return pysbd.Segmenter(language="en", clean=False)


_SENTENCE_SEGMENTER = _make_sentence_segmenter()
_SPEECH_ATTRIBUTION_RE = re.compile(
    r"^(?:"
    r"[A-Z][A-Za-z'.-]*"
    r"|[A-Z][A-Za-z'.-]*\s+[A-Z][A-Za-z'.-]*"
    r"|he|she|they|we|I"
    r")\s+"
    r"(?:said|asked|replied|answered|shouted|whispered|muttered|cried|called|added|continued|explained|admitted|insisted|noted|declared|claimed|remarked)\b",
    flags=re.IGNORECASE,
)
_QUOTED_DIALOGUE_END_RE = re.compile(r"[.!?][\"”']$")


def _normalize_sentence_text(text: str) -> str:
    return " ".join(text.split())


def _normalized_with_char_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    char_map: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if not in_whitespace and normalized_chars:
                normalized_chars.append(" ")
                char_map.append(index)
            in_whitespace = True
            continue
        normalized_chars.append(char)
        char_map.append(index)
        in_whitespace = False
    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        char_map.pop()
    return "".join(normalized_chars), char_map


def _find_normalized_span(source: str, snippet: str, start_after: int = 0) -> tuple[int | None, int | None]:
    normalized_source, char_map = _normalized_with_char_map(source)
    normalized_snippet = _normalize_sentence_text(snippet)
    if not normalized_snippet:
        return None, None

    normalized_start_after = 0
    for normalized_index, source_index in enumerate(char_map):
        if source_index >= start_after:
            normalized_start_after = normalized_index
            break

    found = normalized_source.find(normalized_snippet, normalized_start_after)
    if found == -1:
        found = normalized_source.find(normalized_snippet)
    if found == -1:
        return None, None

    start = char_map[found]
    end = char_map[found + len(normalized_snippet) - 1] + 1
    return start, end


def _trim_span(source: str, start: int | None, end: int | None) -> tuple[int | None, int | None]:
    if start is None or end is None:
        return start, end
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def _coerce_pysbd_result(item: object) -> tuple[str, int | None, int | None]:
    if isinstance(item, dict):
        text = item.get("sent") or item.get("text") or item.get("sentence") or ""
        return str(text), item.get("start"), item.get("end")
    if isinstance(item, tuple):
        text = item[0] if item else ""
        start = item[1] if len(item) > 1 else None
        end = item[2] if len(item) > 2 else None
        return str(text), start, end

    sent = getattr(item, "sent", None)
    if sent is None:
        sent = getattr(item, "text", None)
    if sent is None:
        sent = str(item)
    return str(sent), getattr(item, "start", None), getattr(item, "end", None)


def _span_from_pysbd_result(source: str, item: object, start_after: int = 0) -> SentenceSpan:
    sent, start, end = _coerce_pysbd_result(item)
    if start is None or end is None:
        start, end = _find_normalized_span(source, sent, start_after)
    start, end = _trim_span(source, start, end)
    if start is not None and end is not None:
        sent = source[start:end]
    return SentenceSpan(_normalize_sentence_text(sent), start, end)


def _merge_spans(source: str, first: SentenceSpan, second: SentenceSpan) -> SentenceSpan:
    start = first.start_char
    end = second.end_char
    if start is not None and end is not None:
        text = source[start:end]
    else:
        text = f"{first.text} {second.text}"
    return SentenceSpan(_normalize_sentence_text(text), start, end)


def _should_merge_dialogue_attribution(first: SentenceSpan, second: SentenceSpan) -> bool:
    return bool(
        _QUOTED_DIALOGUE_END_RE.search(first.text)
        and _SPEECH_ATTRIBUTION_RE.search(second.text)
    )


def _merge_dialogue_attributions(source: str, spans: list[SentenceSpan]) -> list[SentenceSpan]:
    merged: list[SentenceSpan] = []
    index = 0
    while index < len(spans):
        current = spans[index]
        if index + 1 < len(spans) and _should_merge_dialogue_attribution(current, spans[index + 1]):
            merged.append(_merge_spans(source, current, spans[index + 1]))
            index += 2
        else:
            merged.append(current)
            index += 1
    return merged


def split_sentence_spans(text: str) -> list[SentenceSpan]:
    """Sentence-split text with pySBD and preserve source offsets when available.

    Postprocessing keeps quoted dialogue with its following attribution, e.g.
    '"I never touched it." Sadie said.'
    """
    spans: list[SentenceSpan] = []
    start_after = 0
    for item in _SENTENCE_SEGMENTER.segment(text):
        span = _span_from_pysbd_result(text, item, start_after)
        if span.text:
            spans.append(span)
        if span.end_char is not None:
            start_after = span.end_char
    return _merge_dialogue_attributions(text, spans)


def split_sentences(text: str) -> list[str]:
    """Compatibility wrapper for existing evidence-bank construction."""
    return [span.text for span in split_sentence_spans(text)]


def build_evidence_bank(mystery_text: str) -> list[dict]:
    sentences = split_sentences(mystery_text)
    return [{"index": i + 1, "text": s} for i, s in enumerate(sentences)]


# ---------------------------------------------------------------------------
# LLM suspect evidence analysis
# ---------------------------------------------------------------------------

_SUSPECT_ANALYSIS_MAX_ATTEMPTS = 3
_SUSPECT_ANALYSIS_SCHEMA_VERSION = "pairwise_side_conditioned_cluster_v5"
_SUSPECT_ANALYSIS_REASONING_MAX_WORDS = 45
_SUSPECT_ANALYSIS_REASONING_MAX_CHARS = 320
_WEAK_EVIDENCE_LANGUAGE = "weakly suggests / could support / may be interpreted as"


class SuspectEvidenceAnalysisError(ValueError):
    """Raised when LLM suspect evidence analysis cannot be validated."""


@dataclass(frozen=True)
class SuspectEvidenceAnalysisResult:
    analysis: list[dict]
    official_clues: list[dict]


def _suspect_evidence_model_config() -> dict:
    return MODEL_CONFIGS.get("suspect_evidence_analyzer", MODEL_CONFIGS["evaluator"])


def _evidence_text_by_index(evidence_bank: list[dict]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for item in evidence_bank:
        index = item.get("index")
        text = item.get("text")
        if not isinstance(index, int) or not isinstance(text, str):
            raise SuspectEvidenceAnalysisError(f"Invalid evidence_bank item: {item!r}")
        lookup[index] = text
    return lookup


def _format_evidence_bank(evidence_bank: list[dict]) -> str:
    lines = []
    for item in evidence_bank:
        lines.append(f"{item['index']}. {item['text']}")
    return "\n".join(lines)


def _format_suspects(suspects: list[dict]) -> str:
    return json.dumps(suspects, ensure_ascii=False, indent=2)


def _format_official_clues(official_clues: list[dict] | None) -> str:
    if not official_clues:
        return "[]"
    return json.dumps(official_clues, ensure_ascii=False, indent=2)


def _official_clues_payload_list(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("official_clues"), list):
        return payload["official_clues"]
    raise SuspectEvidenceAnalysisError("Official clue extraction response must contain an 'official_clues' list")


def validate_official_clues(
    payload: Any,
    *,
    evidence_bank: list[dict],
    suspects: list[dict],
) -> list[dict]:
    raw_clues = _official_clues_payload_list(payload)
    evidence_lookup = _evidence_text_by_index(evidence_bank)
    suspect_names = {str(s["name"]).strip() for s in suspects}
    normalized_clues: list[dict] = []
    for raw_clue in raw_clues:
        if not isinstance(raw_clue, dict):
            raise SuspectEvidenceAnalysisError("Each official clue must be an object")
        clue = raw_clue.get("clue")
        if not isinstance(clue, str) or not clue.strip():
            raise SuspectEvidenceAnalysisError("Each official clue must include a non-empty clue")
        raw_indices = raw_clue.get("evidence_indices")
        if not isinstance(raw_indices, list):
            continue
        if not raw_indices:
            continue
        indices: list[int] = []
        invalid_indices = False
        for raw_index in raw_indices:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                invalid_indices = True
                break
            if raw_index not in evidence_lookup:
                invalid_indices = True
                break
            indices.append(raw_index)
        if invalid_indices:
            continue
        indices = list(dict.fromkeys(indices))
        raw_suspects = raw_clue.get("suspect_names") or []
        if not isinstance(raw_suspects, list):
            raise SuspectEvidenceAnalysisError("official_clue.suspect_names must be a list")
        clue_suspects: list[str] = []
        for suspect_name in raw_suspects:
            if not isinstance(suspect_name, str):
                continue
            if suspect_name.strip() not in suspect_names:
                continue
            if suspect_name.strip() not in clue_suspects:
                clue_suspects.append(suspect_name.strip())
        role = raw_clue.get("role")
        if role not in {
            "incriminates_culprit",
            "exculpates_culprit",
            "incriminates_other_suspect",
            "exculpates_other_suspect",
            "general_official_clue",
        }:
            raise SuspectEvidenceAnalysisError(f"Official clue has invalid role: {role!r}")
        if raw_suspects and not clue_suspects and role != "general_official_clue":
            continue
        reasoning = raw_clue.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise SuspectEvidenceAnalysisError("Official clue must include short reasoning")
        normalized_clues.append({
            "clue": clue.strip(),
            "evidence_indices": indices,
            "evidence_text": [evidence_lookup[index] for index in indices],
            "suspect_names": clue_suspects,
            "role": role,
            "reasoning": " ".join(reasoning.split())[:_SUSPECT_ANALYSIS_REASONING_MAX_CHARS],
        })
    if not normalized_clues:
        raise SuspectEvidenceAnalysisError("Official clue extraction returned no usable clues")
    return normalized_clues


def _build_official_clue_extraction_prompt(
    *,
    case_name: str,
    evidence_bank: list[dict],
    suspects: list[dict],
    correct_name: str,
    outcome_text: str,
    retry_feedback: str | None,
) -> str:
    feedback_block = ""
    if retry_feedback:
        feedback_block = (
            "\nPrevious official clue extraction failed validation. Fix the next JSON response.\n"
            f"Validation feedback: {retry_feedback}\n"
        )
    return f"""Extract the official public clues used by the solution of a detective mystery.

Case name: {case_name}
Correct answer name: {correct_name}
Answer options:
{_format_suspects(suspects)}

Outcome reference:
{outcome_text}

Task:
1. Read the outcome reference to understand the official solution reasoning.
2. Identify the public clues from that reasoning that also appear in the numbered evidence_bank.
3. Map each clue to the evidence indices from the evidence_bank needed to express that clue. Use comparison clusters when the official clue requires comparing public facts, e.g. wobbly handwriting vs neatly printed note.

Priority: physical clue / wordplay / handwriting / ability / access / contradiction > motive / attitude / emotion.

Do not include facts that only appear in outcome_reference and cannot be mapped to evidence_bank.
Return only official solution clues or clue-comparisons that matter for identifying or excluding suspects.
Every official clue must cite the evidence_indices needed to express that clue.
suspect_names must contain only exact names from the answer options above. Do not include witnesses, relatives, narrators, or other story people who are not answer options. If a useful clue is general background and does not map to an answer option, use an empty suspect_names list and role "general_official_clue".

Return strict JSON as one object with this shape and no Markdown code fence:
{{
  "official_clues": [
    {{
      "clue": "Short description of the public official clue.",
      "evidence_indices": [11, 29],
      "suspect_names": ["Uncle Larry"],
      "role": "exculpates_other_suspect",
      "reasoning": "Short explanation of how these evidence items express the official clue."
    }}
  ]
}}

Allowed role values:
- incriminates_culprit
- exculpates_culprit
- incriminates_other_suspect
- exculpates_other_suspect
- general_official_clue
{feedback_block}
Numbered evidence_bank:
{_format_evidence_bank(evidence_bank)}
"""


def extract_official_clues(
    *,
    client: OpenRouterClient,
    case_name: str,
    evidence_bank: list[dict],
    suspects: list[dict],
    correct_name: str,
    outcome_text: str,
) -> list[dict]:
    model_cfg = _suspect_evidence_model_config()
    retry_feedback: str | None = None
    last_error: Exception | None = None
    for attempt in range(1, _SUSPECT_ANALYSIS_MAX_ATTEMPTS + 1):
        try:
            payload = client.complete_json(
                model=model_cfg["model"],
                prompt=_build_official_clue_extraction_prompt(
                    case_name=case_name,
                    evidence_bank=evidence_bank,
                    suspects=suspects,
                    correct_name=correct_name,
                    outcome_text=outcome_text,
                    retry_feedback=retry_feedback,
                ),
                temperature=0,
                max_tokens=1800,
            )
            return validate_official_clues(
                payload,
                evidence_bank=evidence_bank,
                suspects=suspects,
            )
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            if attempt < _SUSPECT_ANALYSIS_MAX_ATTEMPTS:
                print(
                    f"  official clue extraction attempt "
                    f"{attempt}/{_SUSPECT_ANALYSIS_MAX_ATTEMPTS} failed: {exc}"
                )
    raise SuspectEvidenceAnalysisError(
        f"Could not extract official public clues for {case_name!r} "
        f"after {_SUSPECT_ANALYSIS_MAX_ATTEMPTS} attempts: {last_error}"
    )


def _analysis_payload_list(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("analysis"), list):
        return payload["analysis"]
    if isinstance(payload, list):
        return payload
    raise SuspectEvidenceAnalysisError("LLM response must contain an 'analysis' list")


def _normalize_evidence_indices(value: Any, *, suspect_name: str, direction: str) -> list[int]:
    if not isinstance(value, list):
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{direction}.evidence_indices must be a list"
        )
    indices: list[int] = []
    for raw_index in value:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise SuspectEvidenceAnalysisError(
                f"{suspect_name}.{direction}.evidence_indices contains a non-integer index"
            )
        indices.append(raw_index)
    if len(indices) != len(set(indices)):
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{direction}.evidence_indices contains duplicates"
        )
    return indices


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


def _validate_statement_direction(statement: str, *, suspect_name: str, direction: str, context: str) -> None:
    lowered = statement.casefold()
    if direction == "incriminating":
        opposite_patterns = (
            r"\bweaken(?:s|ed|ing)?\s+(?:the\s+)?(?:case|suspicion|accusation)\b",
            r"\bcreate(?:s|d|ing)?\s+(?:reasonable\s+)?doubt\b",
            r"\breduce(?:s|d|ing)?\s+(?:the\s+)?(?:case|suspicion|responsibility)\b",
            r"\b(?:support|supports|suggest|suggests)\s+(?:his|her|their|the)?\s*innocence\b",
            r"\b(?:less\s+likely|unlikely)\s+(?:to\s+be\s+responsible|responsible|guilty|involved)\b",
            r"\b(?:could\s+not|couldn't|did\s+not|didn't)\s+(?:have\s+)?(?:commit|take|steal|hide|damage|cause)\b",
            r"\black(?:ed|s)?\s+(?:the\s+)?(?:opportunity|ability|means|motive)\b",
        )
    elif direction == "exculpatory":
        opposite_patterns = (
            r"\b(?:support|supports|supported|strengthen|strengthens)\s+(?:the\s+)?(?:case|suspicion|accusation)\b",
            r"\b(?:incriminate|incriminates|implicate|implicates)\b",
            r"\bpoint(?:s|ed|ing)?\s+(?:toward|to)\s+(?:his|her|their|the)?\s*(?:guilt|responsibility|involvement)\b",
            r"\b(?:likely|more\s+likely)\s+(?:to\s+be\s+)?(?:responsible|guilty|involved)\b",
            r"\b(?:had|gave\s+\w+)\s+(?:the\s+)?(?:motive|opportunity|ability|means)\s+to\b",
        )
    else:
        raise SuspectEvidenceAnalysisError(f"Unexpected direction for {suspect_name}.{context}: {direction!r}")
    for pattern in opposite_patterns:
        if re.search(pattern, lowered):
            raise SuspectEvidenceAnalysisError(
                f"{suspect_name}.{context}.reason_statements contains reasoning that points against "
                f"the requested {direction} direction"
            )


def _normalize_reason_statement_entry(
    value: Any,
    *,
    suspect_name: str,
    direction: str,
    context: str,
    seen_ids: set[str],
) -> dict:
    if not isinstance(value, dict):
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements entries must be objects"
        )
    interpretation_id = str(value.get("interpretation_id") or "").strip()
    if not interpretation_id:
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements.interpretation_id must be non-empty"
        )
    if interpretation_id in seen_ids:
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements has duplicate interpretation_id {interpretation_id!r}"
        )
    seen_ids.add(interpretation_id)
    statement = value.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements[{interpretation_id}].statement must be non-empty text"
        )
    statement = " ".join(statement.split())
    if len(statement) > _SUSPECT_ANALYSIS_REASONING_MAX_CHARS:
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements[{interpretation_id}].statement is too long; keep it under "
            f"{_SUSPECT_ANALYSIS_REASONING_MAX_CHARS} characters"
        )
    if len(statement.split()) > _SUSPECT_ANALYSIS_REASONING_MAX_WORDS:
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements[{interpretation_id}].statement is too long; keep it under "
            f"{_SUSPECT_ANALYSIS_REASONING_MAX_WORDS} words"
        )
    _validate_statement_direction(
        statement,
        suspect_name=suspect_name,
        direction=direction,
        context=context,
    )
    return {
        "interpretation_id": interpretation_id,
        "statement": statement,
    }


def _normalize_reason_statements(
    value: Any,
    *,
    suspect_name: str,
    direction: str,
    context: str,
) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements must be a non-empty list"
        )
    if len(value) < 2:
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.reason_statements must contain at least two equally strong, distinct interpretations"
        )
    seen_ids: set[str] = set()
    statements: list[dict] = []
    for raw_entry in value:
        entry = _normalize_reason_statement_entry(
            raw_entry,
            suspect_name=suspect_name,
            direction=direction,
            context=context,
            seen_ids=seen_ids,
        )
        for previous in statements:
            if _statements_are_duplicate_or_minor_variant(previous["statement"], entry["statement"]):
                raise SuspectEvidenceAnalysisError(
                    f"{suspect_name}.{context}.reason_statements contains duplicate or near-duplicate statements"
                )
        statements.append(entry)
    return statements


def _normalize_direction_fact_units(value: Any, *, suspect_name: str, direction: str) -> list[str]:
    del suspect_name, direction
    if not isinstance(value, list):
        return []
    return observable_fact_units([
        " ".join(str(item).split())
        for item in value
        if str(item).strip()
    ])


def _normalize_direction_reason(
    raw_reason: Any,
    *,
    suspect_name: str,
    direction: str,
    evidence_lookup: dict[int, str],
    label: str = "primary",
) -> dict:
    if not isinstance(raw_reason, dict):
        raise SuspectEvidenceAnalysisError(f"{suspect_name}.{direction}.{label} must be an object")
    context = f"{direction}.{label}" if label != "primary" else direction
    indices = _normalize_evidence_indices(
        raw_reason.get("evidence_indices"),
        suspect_name=suspect_name,
        direction=context,
    )
    if not indices:
        raise SuspectEvidenceAnalysisError(f"{suspect_name}.{context}.evidence_indices must be non-empty")
    missing = [index for index in indices if index not in evidence_lookup]
    if missing:
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{context}.evidence_indices out of range: {missing}"
        )
    reason_statements = _normalize_reason_statements(
        raw_reason.get("reason_statements"),
        suspect_name=suspect_name,
        direction=direction,
        context=context,
    )
    fact_units = _normalize_direction_fact_units(
        raw_reason.get("fact_units"),
        suspect_name=suspect_name,
        direction=context,
    )
    if not fact_units:
        raise SuspectEvidenceAnalysisError(f"{suspect_name}.{context}.fact_units must be non-empty")
    evidence_text = [evidence_lookup[index] for index in indices]
    return {
        "evidence_indices": indices,
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
        tuple(reason.get("evidence_indices") or []),
        tuple(str(fact).casefold() for fact in reason.get("fact_units") or []),
        " ".join(canonical_statement.casefold().split()),
    )


def _validate_distinct_reasons(primary: dict, additional: dict, *, suspect_name: str, direction: str) -> None:
    primary_ids = set(primary.get("evidence_indices") or [])
    additional_ids = set(additional.get("evidence_indices") or [])
    primary_facts = {str(fact).casefold() for fact in primary.get("fact_units") or []}
    additional_facts = {str(fact).casefold() for fact in additional.get("fact_units") or []}
    if _reason_signature(primary) == _reason_signature(additional):
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{direction}.additional_reason duplicates the primary reason"
        )
    if (primary_ids & additional_ids) and (primary_facts & additional_facts):
        raise SuspectEvidenceAnalysisError(
            f"{suspect_name}.{direction}.additional_reason must use different core facts "
            "or a different diagnostic evidence cluster"
        )


def validate_suspect_evidence_analysis(
    analysis: Any,
    *,
    evidence_bank: list[dict],
    suspects: list[dict],
    correct_name: str,
    require_final_fields: bool = False,
) -> list[dict]:
    """Validate and canonicalize suspect evidence analysis.

    Unknown model fields are intentionally dropped. evidence_text and
    is_correct_answer are always rebuilt from trusted local data.
    """
    raw_entries = _analysis_payload_list(analysis)
    expected_names = [s["name"] for s in suspects]
    expected_set = set(expected_names)
    evidence_lookup = _evidence_text_by_index(evidence_bank)

    seen: set[str] = set()
    by_name: dict[str, dict] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise SuspectEvidenceAnalysisError("Each analysis entry must be an object")
        suspect_name = raw_entry.get("suspect_name")
        if not isinstance(suspect_name, str) or not suspect_name.strip():
            raise SuspectEvidenceAnalysisError("Each analysis entry must include suspect_name")
        suspect_name = suspect_name.strip()
        if suspect_name not in expected_set:
            raise SuspectEvidenceAnalysisError(f"Unexpected suspect in analysis: {suspect_name!r}")
        if suspect_name in seen:
            raise SuspectEvidenceAnalysisError(f"Duplicate suspect in analysis: {suspect_name!r}")
        seen.add(suspect_name)

        normalized_entry = {
            "suspect_name": suspect_name,
            "is_correct_answer": suspect_name == correct_name,
        }
        for direction in ("incriminating", "exculpatory"):
            raw_direction = raw_entry.get(direction)
            primary_reason = _normalize_direction_reason(
                raw_direction,
                suspect_name=suspect_name,
                direction=direction,
                evidence_lookup=evidence_lookup,
            )
            additional_reason = _normalize_direction_reason(
                raw_direction.get("additional_reason") if isinstance(raw_direction, dict) else None,
                suspect_name=suspect_name,
                direction=direction,
                evidence_lookup=evidence_lookup,
                label="additional_reason",
            )
            _validate_distinct_reasons(
                primary_reason,
                additional_reason,
                suspect_name=suspect_name,
                direction=direction,
            )
            if require_final_fields:
                if raw_entry.get("is_correct_answer") != (suspect_name == correct_name):
                    raise SuspectEvidenceAnalysisError(
                        f"{suspect_name}.is_correct_answer is missing or incorrect"
                    )
                if raw_direction.get("evidence_text") != primary_reason["evidence_text"]:
                    raise SuspectEvidenceAnalysisError(
                        f"{suspect_name}.{direction}.evidence_text is missing or does not match evidence_bank"
                    )
                raw_additional = raw_direction.get("additional_reason")
                if not isinstance(raw_additional, dict) or raw_additional.get("evidence_text") != additional_reason["evidence_text"]:
                    raise SuspectEvidenceAnalysisError(
                        f"{suspect_name}.{direction}.additional_reason.evidence_text is missing or does not match evidence_bank"
                    )
            normalized_direction = dict(primary_reason)
            normalized_direction["additional_reason"] = additional_reason
            normalized_entry[direction] = normalized_direction
        by_name[suspect_name] = normalized_entry

    missing_names = [name for name in expected_names if name not in seen]
    if missing_names:
        raise SuspectEvidenceAnalysisError(f"Missing suspects in analysis: {missing_names}")

    return [by_name[name] for name in expected_names]


def _selector_role_for_direction(direction: str) -> str:
    if direction == "incriminating":
        return "prosecution selector"
    if direction == "exculpatory":
        return "defense selector"
    raise SuspectEvidenceAnalysisError(f"Unexpected suspect evidence direction: {direction!r}")


def _build_suspect_direction_prompt(
    *,
    case_name: str,
    mystery_text: str,
    evidence_bank: list[dict],
    suspects: list[dict],
    suspect_name: str,
    direction: str,
    correct_name: str,
    outcome_text: str,
    official_clues: list[dict] | None,
    retry_feedback: str | None,
    use_outcome_reference: bool = True,
) -> str:
    feedback_block = ""
    if retry_feedback:
        feedback_block = (
            "\nPrevious attempt failed validation. Fix the next JSON response.\n"
            f"Validation feedback: {retry_feedback}\n"
        )

    selector_role = _selector_role_for_direction(direction)
    side_task = (
        "Find the strongest plausible accusation supported by the full public story for this suspect."
        if direction == "incriminating"
        else "Find the strongest story evidence cluster that could reasonably weaken the accusation against this suspect."
    )
    side_guidance = (
        "For the prosecution side, search the whole public story for the strongest plausible accusation rather than accepting the first explicit suspicion or weak motive statement. "
        "A strong reason may combine motive with demonstrated ability, opportunity, contradiction, suspicious behavior, or another diagnostic clue. "
        "If the official solution implicates this suspect partly by ruling out others, you may use a compact exclusion-comparison cluster."
        if direction == "incriminating"
        else "For the defense side, prioritize loopholes, apparent lack of opportunity, inability, contradictory behavior, alternate-suspect clues, or story details that make responsibility less likely. "
        "For the true culprit, do not try to prove innocence; find the best plausible defense or ambiguity available in the public story."
    )
    outcome_aids_block = (
        f"""
- Outcome reference:
{outcome_text}
- Official public clue candidates extracted from the outcome and mapped to evidence_bank:
{_format_official_clues(official_clues)}
"""
        if use_outcome_reference
        else ""
    )
    source_rules_intro = (
        "1. First use the correct answer and outcome reference to understand the full case."
        if use_outcome_reference
        else "1. Use the full mystery text and numbered evidence_bank to understand the public case."
    )
    official_clue_guidance = (
        """
Prefer official public clue candidates when they help this side's argument for this suspect. It is allowed for the same story evidence to be debatable on both sides, but each statement must explain this side's interpretation without reversing facts.

Official clues may include clues that help the opposite side. Do not copy an opposite-side official clue's reasoning into this side. If you use the same evidence cluster, reinterpret only the selected story facts for the requested side.
"""
        if use_outcome_reference
        else """
No official outcome clues are provided for this search. Use only the full mystery text and numbered evidence_bank to find the strongest plausible public accusation.
"""
    )

    return f"""You are a {selector_role} for evidence-grounded detective debate.

Case name: {case_name}
Target suspect: {suspect_name}
Target side: {direction}

Use these hidden aids only to understand the case:
- Correct answer name: {correct_name}
- Answer options, exactly as parsed:
{_format_suspects(suspects)}
{outcome_aids_block.rstrip()}

Critical source rules:
{source_rules_intro}
2. Then choose evidence only from the numbered evidence_bank below.
3. The final analysis must not use the hidden answer itself as evidence.
4. The final analysis must not cite hidden aids as evidence.
5. The final analysis must not introduce facts, motives, actions, knowledge, or timeline details that are not in the mystery text / selected evidence items.
6. Keep observed story facts separate from argumentative inference.
7. Do not merge separate evidence items into one definite event, object, or causal relation unless the story explicitly states that relation.
8. Any connection not explicitly stated in the story must appear only in a reason_statements statement and must be phrased as a possibility using language such as "could", "may", "might", or "suggests".
9. For weak evidence, use cautious language such as "{_WEAK_EVIDENCE_LANGUAGE}".

Task:
{side_task}
{side_guidance}

Select 2 independent diagnostic evidence clusters:
- primary reason: the strongest concise cluster for this target suspect and side;
- additional_reason: a second complete reason that uses different core facts or a different diagnostic evidence cluster;
- each reason may combine multiple evidence items when the combination produces a stronger case, such as motive plus demonstrated ability, opportunity, contradiction, or suspicious behavior;
- do not add filler evidence.

Independence requirement:
- additional_reason must not be a rewrite, split, subset, reordered version, or paraphrase of the primary reason.
- The two reasons must be usable independently: each must contain its own evidence_indices, fact_units, and reason_statements.
- Prefer disjoint evidence_indices. If the same evidence item is reused, the fact_units must identify different explicit story facts and the canonical statement must diagnose a different issue.

Fact-unit requirements for each reason:
- Generate fact_units specifically for target suspect {suspect_name}, target side {direction}, and the selected interpretations.
- fact_units must contain only the explicit story facts needed for this reason, not every clause from the selected evidence.
- If the same evidence could support both sides, include only the factual portions needed for this side's reason.
- Preserve evidence boundaries and necessary local context.
- Do not include context-dependent fragments whose referent is missing, such as "It wasn't wood", unless the fact unit also supplies the missing referent.
- Do not turn a speculative connection from reasoning into a fact_unit.

Priority when choosing evidence:
physical clue / wordplay / handwriting / ability / access / contradiction > motive / attitude / emotion.

{official_clue_guidance.rstrip()}

Interpretation requirements for each reason:
- Each reason must include reason_statements: a list of at least two interpretation objects.
- Each interpretation object must have a non-empty interpretation_id and a standalone statement.
- interpretation_id values must be unique within the same reason and should use short snake_case labels such as "direct_opportunity", "staged_reaction", "motive_pressure", "ability_inference", "contradiction", or "comparative_exclusion".
- Every statement in a reason must use exactly that reason's selected evidence_indices and fact_units. Do not add any new story fact, motive, action, knowledge, or timeline detail inside a statement.
- The first entry must be the most direct, strongest canonical interpretation of that evidence cluster.
- Return at least two interpretations for every reason. The second and later entries must be equally strong, genuinely different reasoning paths supported by the same facts, not weaker backups.
- If a candidate evidence cluster cannot support at least two equally strong distinct interpretations, choose a different evidence cluster that can. Do not keep a one-interpretation cluster.
- Usually return 2-3 entries; use 4 only when all four are equally strong and distinct.
- Distinct entries must differ in actual inference path, such as motive, staged reaction, opportunity, ability, contradiction, or comparative inference. Do not create multiple entries just by rewording the same path.
- Do not dilute strong evidence with weak or strained interpretations.

Direction requirement for every interpretation:
- If Target side is incriminating, every statement must explain how the evidence could support suspicion, accusation, responsibility, or involvement by {suspect_name}.
- If Target side is exculpatory, every statement must explain how the evidence could weaken suspicion, create doubt, suggest a loophole, or reduce responsibility for {suspect_name}.
- Never return a statement that concludes the opposite of Target side, even if the selected evidence is weak or debatable.
- Plausible interpretations are allowed in statements, but never upgrade them into definite factual claims.
- For example, if one evidence item says Father carried logs and another says a bag containing chess pieces was found in a tree, you may say the logs could suggest greater physical ability than the injury implies, but you must not state that the logs were the bag, that Father carried the chess pieces, or that Father handled the bag.

Do not say there is no evidence. Do not conclude from the answer key. Evidence indices must come from the numbered evidence_bank.

Statements must be short: each statement is one concise sentence, no more than {_SUSPECT_ANALYSIS_REASONING_MAX_WORDS} words. Do not write a long argument. A statement may explain a comparison relationship, e.g. "Uncle Larry's wobbly handwriting compared with the neatly printed note may weaken suspicion against him."

Return strict JSON as one object with this shape and no Markdown code fence:
{{
  "evidence_indices": [1],
  "fact_units": ["Explicit story fact needed for this selected reason."],
  "reason_statements": [
    {{
      "interpretation_id": "canonical_interpretation",
      "statement": "One short sentence explaining why this primary evidence helps this side."
    }},
    {{
      "interpretation_id": "distinct_equally_strong_interpretation",
      "statement": "A second equally strong sentence using the same selected facts but a different reasoning path."
    }}
  ],
  "additional_reason": {{
    "evidence_indices": [2],
    "fact_units": ["Different explicit story fact needed for a separate reason."],
    "reason_statements": [
      {{
        "interpretation_id": "separate_interpretation",
        "statement": "One short sentence explaining why this independent evidence helps this side."
      }},
      {{
        "interpretation_id": "separate_distinct_interpretation",
        "statement": "A second equally strong sentence using the same additional_reason facts but a different reasoning path."
      }}
    ]
  }}
}}

Do not include evidence_text, suspect_name, is_correct_answer, or the opposite side. Python will add final fields.
{feedback_block}
Mystery text:
{mystery_text}

Numbered evidence_bank:
{_format_evidence_bank(evidence_bank)}
"""


def _direction_selection_payload(payload: Any) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("selection"), dict):
        return payload["selection"]
    if isinstance(payload, dict):
        return payload
    raise SuspectEvidenceAnalysisError("Direction selector response must be an object")


def validate_suspect_direction_selection(
    payload: Any,
    *,
    evidence_bank: list[dict],
    suspect_name: str,
    direction: str,
) -> dict:
    raw_selection = _direction_selection_payload(payload)
    evidence_lookup = _evidence_text_by_index(evidence_bank)
    primary_reason = _normalize_direction_reason(
        raw_selection,
        suspect_name=suspect_name,
        direction=direction,
        evidence_lookup=evidence_lookup,
    )
    additional_reason = _normalize_direction_reason(
        raw_selection.get("additional_reason"),
        suspect_name=suspect_name,
        direction=direction,
        evidence_lookup=evidence_lookup,
        label="additional_reason",
    )
    _validate_distinct_reasons(
        primary_reason,
        additional_reason,
        suspect_name=suspect_name,
        direction=direction,
    )
    out = dict(primary_reason)
    out["additional_reason"] = additional_reason
    return out


def select_suspect_direction_evidence(
    *,
    client: OpenRouterClient,
    case_name: str,
    mystery_text: str,
    evidence_bank: list[dict],
    suspects: list[dict],
    suspect_name: str,
    direction: str,
    correct_name: str,
    outcome_text: str,
    official_clues: list[dict] | None,
) -> dict:
    model_cfg = _suspect_evidence_model_config()
    retry_feedback: str | None = None
    last_error: Exception | None = None
    use_outcome_reference = not (
        direction == "incriminating"
        and _name_key(suspect_name) != _name_key(correct_name)
    )
    for attempt in range(1, _SUSPECT_ANALYSIS_MAX_ATTEMPTS + 1):
        try:
            payload = client.complete_json(
                model=model_cfg["model"],
                prompt=_build_suspect_direction_prompt(
                    case_name=case_name,
                    mystery_text=mystery_text,
                    evidence_bank=evidence_bank,
                    suspects=suspects,
                    suspect_name=suspect_name,
                    direction=direction,
                    correct_name=correct_name,
                    outcome_text=outcome_text,
                    official_clues=official_clues,
                    retry_feedback=retry_feedback,
                    use_outcome_reference=use_outcome_reference,
                ),
                temperature=model_cfg["temperature"],
                max_tokens=1500,
            )
            return validate_suspect_direction_selection(
                payload,
                evidence_bank=evidence_bank,
                suspect_name=suspect_name,
                direction=direction,
            )
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            if attempt < _SUSPECT_ANALYSIS_MAX_ATTEMPTS:
                print(
                    f"  {suspect_name} {direction} selector attempt "
                    f"{attempt}/{_SUSPECT_ANALYSIS_MAX_ATTEMPTS} failed: {exc}"
                )
    raise SuspectEvidenceAnalysisError(
        f"Could not select {direction} evidence for {suspect_name!r} "
        f"in {case_name!r} after {_SUSPECT_ANALYSIS_MAX_ATTEMPTS} attempts: {last_error}"
    )


def _print_suspect_analysis_summary(case_name: str, analysis: list[dict], *, reused: bool = False) -> None:
    prefix = "Reusing suspect evidence analysis" if reused else "Analyzing suspect evidence"
    print(f"{prefix}: {case_name}")
    for entry in analysis:
        incr = entry["incriminating"]
        exculp = entry["exculpatory"]
        print(
            f"  {entry['suspect_name']}: "
            f"incriminating={incr['evidence_indices']}({len(incr['reason_statements'])} interpretations), "
            f"incriminating_additional={incr['additional_reason']['evidence_indices']}({len(incr['additional_reason']['reason_statements'])} interpretations), "
            f"exculpatory={exculp['evidence_indices']}({len(exculp['reason_statements'])} interpretations), "
            f"exculpatory_additional={exculp['additional_reason']['evidence_indices']}({len(exculp['additional_reason']['reason_statements'])} interpretations)"
        )


def analyze_suspect_evidence(
    *,
    case_name: str,
    mystery_text: str,
    evidence_bank: list[dict],
    suspects: list[dict],
    analysis_targets: list[dict],
    correct_name: str,
    outcome_text: str,
) -> SuspectEvidenceAnalysisResult:
    client = OpenRouterClient()
    official_clues = extract_official_clues(
        client=client,
        case_name=case_name,
        evidence_bank=evidence_bank,
        suspects=suspects,
        correct_name=correct_name,
        outcome_text=outcome_text,
    )
    print(
        f"  official_clues="
        f"{[clue['evidence_indices'] for clue in official_clues]}"
    )

    raw_analysis: list[dict] = []
    for suspect in analysis_targets:
        suspect_name = suspect["name"]
        raw_analysis.append({
            "suspect_name": suspect_name,
            "incriminating": select_suspect_direction_evidence(
                client=client,
                case_name=case_name,
                mystery_text=mystery_text,
                evidence_bank=evidence_bank,
                suspects=suspects,
                suspect_name=suspect_name,
                direction="incriminating",
                correct_name=correct_name,
                outcome_text=outcome_text,
                official_clues=official_clues,
            ),
            "exculpatory": select_suspect_direction_evidence(
                client=client,
                case_name=case_name,
                mystery_text=mystery_text,
                evidence_bank=evidence_bank,
                suspects=suspects,
                suspect_name=suspect_name,
                direction="exculpatory",
                correct_name=correct_name,
                outcome_text=outcome_text,
                official_clues=official_clues,
            ),
        })

    try:
        analysis = validate_suspect_evidence_analysis(
            raw_analysis,
            evidence_bank=evidence_bank,
            suspects=analysis_targets,
            correct_name=correct_name,
        )
    except Exception as exc:
        raise SuspectEvidenceAnalysisError(
            f"Could not produce valid suspect evidence analysis for {case_name!r}: {exc}"
        ) from exc

    _print_suspect_analysis_summary(case_name, analysis)
    return SuspectEvidenceAnalysisResult(
        analysis=analysis,
        official_clues=official_clues,
    )


def validate_correct_answer_in_suspects(*, correct_name: str, suspects: list[dict], case_name: str) -> None:
    suspect_names = [s["name"] for s in suspects]
    if correct_name not in suspect_names:
        raise ValueError(
            f"Parsed answer {correct_name!r} for {case_name!r} is not in answer_options: "
            f"{suspect_names}. Fix the source answer before running suspect evidence analysis."
        )


def _cleanup_object(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;\"'“”")
    text = re.sub(r"^(?:all of|all|the|a|an|our|my|his|her|their)\b\s+", "", text, flags=re.I)
    text = re.sub(
        r"\s+(?:from|to|at|because|so that|which|who|and|but)\b.*$",
        "",
        text,
        flags=re.I,
    )
    return text.strip(" .,!?:;\"'“”")


def _title_object(case_name: str) -> str | None:
    title = case_name.strip()
    special_titles = {
        "our-quarterback-is-missing": "causing Eddie's disappearance before school",
    }
    patterns = [
        (r"who scratched (.+)", "scratching {}"),
        (r"who stole (.+)", "stealing {}"),
        (r"the stolen (.+)", "stealing the {}"),
        (r"the missing (.+)", "stealing the {}"),
        (r"mystery of the missing (.+)", "stealing the {}"),
        (r"the disappearing (.+)", "stealing the {}"),
        (r"mystery of the disappearing (.+)", "stealing the {}"),
        (r"(.+?) robbery", "robbing {}"),
        (r"(.+?) murder mystery", "committing the murder in {}"),
    ]
    normalized = slugify(title)
    if normalized in special_titles:
        return special_titles[normalized]
    lowered = title.lower()
    for pattern, template in patterns:
        m = re.search(pattern, lowered)
        if not m:
            continue
        obj = m.group(1)
        obj = re.sub(r"\b(mystery|caper|puzzle|conundrum|case)\b", "", obj, flags=re.I)
        obj = _cleanup_object(obj)
        if obj:
            return template.format(obj)
    return None


def infer_wrongdoing_event(case_name: str, mystery_text: str, outcome_text: str) -> str:
    text = " ".join([mystery_text.strip(), outcome_text.strip()])
    lowered_case_name = case_name.strip().lower()
    if lowered_case_name == "the-diamond-necklace":
        return "stealing Eleanor's diamond necklace"
    title_guess = _title_object(case_name)
    if title_guess:
        return title_guess

    if re.search(r"\bslashed tires?\b", text, flags=re.I):
        return "slashing the narrator's car tires outside the psychology office"

    extraction_patterns = [
        (r"\b(?:stole|steal|stolen|shoplifted|pickpocketed|embezzled)\s+(?:(?:all of|all|the|a|an|our|my|his|her|their)\b\s+)?([^.!?\"]+)", "stealing the {}"),
        (r"\b(?:hid|hide|hidden)\s+(?:(?:all of|all|the|a|an|our|my|his|her|their)\b\s+)?([^.!?\"]+)", "hiding the {}"),
        (r"\b(?:scratched|scratch)\s+(?:the|a|an)?\s*([^.!?\"]+)", "scratching the {}"),
        (r"\b(?:blackmailed|blackmailing)\s+([^.!?\"]+)", "blackmailing {}"),
        (r"\b(?:poisoned|poisoning)\s+([^.!?\"]+)", "poisoning {}"),
        (r"\b(?:murdered|killed|shooting|shot|strangled|stabbed)\s+([^.!?\"]+)", "killing {}"),
        (r"\b(?:robbed|robbing)\s+([^.!?\"]+)", "robbing {}"),
        (r"\b(?:set fire to|burned|burnt|torched)\s+([^.!?\"]+)", "setting fire to {}"),
        (r"\b(?:sabotaged|sabotaging)\s+([^.!?\"]+)", "sabotaging {}"),
        (r"\b(?:leaked|leaking)\s+([^.!?\"]+)", "leaking {}"),
        (r"\b(?:kidnapped|abducted)\s+([^.!?\"]+)", "abducting {}"),
    ]
    for pattern, template in extraction_patterns:
        m = re.search(pattern, text, flags=re.I)
        if not m:
            continue
        obj = _cleanup_object(m.group(1))
        if obj:
            return template.format(obj)

    if "murder" in lowered_case_name or "death" in lowered_case_name:
        return "committing the killing at the center of the case"
    if "computer crash" in lowered_case_name:
        return "causing the fatal computer crash"

    return "committing the specific offense described by the case evidence"


def build_pairwise_motion(first_suspect: str, second_suspect: str, wrongdoing_event: str) -> str:
    return (
        f"{format_responsibility_claim(first_suspect, wrongdoing_event)[:-1]} vs. "
        f"{format_responsibility_claim(second_suspect, wrongdoing_event)}"
    )


def load_accusation_results(path: Path = _ACCUSATION_RESULTS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing accusation results file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of accusation result objects.")
    return payload


def _accusation_results_by_url(results: list[dict]) -> dict[str, dict]:
    by_url: dict[str, dict] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"accusation_results.json entry {index} is not an object.")
        url_key = normalize_case_url(result.get("url", ""))
        if not url_key:
            raise ValueError(f"accusation_results.json entry {index} is missing url.")
        previous = by_url.get(url_key)
        if previous is not None:
            raise ValueError(f"Multiple accusation results resolve to URL {url_key!r}.")
        by_url[url_key] = result
    return by_url


def _select_pair_from_accusation_result(
    *,
    row: dict,
    case_id: str,
    suspects: list[dict],
    accusation_results_by_url: dict[str, dict],
) -> dict:
    url_key = normalize_case_url(row.get("case_url", ""))
    if not url_key:
        raise ValueError(f"Case {case_id!r} is missing case_url.")
    accusation_result = accusation_results_by_url.get(url_key)
    if accusation_result is None:
        raise ValueError(f"No accusation result matches case URL for {case_id!r}: {row.get('case_url')!r}")

    csv_correct = parse_answer_name(row.get("answer", ""))
    result_correct = parse_answer_name(str(accusation_result.get("correct_answer", "")))
    culprit = _canonical_option_name(result_correct, suspects)
    if culprit is None:
        raise ValueError(
            f"Accusation-result culprit {result_correct!r} for {case_id!r} is not an answer-option suspect."
        )
    if _name_key(culprit) != _name_key(csv_correct):
        raise ValueError(
            f"Culprit mismatch for {case_id!r}: CSV answer {csv_correct!r} vs "
            f"accusation_results.json {result_correct!r}."
        )

    ranking = accusation_result.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError(f"Accusation result for {case_id!r} has no ranking list.")
    option_by_key = {_name_key(s["name"]): s["name"] for s in suspects}
    incorrect: list[tuple[str, float]] = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        suspect = parse_answer_name(str(item.get("suspect", "")))
        suspect_key = _name_key(suspect)
        if suspect_key == _name_key(culprit):
            continue
        if suspect_key not in option_by_key:
            continue
        try:
            difference = float(item.get("guilty_minus_innocent"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid guilty_minus_innocent for {case_id!r}, suspect {suspect!r}."
            ) from exc
        incorrect.append((option_by_key[suspect_key], difference))
    if not incorrect:
        raise ValueError(f"No non-culprit accusation ranking is available for {case_id!r}.")

    rival, rival_difference = max(incorrect, key=lambda item: item[1])
    if _name_key(rival) == _name_key(culprit):
        raise ValueError(f"Selected rival equals culprit for {case_id!r}: {culprit!r}.")
    return {
        "culprit": culprit,
        "rival": rival,
        "selection_source": "accusation_results.json",
        "rival_guilty_minus_innocent": rival_difference,
    }


def _reason_id_for(suspect_name: str, direction: str) -> str:
    return f"{slugify(suspect_name).replace('-', '_')}_{direction}"


def _additional_reason_id_for(suspect_name: str, direction: str) -> str:
    return f"{_reason_id_for(suspect_name, direction)}_additional"


def _reason_payload_from_raw(
    *,
    raw: dict,
    suspect_name: str,
    direction: str,
    evidence_bank: list[dict],
    reason_id: str,
) -> dict:
    evidence_ids = list(raw["evidence_indices"])
    evidence_lookup = _evidence_text_by_index(evidence_bank)
    evidence_text = [evidence_lookup[index] for index in evidence_ids]
    return {
        "reason_id": reason_id,
        "evidence_ids": evidence_ids,
        "evidence_text": evidence_text,
        "fact_units": observable_fact_units(raw.get("fact_units") or []),
        "reason_statements": [
            {
                "interpretation_id": statement["interpretation_id"],
                "statement": statement["statement"],
            }
            for statement in raw["reason_statements"]
        ],
    }


def _reason_from_analysis_entry(
    *,
    entry: dict,
    direction: str,
    evidence_bank: list[dict],
) -> dict:
    raw = entry[direction]
    reason = _reason_payload_from_raw(
        raw=raw,
        suspect_name=entry["suspect_name"],
        direction=direction,
        evidence_bank=evidence_bank,
        reason_id=_reason_id_for(entry["suspect_name"], direction),
    )
    reason["additional_reason"] = _reason_payload_from_raw(
        raw=raw["additional_reason"],
        suspect_name=entry["suspect_name"],
        direction=direction,
        evidence_bank=evidence_bank,
        reason_id=_additional_reason_id_for(entry["suspect_name"], direction),
    )
    return reason


def build_pairwise_reasoning_finding(
    *,
    analysis: list[dict],
    evidence_bank: list[dict],
    culprit: str,
    rival: str,
) -> dict:
    selected_names = [culprit, rival]
    by_name = {_name_key(entry["suspect_name"]): entry for entry in analysis}
    missing = [name for name in selected_names if _name_key(name) not in by_name]
    if missing:
        raise ValueError(f"Selected suspect(s) missing analysis: {missing}")
    suspects_payload = []
    for suspect_name in selected_names:
        entry = by_name[_name_key(suspect_name)]
        suspects_payload.append({
            "suspect_name": suspect_name,
            "is_correct_answer": _name_key(suspect_name) == _name_key(culprit),
            "incriminating": _reason_from_analysis_entry(
                entry=entry,
                direction="incriminating",
                evidence_bank=evidence_bank,
            ),
            "exculpatory": _reason_from_analysis_entry(
                entry=entry,
                direction="exculpatory",
                evidence_bank=evidence_bank,
            ),
        })
    finding = {
        "reasoning_finding_version": _PAIRWISE_REASONING_FINDING_VERSION,
        "correct_answer": culprit,
        "rival_suspect": rival,
        "suspects": suspects_payload,
    }
    validate_pairwise_reasoning_finding(finding, evidence_bank=evidence_bank, culprit=culprit, rival=rival)
    return finding


def validate_pairwise_reasoning_finding(
    finding: Any,
    *,
    evidence_bank: list[dict],
    culprit: str,
    rival: str,
) -> dict:
    if not isinstance(finding, dict):
        raise ValueError("reasoning_finding must be an object.")
    if finding.get("reasoning_finding_version") != _PAIRWISE_REASONING_FINDING_VERSION:
        raise ValueError(
            "reasoning_finding has an old or missing schema version; regenerate static reasons."
        )
    if _name_key(finding.get("correct_answer", "")) != _name_key(culprit):
        raise ValueError("reasoning_finding.correct_answer does not match selected culprit.")
    if _name_key(finding.get("rival_suspect", "")) != _name_key(rival):
        raise ValueError("reasoning_finding.rival_suspect does not match selected rival.")
    suspects = finding.get("suspects")
    if not isinstance(suspects, list) or len(suspects) != 2:
        raise ValueError("reasoning_finding must contain exactly two suspects.")
    expected = {_name_key(culprit), _name_key(rival)}
    actual = {_name_key(entry.get("suspect_name", "")) for entry in suspects if isinstance(entry, dict)}
    if actual != expected:
        raise ValueError("reasoning_finding contains a suspect outside the selected culprit/rival pair.")
    valid_ids = {item["index"] for item in evidence_bank if "index" in item}
    def validate_reason_payload(reason: Any, *, suspect_name: str, direction: str, label: str) -> dict:
        if not isinstance(reason, dict):
            raise ValueError(f"{suspect_name}.{direction}.{label} must be an object.")
        for field in ("reason_id", "evidence_ids", "evidence_text", "fact_units", "reason_statements"):
            if field not in reason:
                raise ValueError(f"{suspect_name}.{direction}.{label} missing {field}.")
        evidence_ids = reason["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError(f"{suspect_name}.{direction}.{label}.evidence_ids must be non-empty.")
        invalid = [idx for idx in evidence_ids if idx not in valid_ids]
        if invalid:
            raise ValueError(f"{suspect_name}.{direction}.{label} references invalid evidence IDs: {invalid}")
        if not isinstance(reason.get("fact_units"), list) or not reason["fact_units"]:
            raise ValueError(f"{suspect_name}.{direction}.{label}.fact_units must be non-empty.")
        reason_statements = _normalize_reason_statements(
            reason.get("reason_statements"),
            suspect_name=suspect_name,
            direction=direction,
            context=f"{direction}.{label}",
        )
        return {
            "evidence_indices": list(evidence_ids),
            "fact_units": observable_fact_units(reason.get("fact_units") or []),
            "reason_statements": reason_statements,
        }

    for entry in suspects:
        if not isinstance(entry, dict):
            raise ValueError("reasoning_finding suspect entries must be objects.")
        suspect_name = entry.get("suspect_name")
        for direction in ("incriminating", "exculpatory"):
            reason = entry.get(direction)
            primary = validate_reason_payload(
                reason,
                suspect_name=suspect_name,
                direction=direction,
                label="primary",
            )
            additional = validate_reason_payload(
                reason.get("additional_reason") if isinstance(reason, dict) else None,
                suspect_name=suspect_name,
                direction=direction,
                label="additional_reason",
            )
            _validate_distinct_reasons(primary, additional, suspect_name=suspect_name, direction=direction)
    return finding


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------

def convert_row(
    row: dict,
    *,
    case_id: str | None = None,
    display_case_name: str | None = None,
    pairwise_suspects: dict | None = None,
    existing_suspect_evidence_result: SuspectEvidenceAnalysisResult | None = None,
) -> dict:
    if pairwise_suspects is None:
        raise ValueError("convert_row requires preselected pairwise_suspects.")
    suspects = parse_suspects(row["answer_options"])
    if not suspects:
        raise ValueError(f"Could not parse answer_options: {row['answer_options']!r}")

    correct_name = pairwise_suspects["culprit"]
    rival_suspect = pairwise_suspects["rival"]
    validate_correct_answer_in_suspects(
        correct_name=correct_name,
        suspects=suspects,
        case_name=row["case_name"].strip(),
    )
    selected_target_names = {_name_key(correct_name), _name_key(rival_suspect)}
    analysis_targets = [s for s in suspects if _name_key(s["name"]) in selected_target_names]
    if len(analysis_targets) != 2:
        raise ValueError(
            f"Selected pair must match exactly two answer options: {correct_name!r}, {rival_suspect!r}."
        )
    other_suspects = [s["name"] for s in suspects if _name_key(s["name"]) not in selected_target_names]
    evidence_bank = build_evidence_bank(row["mystery_text"])
    source_case_name = row["case_name"].strip()
    output_case_id = case_id or slugify(source_case_name)
    output_case_name = display_case_name or source_case_name
    wrongdoing_event = infer_wrongdoing_event(source_case_name, row["mystery_text"], row["outcome"])
    pairwise_motion = build_pairwise_motion(correct_name, rival_suspect, wrongdoing_event)

    if existing_suspect_evidence_result is not None:
        suspect_evidence_result = existing_suspect_evidence_result
    else:
        suspect_evidence_result = analyze_suspect_evidence(
            case_name=source_case_name,
            mystery_text=row["mystery_text"].strip(),
            evidence_bank=evidence_bank,
            suspects=suspects,
            analysis_targets=analysis_targets,
            correct_name=correct_name,
            outcome_text=row["outcome"].strip(),
        )
    reasoning_finding = build_pairwise_reasoning_finding(
        analysis=suspect_evidence_result.analysis,
        evidence_bank=evidence_bank,
        culprit=correct_name,
        rival=rival_suspect,
    )

    case = {
        "case_id": output_case_id,
        "case_name": output_case_name,
        "wrongdoing_event": wrongdoing_event,
        "motion": pairwise_motion,
        "pairwise_motion": pairwise_motion,
        "debate_format": "evidence_grounded",
        "agent_a_stance": format_responsibility_claim(correct_name, wrongdoing_event),
        "agent_b_stance": format_responsibility_claim(rival_suspect, wrongdoing_event),
        # Full story text — given to agents as background context
        "context": row["mystery_text"].strip(),
        # Sentence-split evidence items — the shared bank both agents reason over
        "evidence_bank": evidence_bank,
        # Hidden metadata: NEVER shown to debate agents; used only for evaluation/logging
        "meta": {
            "correct_answer": correct_name,
            "rival_suspect": rival_suspect,
            "all_suspects": [s["name"] for s in suspects],
            "suspect_options": suspects,
            "other_suspects": other_suspects,
            "answer_raw": row["answer"].strip(),
            "answer_options_raw": row["answer_options"].strip(),
            "solve_rate": float(row["solve_rate"]),
            "outcome_reference": row["outcome"].strip(),
            "case_url": row["case_url"].strip(),
            "author_name": row["author_name"].strip(),
        },
        "pairwise_suspects": pairwise_suspects,
        "reasoning_finding": reasoning_finding,
    }
    case["meta"]["suspect_evidence_official_clues"] = suspect_evidence_result.official_clues
    case["meta"]["suspect_evidence_analysis_version"] = _SUSPECT_ANALYSIS_SCHEMA_VERSION

    return case


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the four configured experiment cases from the True Detective CSV into debate JSON files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE_PATH,
        help="True Detective CSV, or the dataset zip containing it. Default: data/true-detective/data/data.zip.",
    )
    parser.add_argument(
        "--analyze-suspect-evidence",
        action="store_true",
        help="Deprecated compatibility flag; pairwise suspect evidence analysis is always generated or reused.",
    )
    parser.add_argument(
        "--overwrite-analysis",
        action="store_true",
        help="Regenerate suspect evidence analysis even when an existing complete analysis is present.",
    )
    parser.add_argument(
        "--case",
        dest="case_filter",
        help="Only convert one configured case, matching display name, case_id, or original CSV case_name.",
    )
    return parser.parse_args()


def _case_ids_in_order() -> list[str]:
    return [case_id for _display_name, case_id in CASE_SPECS]


def _configured_rows_by_case_id(rows: list[dict]) -> dict[str, dict]:
    required_ids = set(_case_ids_in_order())
    matches: dict[str, list[dict]] = {case_id: [] for case_id in required_ids}
    for row in rows:
        row_case_id = slugify(row.get("case_name", "").strip())
        if row_case_id in required_ids:
            matches[row_case_id].append(row)

    missing = [case_id for _display_name, case_id in CASE_SPECS if not matches[case_id]]
    if missing:
        raise ValueError("Missing required experiment case(s) in the source CSV: " + ", ".join(missing))

    duplicates = [case_id for case_id, case_rows in matches.items() if len(case_rows) > 1]
    if duplicates:
        raise ValueError(
            "Multiple CSV rows resolve to required experiment case ID(s): "
            + ", ".join(sorted(duplicates))
        )

    return {case_id: case_rows[0] for case_id, case_rows in matches.items()}


def _case_filter_aliases(*, display_name: str, case_id: str, source_case_name: str) -> set[str]:
    return {
        display_name.strip().lower(),
        slugify(display_name),
        case_id.strip().lower(),
        source_case_name.strip().lower(),
        slugify(source_case_name),
    }


def _resolve_selected_specs(
    *,
    rows_by_case_id: dict[str, dict],
    case_filter: str | None,
) -> list[tuple[str, str, dict]]:
    if not case_filter:
        return [
            (display_name, case_id, rows_by_case_id[case_id])
            for display_name, case_id in CASE_SPECS
        ]

    wanted = case_filter.strip().lower()
    wanted_slug = slugify(case_filter)
    for display_name, case_id in CASE_SPECS:
        row = rows_by_case_id[case_id]
        aliases = _case_filter_aliases(
            display_name=display_name,
            case_id=case_id,
            source_case_name=row.get("case_name", ""),
        )
        if wanted in aliases or wanted_slug in aliases:
            return [(display_name, case_id, row)]

    raise ValueError(
        f"--case {case_filter!r} is not one of the four configured experiment cases. "
        "Use a configured display name, configured case_id, or original CSV case_name."
    )


def _load_existing_suspect_analysis(
    *,
    out_path: Path,
    evidence_bank: list[dict],
    suspects: list[dict],
    analysis_targets: list[dict],
    correct_name: str,
    rival_suspect: str,
) -> SuspectEvidenceAnalysisResult | None:
    if not out_path.exists():
        return None
    try:
        existing_case = json.loads(out_path.read_text(encoding="utf-8"))
        if existing_case.get("meta", {}).get("suspect_evidence_analysis_version") != _SUSPECT_ANALYSIS_SCHEMA_VERSION:
            print(
                f"  Existing analysis at {out_path.name} uses an older analysis schema; regenerating."
            )
            return None
        finding = existing_case.get("reasoning_finding")
        if finding is None:
            return None
        validate_pairwise_reasoning_finding(
            finding,
            evidence_bank=evidence_bank,
            culprit=correct_name,
            rival=rival_suspect,
        )
        existing_analysis: list[dict] = []
        for entry in finding["suspects"]:
            normalized_entry = {
                "suspect_name": entry["suspect_name"],
                "is_correct_answer": entry["is_correct_answer"],
            }
            for direction in ("incriminating", "exculpatory"):
                reason = entry[direction]
                normalized_entry[direction] = {
                    "evidence_indices": reason["evidence_ids"],
                    "evidence_text": reason["evidence_text"],
                    "fact_units": reason.get("fact_units") or [],
                    "reason_statements": reason["reason_statements"],
                    "additional_reason": {
                        "evidence_indices": reason["additional_reason"]["evidence_ids"],
                        "evidence_text": reason["additional_reason"]["evidence_text"],
                        "fact_units": reason["additional_reason"].get("fact_units") or [],
                        "reason_statements": reason["additional_reason"]["reason_statements"],
                    },
                }
            existing_analysis.append(normalized_entry)
        existing_official_clues = existing_case.get("meta", {}).get("suspect_evidence_official_clues")
        if existing_official_clues is None:
            print(f"  Existing analysis at {out_path.name} has no official clues debug metadata; regenerating.")
            return None
        analysis = validate_suspect_evidence_analysis(
            existing_analysis,
            evidence_bank=evidence_bank,
            suspects=analysis_targets,
            correct_name=correct_name,
            require_final_fields=True,
        )
        official_clues = validate_official_clues(
            {"official_clues": existing_official_clues},
            evidence_bank=evidence_bank,
            suspects=suspects,
        )
        return SuspectEvidenceAnalysisResult(
            analysis=analysis,
            official_clues=official_clues,
        )
    except Exception as exc:
        print(f"  Existing analysis at {out_path.name} is not reusable: {exc}")
        return None


def _index_entry_for_case(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "case_name": case["case_name"],
        "evidence_count": len(case["evidence_bank"]),
        "solve_rate": case["meta"]["solve_rate"],
        "correct_answer": case["meta"]["correct_answer"],
    }


def _write_index(out_dir: Path, index_entries: list[dict], *, case_filter: str | None) -> None:
    index_path = out_dir / "_index.json"
    if case_filter is None:
        index = index_entries
    else:
        existing_index: list[dict] = []
        if index_path.exists():
            try:
                loaded = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing_index = [entry for entry in loaded if isinstance(entry, dict)]
            except json.JSONDecodeError:
                existing_index = []
        update_by_id = {entry["case_id"]: entry for entry in index_entries}
        index = []
        updated_ids = set()
        for entry in existing_index:
            entry_id = entry.get("case_id")
            if entry_id in update_by_id:
                index.append(update_by_id[entry_id])
                updated_ids.add(entry_id)
            else:
                index.append(entry)
        for entry in index_entries:
            if entry["case_id"] not in updated_ids:
                index.append(entry)

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))


def _remove_stale_case_files(out_dir: Path) -> None:
    allowed_ids = set(_case_ids_in_order())
    for path in out_dir.glob("*.json"):
        if path.name == "_index.json":
            continue
        if path.stem not in allowed_ids:
            path.unlink()


def convert_cases(
    *,
    rows: list[dict],
    out_dir: Path,
    analyze_suspect_evidence_flag: bool = False,
    overwrite_analysis: bool = False,
    case_filter: str | None = None,
) -> list[dict]:
    del analyze_suspect_evidence_flag
    rows_by_case_id = _configured_rows_by_case_id(rows)
    selected_specs = _resolve_selected_specs(
        rows_by_case_id=rows_by_case_id,
        case_filter=case_filter,
    )
    accusation_results_by_url = _accusation_results_by_url(load_accusation_results())

    print(f"Converting {len(selected_specs)} configured experiment case(s)...")
    index: list[dict] = []

    for display_name, case_id, row in selected_specs:
        existing_result = None
        source_case_name = row.get("case_name", "?")
        try:
            suspects = parse_suspects(row["answer_options"])
            pairwise_suspects = _select_pair_from_accusation_result(
                row=row,
                case_id=case_id,
                suspects=suspects,
                accusation_results_by_url=accusation_results_by_url,
            )
            correct_name = pairwise_suspects["culprit"]
            rival_suspect = pairwise_suspects["rival"]
            validate_correct_answer_in_suspects(
                correct_name=correct_name,
                suspects=suspects,
                case_name=source_case_name,
            )
            selected_target_names = {_name_key(correct_name), _name_key(rival_suspect)}
            analysis_targets = [s for s in suspects if _name_key(s["name"]) in selected_target_names]
            evidence_bank = build_evidence_bank(row["mystery_text"])
            out_path = out_dir / f"{case_id}.json"
            if not overwrite_analysis:
                existing_result = _load_existing_suspect_analysis(
                    out_path=out_path,
                    evidence_bank=evidence_bank,
                    suspects=suspects,
                    analysis_targets=analysis_targets,
                    correct_name=correct_name,
                    rival_suspect=rival_suspect,
                )
                if existing_result is not None:
                    _print_suspect_analysis_summary(
                        source_case_name.strip(),
                        existing_result.analysis,
                        reused=True,
                    )

            case = convert_row(
                row,
                case_id=case_id,
                display_case_name=display_name,
                pairwise_suspects=pairwise_suspects,
                existing_suspect_evidence_result=existing_result,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert required experiment case {case_id!r} "
                f"({display_name}; source title {source_case_name!r}): {exc}"
            ) from exc

        out_path = out_dir / f"{case['case_id']}.json"
        out_path.write_text(json.dumps(case, ensure_ascii=False, indent=2))
        index.append(_index_entry_for_case(case))

    if case_filter is None:
        _remove_stale_case_files(out_dir)
    _write_index(out_dir, index, case_filter=case_filter)
    return index


def read_source_rows(src: Path) -> list[dict]:
    """Read case rows from the True Detective CSV, or from the zip archive that contains it."""
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"expected exactly one CSV inside {src}, found {members}")
            with archive.open(members[0]) as raw:
                return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")))
    with open(src, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()

    src = args.source
    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        if src == _DEFAULT_SOURCE_PATH:
            print(
                "Clone the dataset first: git clone https://github.com/MaksymDel/true-detective data/true-detective",
                file=sys.stderr,
            )
        sys.exit(1)

    out_dir = _CASES_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_source_rows(src)

    try:
        index = convert_cases(
            rows=rows,
            out_dir=out_dir,
            analyze_suspect_evidence_flag=args.analyze_suspect_evidence,
            overwrite_analysis=args.overwrite_analysis,
            case_filter=args.case_filter,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. {len(index)} configured experiment case(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
