"""Loader for converted True Detective cases.

Returns a topic_cfg-compatible dict that the debate pipeline understands.
The `evidence_bank` key in the returned dict is the signal used by the
rollout and planner to switch to evidence-grounded mode.

Usage:
    from configs.detective_cases import load_case, list_cases
    topic_cfg = load_case("sweat-it-out")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from utils.detective_claims import (
    format_responsibility_claim,
    format_responsibility_question,
)
from urllib.parse import unquote, urlsplit


_CASES_DIR = Path(__file__).resolve().parents[2] / "data" / "detective_cases"
GROUND_TRUTH_SIDE_ORDER_CHOICES = ("gt_first", "gt_second")
_SUSPECT_PREFIX_RE = re.compile(r"^\s*\([a-z0-9]+\)\s*", flags=re.IGNORECASE)
_REASONING_FINDING_VERSION = "pairwise_suspect_reasons_v5"
_SUSPECT_ANALYSIS_SCHEMA_VERSION = "pairwise_side_conditioned_cluster_v5"


def normalize_case_url(url: str) -> str:
    """Normalize a case URL for matching across generated JSON and CSV inputs."""
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


def normalize_suspect_name(name: str) -> str:
    """Remove answer-option prefixes such as '(a)' and normalize spacing."""
    return " ".join(_SUSPECT_PREFIX_RE.sub("", str(name or "")).split()).strip()


def _name_key(name: str) -> str:
    return normalize_suspect_name(name).casefold()


def build_pairwise_detective_motion(first_suspect: str, second_suspect: str, wrongdoing_event: str) -> str:
    """Return the pairwise debate question exposed as the motion."""
    return format_responsibility_question(wrongdoing_event, first_suspect, second_suspect)


def _validate_static_reasoning_finding(raw: dict, *, culprit: str, rival: str) -> dict:
    meta = raw.get("meta") or {}
    if meta.get("suspect_evidence_analysis_version") != _SUSPECT_ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"Detective case {raw.get('case_id')!r} has an old or missing suspect_evidence_analysis_version. "
            f"Expected {_SUSPECT_ANALYSIS_SCHEMA_VERSION!r}; rerun `python code/convert_true_detective.py`."
        )
    finding = raw.get("reasoning_finding")
    if not isinstance(finding, dict):
        raise ValueError(
            f"Detective case {raw.get('case_id')!r} has no static reasoning_finding. "
            "Rerun `python code/convert_true_detective.py`."
        )
    if finding.get("reasoning_finding_version") != _REASONING_FINDING_VERSION:
        raise ValueError(
            f"Detective case {raw.get('case_id')!r} has an old static reasoning_finding schema. "
            "Rerun `python code/convert_true_detective.py`."
        )
    suspects = finding.get("suspects")
    if not isinstance(suspects, list):
        raise ValueError(
            f"Detective case {raw.get('case_id')!r} has an invalid reasoning_finding.suspects value. "
            "Rerun `python code/convert_true_detective.py`."
        )
    expected = {_name_key(culprit), _name_key(rival)}
    actual = {_name_key(entry.get("suspect_name", "")) for entry in suspects if isinstance(entry, dict)}
    if actual != expected or len(suspects) != 2:
        raise ValueError(
            f"Detective case {raw.get('case_id')!r} reasoning_finding must contain exactly the culprit and rival. "
            "Rerun `python code/convert_true_detective.py`."
        )
    evidence_ids = {item.get("index") for item in raw.get("evidence_bank", [])}
    primary_counts: list[tuple[str, int]] = []
    additional_counts: list[tuple[str, int]] = []

    def validate_reason_statements(reason: dict, *, label: str) -> int:
        if "reason_statement" in reason:
            raise ValueError(
                f"Detective case {raw.get('case_id')!r} has obsolete scalar {label}.reason_statement. "
                "Expected v5 reason_statements interpretation list."
            )
        statements = reason.get("reason_statements")
        if not isinstance(statements, list) or len(statements) < 2:
            raise ValueError(
                f"Detective case {raw.get('case_id')!r} is missing {label}.reason_statements. "
                "Rerun `python code/convert_true_detective.py`."
            )
        seen_ids: set[str] = set()
        for index, statement_entry in enumerate(statements, start=1):
            if not isinstance(statement_entry, dict):
                raise ValueError(
                    f"Detective case {raw.get('case_id')!r} has invalid {label}.reason_statements[{index}]. "
                    "Expected objects with interpretation_id and statement."
                )
            interpretation_id = str(statement_entry.get("interpretation_id") or "").strip()
            statement = str(statement_entry.get("statement") or "").strip()
            if not interpretation_id or not statement:
                raise ValueError(
                    f"Detective case {raw.get('case_id')!r} has invalid {label}.reason_statements[{index}]. "
                    "Both interpretation_id and statement are required."
                )
            if interpretation_id in seen_ids:
                raise ValueError(
                    f"Detective case {raw.get('case_id')!r} has duplicate interpretation_id "
                    f"{interpretation_id!r} in {label}."
                )
            seen_ids.add(interpretation_id)
        return len(statements)

    for entry in suspects:
        for direction in ("incriminating", "exculpatory"):
            reason = entry.get(direction)
            label = f"{entry.get('suspect_name')}.{direction}"
            if not isinstance(reason, dict):
                raise ValueError(
                    f"Detective case {raw.get('case_id')!r} is missing "
                    f"{entry.get('suspect_name')}.{direction} reasoning_statements. "
                    "Rerun `python code/convert_true_detective.py`."
                )
            primary_counts.append((label, validate_reason_statements(reason, label=label)))
            invalid_ids = [idx for idx in reason.get("evidence_ids", []) if idx not in evidence_ids]
            if invalid_ids:
                raise ValueError(
                    f"Detective case {raw.get('case_id')!r} has invalid evidence IDs in "
                    f"{entry.get('suspect_name')}.{direction}: {invalid_ids}. "
                    "Rerun `python code/convert_true_detective.py`."
                )
            additional = reason.get("additional_reason")
            additional_label = f"{label}.additional_reason"
            if not isinstance(additional, dict):
                raise ValueError(
                    f"Detective case {raw.get('case_id')!r} is missing "
                    f"{entry.get('suspect_name')}.{direction}.additional_reason.reason_statements. "
                    "Rerun `python code/convert_true_detective.py`."
                )
            additional_counts.append((additional_label, validate_reason_statements(additional, label=additional_label)))
            invalid_additional_ids = [idx for idx in additional.get("evidence_ids", []) if idx not in evidence_ids]
            if invalid_additional_ids:
                raise ValueError(
                    f"Detective case {raw.get('case_id')!r} has invalid evidence IDs in "
                    f"{entry.get('suspect_name')}.{direction}.additional_reason: {invalid_additional_ids}. "
                    "Rerun `python code/convert_true_detective.py`."
                )
    if len({count for _label, count in primary_counts}) != 1:
        raise ValueError(
            f"Detective case {raw.get('case_id')!r} has inconsistent primary interpretation counts: "
            + ", ".join(f"{label}={count}" for label, count in primary_counts)
        )
    if len({count for _label, count in additional_counts}) != 1:
        raise ValueError(
            f"Detective case {raw.get('case_id')!r} has inconsistent additional interpretation counts: "
            + ", ".join(f"{label}={count}" for label, count in additional_counts)
        )
    return finding


def load_case(case_id: str, ground_truth_side_order: str = "gt_first", *, log_pair: bool = False) -> dict:
    """Load a detective case and return a topic_cfg-compatible dict.

    Keys returned:
        Standard pipeline keys (expected by planner / rollout):
            topic           — run-specific pairwise motion used as topic text
            question        — same pairwise motion, exposed separately for prompts
            agent_a_stance  — side assigned to Agent A for this run
            agent_b_stance  — side assigned to Agent B for this run
        Evidence-grounded keys (detected by planner / rollout):
            evidence_bank   — list of {"index": int, "text": str}
            case_context     — original case text, hidden from agents but useful
                               for the viewer
        Metadata keys (available in pipeline for logging/evaluation):
            case_id         — slug
            case_name       — human-readable title
            ground_truth_side_order — "gt_first" or "gt_second"
            correct_answer  — the guilty party name (hidden from agents)
            all_suspects    — list of all suspect names
            suspect_options — answer-option-derived suspect records (hidden)
            solve_rate      — float, human solve rate from the original dataset
            outcome_reference — full solution text (hidden from agents)
            wrongdoing_event — concrete offense phrase used in the debate question
    """
    path = _CASES_DIR / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Detective case '{case_id}' not found at {path}.\n"
            f"Run `python code/convert_true_detective.py` to generate cases first."
        )
    if ground_truth_side_order not in GROUND_TRUTH_SIDE_ORDER_CHOICES:
        raise ValueError(
            f"Unknown ground_truth_side_order '{ground_truth_side_order}'. "
            f"Expected one of: {', '.join(GROUND_TRUTH_SIDE_ORDER_CHOICES)}"
        )
    raw = json.loads(path.read_text())
    pairwise_suspects = raw.get("pairwise_suspects")
    if not isinstance(pairwise_suspects, dict):
        raise ValueError(
            f"Detective case {case_id!r} has no pairwise_suspects block. "
            "Rerun `python code/convert_true_detective.py`."
        )
    correct_answer = normalize_suspect_name(pairwise_suspects.get("culprit", ""))
    rival_suspect = normalize_suspect_name(pairwise_suspects.get("rival", ""))
    if not correct_answer or not rival_suspect:
        raise ValueError(
            f"Detective case {case_id!r} has an incomplete pairwise_suspects block. "
            "Rerun `python code/convert_true_detective.py`."
        )
    if _name_key(correct_answer) == _name_key(rival_suspect):
        raise ValueError(f"Detective case {case_id!r} has duplicate culprit/rival suspects.")
    reasoning_finding = _validate_static_reasoning_finding(raw, culprit=correct_answer, rival=rival_suspect)

    wrongdoing_event = raw.get("wrongdoing_event")
    if not wrongdoing_event:
        raise ValueError(f"Detective case {case_id!r} is missing wrongdoing_event.")
    side_a_suspect = correct_answer if ground_truth_side_order == "gt_first" else rival_suspect
    side_b_suspect = rival_suspect if ground_truth_side_order == "gt_first" else correct_answer
    agent_a_stance = format_responsibility_claim(side_a_suspect, wrongdoing_event)
    agent_b_stance = format_responsibility_claim(side_b_suspect, wrongdoing_event)
    if ground_truth_side_order == "gt_first":
        ground_truth_supporting_speaker = "A"
        ground_truth_opposing_speaker = "B"
    else:
        ground_truth_supporting_speaker = "B"
        ground_truth_opposing_speaker = "A"
    normalized_motion = build_pairwise_detective_motion(side_a_suspect, side_b_suspect, wrongdoing_event)
    pair_metadata = {
        **pairwise_suspects,
        "correct_suspect": correct_answer,
        "rival_suspect": rival_suspect,
        "rival_difference": pairwise_suspects.get("rival_guilty_minus_innocent"),
        "case_url": raw.get("case_url") or raw.get("meta", {}).get("case_url"),
        "ground_truth_side_order": ground_truth_side_order,
        "side_a_suspect": side_a_suspect,
        "side_b_suspect": side_b_suspect,
        "side_a_is_ground_truth": ground_truth_side_order == "gt_first",
        "side_b_is_ground_truth": ground_truth_side_order == "gt_second",
    }
    if log_pair:
        print(
            f"case={raw['case_id']} correct={correct_answer} rival={rival_suspect} "
            f"rival_difference={pairwise_suspects.get('rival_guilty_minus_innocent')} "
            f"order={ground_truth_side_order}"
        )
    return {
        # Standard topic_cfg shape
        "topic": normalized_motion,
        "question": normalized_motion,
        "agent_a_stance": agent_a_stance,
        "agent_b_stance": agent_b_stance,
        # Evidence-grounded extension — presence of this key activates evidence mode
        "evidence_bank": raw["evidence_bank"],
        "case_context": raw.get("context"),
        # Metadata
        "case_id": raw["case_id"],
        "case_name": raw["case_name"],
        "ground_truth_side_order": ground_truth_side_order,
        "ground_truth_supporting_speaker": ground_truth_supporting_speaker,
        "ground_truth_opposing_speaker": ground_truth_opposing_speaker,
        "wrongdoing_event": wrongdoing_event,
        "motion": normalized_motion,
        "correct_answer": correct_answer,
        "rival_suspect": rival_suspect,
        "side_a_suspect": side_a_suspect,
        "side_b_suspect": side_b_suspect,
        "pairwise_suspects": pairwise_suspects,
        "pairwise_suspect_pair": pair_metadata,
        "all_suspects": raw["meta"]["all_suspects"],
        "suspect_options": raw["meta"].get(
            "suspect_options",
            [{"name": name} for name in raw["meta"]["all_suspects"]],
        ),
        "other_suspects": raw["meta"].get("other_suspects", []),
        "reasoning_finding": reasoning_finding,
        "converted_case_source": str(path),
        "solve_rate": raw["meta"]["solve_rate"],
        "outcome_reference": raw["meta"]["outcome_reference"],
    }


def list_cases() -> list[dict]:
    """Return the index of all converted cases."""
    index_path = _CASES_DIR / "_index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text())
