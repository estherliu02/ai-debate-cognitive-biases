from __future__ import annotations

from collections import Counter

from configs import bias_runtime, traitsV2
from configs.models import MODEL_CONFIGS
from core.trait_evaluators import build_verbosity_gap_report
from prompts.evaluator_detective_v2 import (
    BIAS_OPTIONS,
    build_detective_holistic_bias_audit_prompt,
    build_detective_bias_presence_prompt,
    build_detective_bias_side_selection_prompt,
)
from schemas.eval_schema import (
    BiasSideSummary,
    BiasTurnAggregate,
    CrossModelDetectiveEvalResult,
    DetectiveBiasEvalResult,
    TurnBiasScore,
)

TRAIT_LIBRARY = traitsV2.TRAIT_LIBRARY
_SEMANTIC_BIAS_TYPE_KEYS = {key for key, _ in BIAS_OPTIONS}
_HOLISTIC_SEMANTIC_BIAS_KEYS = (
    "anchoring_bias",
    # DISABLED: confirmation_bias removed from the current experiment.
    # "confirmation_bias",
    "pro_jargon_bias",
    "fallacy_trait",
)
_BASELINE_CONTROL_TRAIT_NAMES = {"N/A", "__baseline_control__"}
_NO_BIAS_TYPE = "none"


def _judge_status_from_match_count(match_count: int) -> str:
    if match_count == 3:
        return "pass"
    if match_count in {1, 2}:
        return "weak_pass"
    return "fail"


def _is_baseline_control_trait(trait_name: str) -> bool:
    return trait_name in _BASELINE_CONTROL_TRAIT_NAMES


def _infer_bias_type(trait_name: str) -> str:
    if _is_baseline_control_trait(trait_name):
        raise ValueError("N/A baseline control has no expected bias type.")
    if trait_name.startswith("fallacy_trait__"):
        return "fallacy_trait"
    if trait_name == "verbosity_bias":
        return trait_name
    if trait_name in _SEMANTIC_BIAS_TYPE_KEYS:
        return trait_name
    raise ValueError(f"Unsupported bias trait for detective bias eval: {trait_name}")


def _coerce_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y"}:
            return True
        if normalized in {"false", "no", "n"}:
            return False
    return None


def _normalize_speaker_choice(value) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {
        "a": "A",
        "speaker_a": "A",
        "speaker a": "A",
        "agent_a": "A",
        "agent a": "A",
        "side_a": "A",
        "side a": "A",
        "b": "B",
        "speaker_b": "B",
        "speaker b": "B",
        "agent_b": "B",
        "agent b": "B",
        "side_b": "B",
        "side b": "B",
        "both": "Both",
        "both_speakers": "Both",
        "both speakers": "Both",
        "a_and_b": "Both",
        "a and b": "Both",
        "a+b": "Both",
        "neither": "neither",
        "none": "neither",
        "no_one": "neither",
    }
    return aliases.get(normalized)


def _expected_biased_speaker_for_side_selection(speaker_variants: dict[str, str], trait_name: str) -> str | None:
    if trait_name == "verbosity_bias":
        return "neither"
    if _is_baseline_control_trait(trait_name):
        if speaker_variants.get("A") == "baseline" and speaker_variants.get("B") == "baseline":
            return "neither"
        return None
    if speaker_variants.get("A") == "active" and speaker_variants.get("B") == "baseline":
        return "A"
    if speaker_variants.get("A") == "baseline" and speaker_variants.get("B") == "active":
        return "B"
    if speaker_variants.get("A") == "baseline" and speaker_variants.get("B") == "baseline":
        return "neither"
    return None


def _normalize_bias_type(value, *, trait_name: str | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"none", "no_bias", "neither", "not_present", "absent"}:
        return _NO_BIAS_TYPE
    aliases = {
        "anchoring": "anchoring_bias",
        # DISABLED: confirmation_bias removed from the current experiment.
        # "confirmation": "confirmation_bias",
        "fallacy": "fallacy_trait",
        "logical_fallacy": "fallacy_trait",
        "flawed_reasoning": "fallacy_trait",
        "pro_jargon": "pro_jargon_bias",
        "professional_jargon": "pro_jargon_bias",
        "jargon": "pro_jargon_bias",
        # "sentiment": "sentiment_bias",
        # "emotional_framing": "sentiment_bias",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in _SEMANTIC_BIAS_TYPE_KEYS:
        return normalized
    if normalized.startswith("fallacy_trait__"):
        return normalized
    if normalized == "fallacy_trait":
        return trait_name if isinstance(trait_name, str) and trait_name.startswith("fallacy_trait__") else normalized
    if isinstance(trait_name, str) and trait_name.startswith("fallacy_trait__"):
        subtrait_key = trait_name.removeprefix("fallacy_trait__")
        if normalized == subtrait_key:
            return trait_name
    return normalized if normalized in _SEMANTIC_BIAS_TYPE_KEYS else None


def _compact_eval_reason(value: object, *, max_chars: int = 220) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Evaluator did not provide a usable side-selection reason."
    cleaned = " ".join(value.split())
    sentence_ends = [cleaned.find(mark) for mark in (".", "!", "?") if cleaned.find(mark) != -1]
    if sentence_ends:
        cleaned = cleaned[: min(sentence_ends) + 1]
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "."


def _expected_bias_type_for_side_selection(trait_name: str, expected_biased_speaker: str) -> str:
    if expected_biased_speaker == "neither":
        return _NO_BIAS_TYPE
    if trait_name.startswith("fallacy_trait__"):
        return trait_name
    if trait_name in _SEMANTIC_BIAS_TYPE_KEYS:
        return trait_name
    if trait_name == "fallacy_trait":
        return trait_name
    raise ValueError(f"Unsupported trait for holistic bias eval: {trait_name!r}")


def _target_bias_family(trait_name: str) -> str | None:
    if _is_baseline_control_trait(trait_name):
        return None
    if trait_name == "verbosity_bias":
        return "verbosity_bias"
    if trait_name.startswith("fallacy_trait__"):
        return "fallacy_trait"
    if trait_name in _HOLISTIC_SEMANTIC_BIAS_KEYS:
        return trait_name
    return None


def _format_content_only_context(turn_plan) -> str:
    if turn_plan is None:
        return "No previous opposing-turn content summary."
    parts = []
    claim = getattr(turn_plan, "claim_to_defend", None)
    if claim:
        parts.append(f"Previous opposing-side claim: {claim}")
    point = getattr(turn_plan, "opponent_point_to_attack", None)
    if point:
        parts.append(f"Previous opposing-side target: {point}")
    move = getattr(turn_plan, "required_move", None)
    if move:
        parts.append(f"Previous opposing-side required move: {move}")
    return "\n".join(parts) if parts else "No previous opposing-turn content summary."


def _target_marker_guidance(trait_name: str, variant_name: str, speaker: str) -> str:
    if _is_baseline_control_trait(trait_name):
        return (
            "No assigned target trait. The target turn should use ordinary evidence-based "
            "courtroom reasoning without an assigned bias style."
        )
    try:
        return TRAIT_LIBRARY[trait_name][variant_name]["eval_guidance"].format(s=speaker)
    except KeyError as exc:
        raise ValueError(
            f"Missing eval_guidance for trait={trait_name!r}, variant={variant_name!r}."
        ) from exc


def _trait_active_field(trait_name: str, field_name: str) -> str | None:
    return _trait_variant_field(trait_name, "active", field_name)


def _trait_variant_field(trait_name: str, variant_name: str, field_name: str) -> str | None:
    cfg = TRAIT_LIBRARY.get(trait_name) or traitsV2.TRAIT_LIBRARY.get(trait_name)
    if cfg is None and trait_name.startswith("fallacy_trait__") and variant_name == "active":
        subtrait_key = trait_name.removeprefix("fallacy_trait__")
        subtrait_cfg = (
            traitsV2.TRAIT_LIBRARY
            .get("fallacy_trait", {})
            .get("subtraits", {})
            .get(subtrait_key)
        )
        if isinstance(subtrait_cfg, dict):
            value = subtrait_cfg.get(field_name)
            return value if isinstance(value, str) else None
    if not isinstance(cfg, dict):
        return None
    variant = cfg.get(variant_name)
    if not isinstance(variant, dict):
        return None
    value = variant.get(field_name)
    return value if isinstance(value, str) else None


def _fallacy_subdefinition_entries() -> list[tuple[str, str, str | None]]:
    entries: list[tuple[str, str, str | None]] = []
    for key, cfg in TRAIT_LIBRARY.items():
        if not key.startswith("fallacy_trait__") or not isinstance(cfg, dict):
            continue
        active = cfg.get("active")
        if not isinstance(active, dict):
            continue
        description = active.get("definition")
        eval_guidance = active.get("eval_guidance")
        entries.append((
            key,
            description if isinstance(description, str) else "No description provided.",
            eval_guidance if isinstance(eval_guidance, str) else None,
        ))
    if entries:
        return entries

    subtraits = (
        traitsV2.TRAIT_LIBRARY
        .get("fallacy_trait", {})
            .get("subtraits", {})
    )
    for subtrait_key, subtrait_cfg in subtraits.items():
        if not isinstance(subtrait_cfg, dict):
            continue
        description = subtrait_cfg.get("definition")
        eval_guidance = subtrait_cfg.get("eval_guidance")
        entries.append((
            f"fallacy_trait__{subtrait_key}",
            description if isinstance(description, str) else "No description provided.",
            eval_guidance if isinstance(eval_guidance, str) else None,
        ))
    return entries


def _build_holistic_bias_definitions(trait_name: str) -> str:
    entries: list[tuple[str, str, str | None]] = []
    for key in ("anchoring_bias", "pro_jargon_bias"):
        entries.append((
            key,
            _trait_active_field(key, "definition") or "No definition provided.",
            _trait_active_field(key, "eval_guidance"),
        ))

    if trait_name.startswith("fallacy_trait__"):
        entries.append((
            trait_name,
            _trait_active_field(trait_name, "definition")
            or _trait_active_field("fallacy_trait", "definition")
            or "Uses a recognizable logical-fallacy reasoning pattern.",
            _trait_active_field(trait_name, "eval_guidance"),
        ))
    else:
        entries.append((
            "fallacy_trait",
            _trait_active_field("fallacy_trait", "definition")
            or "Uses a recognizable logical-fallacy reasoning pattern.",
            _trait_active_field("fallacy_trait", "eval_guidance"),
        ))
        entries.extend(_fallacy_subdefinition_entries())

    lines: list[str] = []
    for key, description, eval_guidance in entries:
        lines.append(f"- {key}")
        lines.append(f"  Definition: {description}")
        if eval_guidance:
            lines.append(f"  Active eval_guidance: {eval_guidance.format(s='the biased speaker')}")
        if key == trait_name:
            baseline_description = _trait_variant_field(key, "baseline", "definition")
            baseline_guidance = _trait_variant_field(key, "baseline", "eval_guidance")
            if baseline_description:
                lines.append(f"  Non-biased contrast: {baseline_description}")
            if baseline_guidance:
                lines.append(
                    f"  Baseline eval_guidance: {baseline_guidance.format(s='the non-biased speaker')}"
                )
    return "\n".join(lines)


def _turn_type_for_id(turn_id: int, total_turns: int) -> str:
    if turn_id <= 2:
        return "opening"
    if turn_id > total_turns - 2:
        return "final_focus"
    if turn_id > total_turns - 4:
        return "summary"
    return "rebuttal"


def _dialogue_to_bias_eval_text(dialogue, trait_name: str | None = None) -> str:
    dialogue_turns = dialogue.turns if hasattr(dialogue, "turns") else dialogue["turns"]
    total_turns = len(dialogue_turns)
    lines = []
    for turn in dialogue_turns:
        turn_id = turn.turn_id if hasattr(turn, "turn_id") else turn["turn_id"]
        speaker = turn.speaker if hasattr(turn, "speaker") else turn["speaker"]
        turn_type = _turn_type_for_id(turn_id, total_turns)
        if turn_type not in {"rebuttal", "summary"}:
            # Bias transformations are only evaluated on Turn 3-6.
            # lines.append(f"Turn {turn_id} | {speaker}: {utterance}")
            continue
        utterance = turn.utterance if hasattr(turn, "utterance") else turn["utterance"]
        lines.append(f"Turn {turn_id} | {speaker}: {utterance}")
    return "\n".join(lines)


def _target_evaluation_turn_rule(trait_name: str, variant_name: str) -> dict:
    del trait_name, variant_name
    return {
        "turn_types": list(bias_runtime.PAIRWISE_BIAS_TURN_TYPES),
        "reason": "Bias guidance is evaluated on pairwise argument and rebuttal turns.",
    }


def _turn_required_for_guidance_pass(
    *,
    trait_name: str,
    variant_name: str,
    turn_id: int,
    turn_type: str,
    speaker_turn_index: int,
) -> tuple[bool, str]:
    if turn_type not in set(bias_runtime.PAIRWISE_BIAS_TURN_TYPES):
        return False, "only Turn 3-6 argument and rebuttal turns are checked for bias guidance"

    if variant_name == "baseline":
        return True, "baseline variants require every Turn 3-6 argument and rebuttal turn to satisfy baseline guidance"

    rule = _target_evaluation_turn_rule(trait_name, variant_name)
    reason = str(rule.get("reason") or "configured by pairwise pipeline invariants")
    turn_ids = set(rule.get("turn_ids") or [])
    speaker_turn_indices = set(rule.get("speaker_turn_indices") or [])
    turn_types = set(rule.get("turn_types") or [])

    if turn_ids and turn_id in turn_ids:
        return True, reason
    if speaker_turn_indices and speaker_turn_index in speaker_turn_indices:
        return True, reason
    if turn_types and turn_type in turn_types:
        return True, reason
    return False, reason


def _build_side_summaries(turn_outcomes: list[BiasTurnAggregate], speaker_variants: dict[str, str]) -> list[BiasSideSummary]:
    side_summaries: list[BiasSideSummary] = []
    for speaker in ("A", "B"):
        speaker_turns = [item for item in turn_outcomes if item.speaker == speaker]
        required_speaker_turns = [item for item in speaker_turns if item.required_for_pass]
        total_speaker_turns = len(required_speaker_turns)
        passed_speaker_turns = sum(1 for item in required_speaker_turns if item.passed)
        non_applicable_turns = len(speaker_turns) - total_speaker_turns
        side_summaries.append(
            BiasSideSummary(
                speaker=speaker,
                variant_name=speaker_variants[speaker],
                passed_turns=passed_speaker_turns,
                total_turns=total_speaker_turns,
                non_applicable_turns=non_applicable_turns,
                passed=total_speaker_turns > 0 and passed_speaker_turns == total_speaker_turns,
            )
        )
    return side_summaries


def _build_deterministic_verbosity_eval(dialogue, speaker_variants: dict[str, str]) -> CrossModelDetectiveEvalResult:
    verbosity_report = build_verbosity_gap_report(dialogue, speaker_variants)
    dialogue_turns = dialogue.turns if hasattr(dialogue, "turns") else dialogue["turns"]
    turn_outcomes: list[BiasTurnAggregate] = []
    for turn in dialogue_turns:
        turn_id = turn.turn_id if hasattr(turn, "turn_id") else turn["turn_id"]
        speaker = turn.speaker if hasattr(turn, "speaker") else turn["speaker"]
        variant_name = speaker_variants[speaker]
        expected_has_bias = variant_name == "active"
        turn_outcomes.append(
            BiasTurnAggregate(
                turn_id=turn_id,
                speaker=speaker,
                variant_name=variant_name,
                expected_has_bias=expected_has_bias,
                expected_bias_type="verbosity_bias" if expected_has_bias else None,
                models_passing=1 if verbosity_report["passed"] else 0,
                models_total=1,
                passed=verbosity_report["passed"],
            )
        )

    side_summaries = _build_side_summaries(turn_outcomes, speaker_variants)
    passed_turns = sum(1 for item in turn_outcomes if item.passed)
    total_turns = len(turn_outcomes)
    failure_modes = [] if verbosity_report["passed"] else ["deterministic_verbosity_gap_failed"]
    deterministic_details = {
        "trait_name": "verbosity_bias",
        "measure": verbosity_report["measure"],
        "threshold": verbosity_report["threshold"],
        "mode": verbosity_report["mode"],
        "gap": verbosity_report["gap"],
        "speaker_stats": verbosity_report["speaker_stats"],
        "reason": verbosity_report["reason"],
    }
    print(
        "[debug] detective verbosity eval (deterministic): "
        f"mode={verbosity_report['mode']} gap={verbosity_report['gap']} "
        f"threshold={verbosity_report['threshold']} passed={verbosity_report['passed']}"
    )
    return CrossModelDetectiveEvalResult(
        evaluation_type="deterministic",
        deterministic_details=deterministic_details,
        per_model_results=[],
        turn_outcomes=turn_outcomes,
        side_summaries=side_summaries,
        passed_turns=passed_turns,
        total_turns=total_turns,
        models_with_all_turns_passing=1 if verbosity_report["passed"] else 0,
        passed=verbosity_report["passed"],
        reason=verbosity_report["reason"],
        failure_modes=failure_modes,
    )


def _build_computational_verbosity_verification(
    dialogue,
    speaker_variants: dict[str, str],
    trait_name: str,
) -> dict:
    report = dict(build_verbosity_gap_report(dialogue, speaker_variants))
    if trait_name == "verbosity_bias":
        passed = bool(report.get("passed"))
        reason = report.get("reason") or "Computational verbosity check completed."
    else:
        threshold = report.get("threshold", bias_runtime.get_verbosity_word_gap_threshold())
        round_gaps = report.get("round_gaps") or []
        pair_absolute_gaps = [
            int(item.get("absolute_gap", 0))
            for item in round_gaps
            if item.get("comparable") is True
        ]
        absolute_gap = max(pair_absolute_gaps, default=0)
        passed = bool(pair_absolute_gaps) and all(gap <= threshold for gap in pair_absolute_gaps)
        report["mode"] = "non_verbosity_pairwise_isolation"
        report["absolute_gap"] = absolute_gap
        report["pair_absolute_gaps"] = pair_absolute_gaps
        reason = (
            "Computational verbosity isolation check: "
            f"both paired absolute gaps for Turn 3/4 and Turn 5/6 must be <= {threshold}; "
            f"pair_absolute_gaps={pair_absolute_gaps}."
        )

    report.update({
        "trait_name": trait_name,
        "passed": passed,
        "reason": reason,
    })
    return report


def _build_deterministic_verbosity_only_eval(
    dialogue,
    speaker_variants: dict[str, str],
) -> CrossModelDetectiveEvalResult:
    details = _build_computational_verbosity_verification(
        dialogue,
        speaker_variants,
        "verbosity_bias",
    )
    passed = bool(details.get("passed"))
    status = "pass" if passed else "fail"
    failure_modes = [] if passed else ["computational_verbosity_failed"]
    return CrossModelDetectiveEvalResult(
        evaluation_type="deterministic",
        deterministic_details=details,
        computational_verbosity_passed=passed,
        verbosity_length_difference=details.get("gap"),
        verbosity_abs_length_difference=details.get("absolute_gap"),
        verbosity_threshold=details.get("threshold"),
        judge_match_count=None,
        judge_total_count=0,
        judge_status=None,
        llm_judge_passed=None,
        final_bias_eval_status=status,
        final_bias_eval_passed=passed,
        expected_biased_speaker="neither",
        predicted_biased_speaker=None,
        side_selection_correct=None,
        side_selection_model_votes={},
        bias_present=None,
        expected_bias_type="none",
        predicted_bias_type=None,
        bias_type_correct=None,
        bias_type_model_votes={},
        per_model_results=[],
        turn_outcomes=[],
        side_summaries=[
            BiasSideSummary(
                speaker=speaker,
                variant_name=speaker_variants[speaker],
                passed_turns=0,
                total_turns=0,
                non_applicable_turns=0,
                passed=passed,
            )
            for speaker in ("A", "B")
        ],
        passed_turns=1 if passed else 0,
        total_turns=1,
        non_applicable_turns=0,
        models_with_all_turns_passing=0,
        passed=passed,
        reason=details.get("reason", "Deterministic verbosity check completed."),
        failure_modes=failure_modes,
    )


def _normalize_audit_entry(value: object, *, include_subtype: bool = False) -> dict:
    payload = value if isinstance(value, dict) else {}
    present = _coerce_bool(payload.get("present"))
    if present is None:
        present = False
    speaker = _normalize_speaker_choice(payload.get("speaker")) or "neither"
    if not present:
        speaker = "neither"
    reason = _compact_eval_reason(payload.get("reason"), max_chars=260)
    out = {
        "present": present,
        "speaker": speaker,
        "reason": reason,
    }
    if include_subtype:
        subtype = payload.get("subtype")
        if not isinstance(subtype, str) or not subtype.strip() or not present:
            subtype = "none"
        subtype = subtype.strip()
        if subtype != "none" and not subtype.startswith("fallacy_trait__"):
            subtype = "fallacy_trait__" + subtype
        out["subtype"] = subtype
    return out


def _normalize_holistic_audit_payload(payload: object) -> dict[str, dict]:
    raw = payload if isinstance(payload, dict) else {}
    return {
        "anchoring_bias": _normalize_audit_entry(raw.get("anchoring_bias")),
        # DISABLED: confirmation_bias removed from the current experiment.
        # "confirmation_bias": _normalize_audit_entry(raw.get("confirmation_bias")),
        "pro_jargon_bias": _normalize_audit_entry(raw.get("pro_jargon_bias")),
        "fallacy_trait": _normalize_audit_entry(raw.get("fallacy_trait"), include_subtype=True),
    }


def _speaker_has_bias(entry: dict, speaker: str) -> bool:
    if not entry.get("present"):
        return False
    return entry.get("speaker") in {speaker, "Both"}


def _entry_has_any_bias(entry: dict) -> bool:
    return bool(entry.get("present")) and entry.get("speaker") != "neither"


def _selected_fallacy_subtype(trait_name: str) -> str | None:
    return trait_name if trait_name.startswith("fallacy_trait__") else None


def _evaluate_holistic_audit_pass(
    *,
    audit: dict[str, dict],
    trait_name: str,
    speaker_variants: dict[str, str],
) -> tuple[bool, list[str], str | None, str | None, bool | None, bool | None]:
    target_family = _target_bias_family(trait_name)
    active_speaker = next((speaker for speaker, variant in speaker_variants.items() if variant == "active"), None)
    baseline_speaker = next((speaker for speaker, variant in speaker_variants.items() if variant == "baseline"), None)
    predicted_speaker: str | None = None
    predicted_bias_type: str | None = None
    side_selection_correct: bool | None = None
    bias_type_correct: bool | None = None
    failure_modes: list[str] = []

    if target_family == "verbosity_bias" or target_family is None:
        for family, entry in audit.items():
            if _entry_has_any_bias(entry):
                suffix = "jargon" if family == "pro_jargon_bias" else "fallacy" if family == "fallacy_trait" else family.removesuffix("_bias")
                failure_modes.append(f"non_target_{suffix}_present")
        predicted_speaker = "neither"
        predicted_bias_type = "none"
        side_selection_correct = True
        bias_type_correct = True
        return not failure_modes, failure_modes, predicted_speaker, predicted_bias_type, side_selection_correct, bias_type_correct

    target_entry = audit[target_family]
    predicted_speaker = target_entry.get("speaker") or "neither"
    predicted_bias_type = trait_name if target_family == "fallacy_trait" and target_entry.get("subtype") != "none" else target_family
    if target_family == "fallacy_trait" and target_entry.get("subtype") != trait_name:
        failure_modes.append("target_bias_missing")
        bias_type_correct = False
    else:
        bias_type_correct = True

    if active_speaker is None:
        failure_modes.append("target_bias_missing")
        side_selection_correct = False
    elif not _speaker_has_bias(target_entry, active_speaker):
        failure_modes.append("target_bias_missing")
        side_selection_correct = False
    elif target_entry.get("speaker") != active_speaker:
        failure_modes.append("target_bias_on_wrong_speaker")
        side_selection_correct = False
    else:
        side_selection_correct = True

    if baseline_speaker and _speaker_has_bias(target_entry, baseline_speaker):
        failure_modes.append("target_bias_leaked_to_baseline")
        side_selection_correct = False

    for family, entry in audit.items():
        if family == target_family:
            continue
        if _entry_has_any_bias(entry):
            suffix = "jargon" if family == "pro_jargon_bias" else "fallacy" if family == "fallacy_trait" else family.removesuffix("_bias")
            failure_modes.append(f"non_target_{suffix}_present")

    return not failure_modes, failure_modes, predicted_speaker, predicted_bias_type, side_selection_correct, bias_type_correct


class DialogueEvaluator:
    def __init__(self, client):
        self.client = client

    def evaluate_computational_verbosity(
        self,
        dialogue,
        speaker_variants: dict[str, str],
        trait_name: str,
    ) -> dict:
        return _build_computational_verbosity_verification(
            dialogue,
            speaker_variants,
            trait_name,
        )

    def _evaluate_holistic_bias_audit(
        self,
        dialogue,
        trait_name: str,
        speaker_variants: dict[str, str],
        computational_verbosity_details: dict,
    ) -> CrossModelDetectiveEvalResult:
        model_cfgs = MODEL_CONFIGS.get("detective_bias_evaluators") or [MODEL_CONFIGS["evaluator"]]
        topic = dialogue.topic if hasattr(dialogue, "topic") else dialogue.get("topic")
        expected_biased_speaker = _expected_biased_speaker_for_side_selection(speaker_variants, trait_name) or "neither"
        target_family = _target_bias_family(trait_name)
        expected_bias_type = (
            "none"
            if target_family in {None, "verbosity_bias"}
            else trait_name if trait_name.startswith("fallacy_trait__") else target_family
        )
        dialogue_text = _dialogue_to_bias_eval_text(dialogue, trait_name=trait_name)
        prompt = build_detective_holistic_bias_audit_prompt(
            topic=topic,
            trait_name=trait_name,
            selected_fallacy_subtype=_selected_fallacy_subtype(trait_name),
            dialogue_text=dialogue_text,
        )

        per_model_results: list[DetectiveBiasEvalResult] = []
        speaker_vote_counts: Counter[str] = Counter()
        bias_type_vote_counts: Counter[str] = Counter()
        holistic_audits: list[dict] = []

        for cfg in model_cfgs:
            try:
                payload = self.client.complete_json(
                    model=cfg["model"],
                    prompt=prompt,
                    temperature=cfg.get("temperature", 0.1),
                    max_tokens=cfg.get("max_tokens", 3000),
                )
                audit = _normalize_holistic_audit_payload(payload)
                semantic_passed, semantic_failure_modes, predicted_speaker, predicted_bias_type, side_correct, type_correct = (
                    _evaluate_holistic_audit_pass(
                        audit=audit,
                        trait_name=trait_name,
                        speaker_variants=speaker_variants,
                    )
                )
                reason = "; ".join(
                    audit[key].get("reason", "")
                    for key in _HOLISTIC_SEMANTIC_BIAS_KEYS
                    if audit[key].get("reason")
                )
            except Exception as exc:
                audit = {}
                semantic_passed = False
                semantic_failure_modes = ["model_eval_error"]
                predicted_speaker = None
                predicted_bias_type = None
                side_correct = False
                type_correct = False
                payload = {"error": f"{type(exc).__name__}: {exc}"}
                reason = f"Evaluator model failed: {type(exc).__name__}"

            holistic_audits.append(audit)
            speaker_vote_counts[predicted_speaker or "malformed"] += 1
            bias_type_vote_counts[predicted_bias_type or "malformed"] += 1

            per_model_results.append(
                DetectiveBiasEvalResult(
                    model=cfg["model"],
                    turn_scores=[],
                    expected_biased_speaker=expected_biased_speaker,
                    predicted_biased_speaker=predicted_speaker,
                    side_selection_correct=side_correct,
                    bias_present=predicted_bias_type not in {None, "none"},
                    expected_bias_type=expected_bias_type,
                    predicted_bias_type=predicted_bias_type,
                    bias_type_correct=type_correct,
                    confidence=None,
                    correct_question_1_turns=1 if semantic_passed else 0,
                    correct_question_2_turns=0,
                    passed_turns=1 if semantic_passed else 0,
                    total_turns=1,
                    passed_turns_a=1 if semantic_passed and expected_biased_speaker == "A" else 0,
                    total_turns_a=1 if expected_biased_speaker == "A" else 0,
                    passed_turns_b=1 if semantic_passed and expected_biased_speaker == "B" else 0,
                    total_turns_b=1 if expected_biased_speaker == "B" else 0,
                    non_applicable_turns=0,
                    passed=semantic_passed,
                    reason=_compact_eval_reason(reason, max_chars=260),
                    failure_modes=semantic_failure_modes,
                    raw_payload={
                        "holistic_bias_audit": audit,
                        "raw_payload": payload if isinstance(payload, dict) else None,
                    },
                )
            )

        semantic_pass_count = sum(1 for result in per_model_results if result.passed)
        judge_status = _judge_status_from_match_count(semantic_pass_count)
        llm_judge_passed = judge_status in {"pass", "weak_pass"}
        computational_verbosity_passed = bool(computational_verbosity_details.get("passed"))
        final_bias_eval_status = judge_status if computational_verbosity_passed else "fail"
        passed = final_bias_eval_status in {"pass", "weak_pass"}

        predicted_biased_speaker = None
        if speaker_vote_counts:
            predicted_biased_speaker = speaker_vote_counts.most_common(1)[0][0]
            if predicted_biased_speaker == "malformed":
                predicted_biased_speaker = None
        predicted_bias_type = None
        if bias_type_vote_counts:
            predicted_bias_type = bias_type_vote_counts.most_common(1)[0][0]
            if predicted_bias_type == "malformed":
                predicted_bias_type = None

        failure_modes: list[str] = []
        if not computational_verbosity_passed:
            failure_modes.append(
                "verbosity_target_failed" if trait_name == "verbosity_bias" else "verbosity_isolation_failed"
            )
        if judge_status == "fail":
            seen: set[str] = set()
            for result in per_model_results:
                for mode in result.failure_modes:
                    if mode not in seen:
                        seen.add(mode)
                        failure_modes.append(mode)
            if not seen:
                failure_modes.append("holistic_audit_failed")

        side_summaries = [
            BiasSideSummary(
                speaker=speaker,
                variant_name=speaker_variants[speaker],
                passed_turns=0,
                total_turns=0,
                non_applicable_turns=0,
                passed=passed,
            )
            for speaker in ("A", "B")
        ]
        reason = (
            f"Holistic Turns 3-6 bias audit: {semantic_pass_count}/{len(per_model_results)} "
            f"model(s) satisfied semantic pass rules; judge_status={judge_status}. "
            f"{computational_verbosity_details.get('reason')}"
        )
        deterministic_details = {
            "holistic_bias_audits": holistic_audits,
            "computational_verbosity": computational_verbosity_details,
            "evaluated_turn_scope": "turns_3_to_6",
            "pass_rules": {
                "target_trait": trait_name,
                "expected_biased_speaker": expected_biased_speaker,
                "expected_bias_type": expected_bias_type,
            },
        }

        return CrossModelDetectiveEvalResult(
            evaluation_type="llm_holistic_bias_audit",
            deterministic_details=deterministic_details,
            computational_verbosity_passed=computational_verbosity_passed,
            verbosity_length_difference=computational_verbosity_details.get("gap"),
            verbosity_abs_length_difference=computational_verbosity_details.get("absolute_gap"),
            verbosity_threshold=computational_verbosity_details.get("threshold"),
            judge_match_count=semantic_pass_count,
            judge_total_count=len(per_model_results),
            judge_status=judge_status,
            llm_judge_passed=llm_judge_passed,
            final_bias_eval_status=final_bias_eval_status,
            final_bias_eval_passed=passed,
            expected_biased_speaker=expected_biased_speaker,
            predicted_biased_speaker=predicted_biased_speaker,
            side_selection_correct=predicted_biased_speaker == expected_biased_speaker,
            side_selection_model_votes=dict(speaker_vote_counts),
            bias_present=predicted_bias_type not in {None, "none"},
            expected_bias_type=expected_bias_type,
            predicted_bias_type=predicted_bias_type,
            bias_type_correct=predicted_bias_type == expected_bias_type,
            bias_type_model_votes=dict(bias_type_vote_counts),
            per_model_results=per_model_results,
            turn_outcomes=[],
            side_summaries=side_summaries,
            passed_turns=1 if passed else 0,
            total_turns=1,
            non_applicable_turns=0,
            models_with_all_turns_passing=semantic_pass_count,
            passed=passed,
            reason=reason,
            failure_modes=failure_modes,
        )

    def _evaluate_bias_side_selection(
        self,
        dialogue,
        trait_name: str,
        speaker_variants: dict[str, str],
        expected_biased_speaker: str,
        computational_verbosity_details: dict | None = None,
        current_trait_only: bool = True,
    ) -> CrossModelDetectiveEvalResult:
        model_cfgs = MODEL_CONFIGS.get("detective_bias_evaluators") or [MODEL_CONFIGS["evaluator"]]
        topic = dialogue.topic if hasattr(dialogue, "topic") else dialogue.get("topic")
        expected_bias_type = _expected_bias_type_for_side_selection(trait_name, expected_biased_speaker)
        prompt = build_detective_bias_side_selection_prompt(
            topic=topic,
            trait_name=trait_name,
            bias_definitions=_build_holistic_bias_definitions(trait_name),
            dialogue_text=_dialogue_to_bias_eval_text(dialogue, trait_name=trait_name),
            current_trait_only=current_trait_only,
        )
        per_model_results: list[DetectiveBiasEvalResult] = []
        speaker_vote_counts: Counter[str] = Counter()
        bias_type_vote_counts: Counter[str] = Counter()

        for cfg in model_cfgs:
            try:
                payload = self.client.complete_json(
                    model=cfg["model"],
                    prompt=prompt,
                    temperature=cfg.get("temperature", 0.1),
                    max_tokens=cfg.get("max_tokens", 3000),
                )
            except Exception as exc:
                print(
                    f"[warning] detective side-selection evaluator failed "
                    f"({cfg['model']}): {type(exc).__name__}: {exc}"
                )
                per_model_results.append(
                    DetectiveBiasEvalResult(
                        model=cfg["model"],
                        turn_scores=[],
                        expected_biased_speaker=expected_biased_speaker,
                        predicted_biased_speaker=None,
                        side_selection_correct=False,
                        bias_present=None,
                        expected_bias_type=expected_bias_type,
                        predicted_bias_type=None,
                        bias_type_correct=False,
                        confidence=None,
                        correct_question_1_turns=0,
                        correct_question_2_turns=0,
                        passed_turns=0,
                        total_turns=1,
                        passed_turns_a=0,
                        total_turns_a=1 if expected_biased_speaker == "A" else 0,
                        passed_turns_b=0,
                        total_turns_b=1 if expected_biased_speaker == "B" else 0,
                        non_applicable_turns=0,
                        passed=False,
                        reason=f"Evaluator model failed: {type(exc).__name__}",
                        failure_modes=["model_eval_error"],
                        raw_payload={"error": f"{type(exc).__name__}: {exc}"},
                    )
                )
                continue

            print(f"[debug] detective side-selection payload ({cfg['model']}):", payload)
            payload_dict = payload if isinstance(payload, dict) else {}

            predicted_biased_speaker = _normalize_speaker_choice(
                payload_dict.get("predicted_biased_speaker")
            )
            predicted_bias_type = _normalize_bias_type(payload_dict.get("bias_type"), trait_name=trait_name)
            speaker_vote_key = predicted_biased_speaker or "malformed"
            bias_type_vote_key = predicted_bias_type or "malformed"
            speaker_vote_counts[speaker_vote_key] += 1
            bias_type_vote_counts[bias_type_vote_key] += 1
            side_selection_correct = predicted_biased_speaker == expected_biased_speaker
            bias_type_correct = predicted_bias_type == expected_bias_type
            fallacy_check_keys = (
                "content_preservation",
                "public_fact_grounding",
                "target_fallacy_realization",
                "absence_of_second_dominant_fallacy",
                "naturalness_coherence_stance",
            )
            fallacy_checks = {
                key: _coerce_bool(payload_dict.get(key))
                for key in fallacy_check_keys
                if key in payload_dict
            }
            fallacy_checks_passed = None
            if trait_name.startswith("fallacy_trait__") and fallacy_checks:
                fallacy_checks_passed = all(value is True for value in fallacy_checks.values())
            model_passed = (
                side_selection_correct and bias_type_correct
                if expected_biased_speaker == "neither"
                else side_selection_correct
            )
            if fallacy_checks_passed is False:
                model_passed = False
            failure_modes = []
            if not side_selection_correct:
                failure_modes.append("wrong_biased_speaker_selected")
            if not bias_type_correct:
                failure_modes.append("wrong_bias_type_selected")
            if fallacy_checks_passed is False:
                failure_modes.extend(
                    f"fallacy_{key}_failed"
                    for key, value in fallacy_checks.items()
                    if value is not True
                )
            reason = _compact_eval_reason(payload_dict.get("reason"))
            bias_present = _coerce_bool(payload_dict.get("bias_present"))

            per_model_results.append(
                DetectiveBiasEvalResult(
                    model=cfg["model"],
                    turn_scores=[],
                    expected_biased_speaker=expected_biased_speaker,
                    predicted_biased_speaker=predicted_biased_speaker,
                    side_selection_correct=side_selection_correct,
                    bias_present=bias_present,
                    expected_bias_type=expected_bias_type,
                    predicted_bias_type=predicted_bias_type,
                    bias_type_correct=bias_type_correct,
                    confidence=None,
                    correct_question_1_turns=1 if model_passed else 0,
                    correct_question_2_turns=0,
                    passed_turns=1 if model_passed else 0,
                    total_turns=1,
                    passed_turns_a=1 if model_passed and expected_biased_speaker == "A" else 0,
                    total_turns_a=1 if expected_biased_speaker == "A" else 0,
                    passed_turns_b=1 if model_passed and expected_biased_speaker == "B" else 0,
                    total_turns_b=1 if expected_biased_speaker == "B" else 0,
                    non_applicable_turns=0,
                    passed=model_passed,
                    reason=reason,
                    failure_modes=failure_modes,
                    raw_payload=payload_dict or None,
                )
            )

        correct_model_count = sum(1 for result in per_model_results if result.passed)
        requires_baseline_neither = (
            expected_biased_speaker == "neither"
            and speaker_variants.get("A") == "baseline"
            and speaker_variants.get("B") == "baseline"
        )
        if requires_baseline_neither:
            judge_status = "pass" if correct_model_count == len(per_model_results) else (
                "weak_pass" if correct_model_count >= 2 else "fail"
            )
        else:
            judge_status = _judge_status_from_match_count(correct_model_count)
        llm_judge_passed = judge_status in {"pass", "weak_pass"}
        predicted_biased_speaker = None
        if speaker_vote_counts:
            predicted_biased_speaker = speaker_vote_counts.most_common(1)[0][0]
            if predicted_biased_speaker == "malformed":
                predicted_biased_speaker = None
        predicted_bias_type = None
        if bias_type_vote_counts:
            predicted_bias_type = bias_type_vote_counts.most_common(1)[0][0]
            if predicted_bias_type == "malformed":
                predicted_bias_type = None
        side_selection_correct = predicted_biased_speaker == expected_biased_speaker
        bias_type_correct = predicted_bias_type == expected_bias_type
        computational_verbosity_passed = (
            computational_verbosity_details.get("passed")
            if computational_verbosity_details is not None
            else None
        )
        final_bias_eval_status = judge_status
        if computational_verbosity_passed is False:
            final_bias_eval_status = "fail"
        passed = final_bias_eval_status in {"pass", "weak_pass"}
        side_summaries = [
            BiasSideSummary(
                speaker=speaker,
                variant_name=speaker_variants[speaker],
                passed_turns=0 if expected_biased_speaker == "neither" else 1 if speaker == expected_biased_speaker and llm_judge_passed else 0,
                total_turns=0 if expected_biased_speaker == "neither" else 1 if speaker == expected_biased_speaker else 0,
                non_applicable_turns=0,
                passed=(llm_judge_passed if expected_biased_speaker == "neither" or speaker == expected_biased_speaker else True),
            )
            for speaker in ("A", "B")
        ]
        failure_modes = []
        if judge_status == "fail":
            if requires_baseline_neither and correct_model_count < 2:
                failure_modes.append("baseline_neither_below_threshold")
            if not side_selection_correct:
                failure_modes.append("biased_side_not_identified")
            if not bias_type_correct:
                failure_modes.append("bias_type_not_identified")
            if not failure_modes:
                failure_modes.append("no_judge_label_matches")
        if computational_verbosity_passed is False:
            failure_modes.append("computational_verbosity_failed")
        reason = (
            f"{correct_model_count}/{len(per_model_results)} evaluator model(s) selected "
            f"the expected label {expected_biased_speaker}; judge status={judge_status}. "
            f"Expected bias type was {expected_bias_type}."
        )
        if not current_trait_only and expected_biased_speaker == "neither":
            reason = (
                "Non-verbosity bias audit requires no speaker to show "
                f"pro-jargon or fallacy bias. {reason}"
            )
        if requires_baseline_neither:
            reason = (
                "Baseline/baseline control requires at least 2/3 neither. "
                f"{reason}"
            )
        if computational_verbosity_details is not None:
            reason = f"{computational_verbosity_details['reason']} LLM judge: {reason}"

        print("[debug] detective side-selection eval per-model scores:")
        for result in per_model_results:
            print(
                f"  - {result.model}: expected={result.expected_biased_speaker} "
                f"predicted={result.predicted_biased_speaker} "
                f"expected_type={result.expected_bias_type} predicted_type={result.predicted_bias_type} "
                f"speaker_correct={result.side_selection_correct} type_correct={result.bias_type_correct}"
            )
        print(
            "[debug] detective side-selection eval aggregate: "
            f"expected={expected_biased_speaker} predicted={predicted_biased_speaker} "
            f"expected_type={expected_bias_type} predicted_type={predicted_bias_type} "
            f"correct_models={correct_model_count}/{len(per_model_results)} "
            f"judge_status={judge_status} final_status={final_bias_eval_status} "
            f"final_passed={passed} failure_modes={failure_modes}"
        )

        return CrossModelDetectiveEvalResult(
            evaluation_type="llm_side_selection",
            deterministic_details=computational_verbosity_details,
            computational_verbosity_passed=computational_verbosity_passed,
            verbosity_length_difference=(
                computational_verbosity_details.get("gap")
                if computational_verbosity_details is not None
                else None
            ),
            verbosity_abs_length_difference=(
                computational_verbosity_details.get("absolute_gap")
                if computational_verbosity_details is not None
                else None
            ),
            verbosity_threshold=(
                computational_verbosity_details.get("threshold")
                if computational_verbosity_details is not None
                else None
            ),
            judge_match_count=correct_model_count,
            judge_total_count=len(per_model_results),
            judge_status=judge_status,
            llm_judge_passed=llm_judge_passed,
            final_bias_eval_status=final_bias_eval_status,
            final_bias_eval_passed=passed,
            expected_biased_speaker=expected_biased_speaker,
            predicted_biased_speaker=predicted_biased_speaker,
            side_selection_correct=side_selection_correct,
            side_selection_model_votes=dict(speaker_vote_counts),
            bias_present=None if predicted_bias_type is None else predicted_bias_type != _NO_BIAS_TYPE,
            expected_bias_type=expected_bias_type,
            predicted_bias_type=predicted_bias_type,
            bias_type_correct=bias_type_correct,
            bias_type_model_votes=dict(bias_type_vote_counts),
            per_model_results=per_model_results,
            turn_outcomes=[],
            side_summaries=side_summaries,
            passed_turns=1 if passed else 0,
            total_turns=1,
            non_applicable_turns=0,
            models_with_all_turns_passing=correct_model_count,
            passed=passed,
            reason=reason,
            failure_modes=failure_modes,
        )

    def evaluate_evidence_debate_v2(
        self,
        dialogue,
        content_plan,
        evidence_bank: list[dict],
        trait_name: str,
        speaker_variants: dict[str, str],
        include_computational_verbosity: bool = True,
    ) -> CrossModelDetectiveEvalResult:
        del evidence_bank

        del content_plan, include_computational_verbosity
        computational_verbosity_details = _build_computational_verbosity_verification(
            dialogue,
            speaker_variants,
            trait_name,
        )
        return self._evaluate_holistic_bias_audit(
            dialogue=dialogue,
            trait_name=trait_name,
            speaker_variants=speaker_variants,
            computational_verbosity_details=computational_verbosity_details,
        )
