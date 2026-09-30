from __future__ import annotations

import re

from configs.models import MODEL_CONFIGS
from configs import traitsV2
from prompts.content_attention_question import build_content_attention_question_prompt
from prompts.content_planner_detective import build_detective_planner_prompt
from prompts.style_planner import build_style_prompt
from core.reasoning_finder_detective import interpretation_dimensions
from schemas.plan_schema import ContentAttentionQuestion, ContentPlan, TransformationPlan, TurnPlan
from utils.content_attention import (
    build_default_content_attention_question,
    validate_content_attention_question_payload,
)
from utils.detective_claims import (
    accusation_clause_for_suspect,
    build_claim_realization_prompt,
    claim_realization_bundle_to_dict,
    deterministic_claim_realization_bundle,
    format_motion_from_claim_bundle,
    guilt_conclusion_for_suspect,
    innocence_conclusion_for_suspect,
    normalize_wrongdoing_event,
    validate_claim_realization_payload,
)
from utils.fact_units import observable_fact_units
from utils.reason_text import normalize_anchor_reason

_DETECTIVE_PLAN_MAX_TOKENS = 12000
_ATTENTION_QUESTION_MAX_TOKENS = 700
_POST_OPENING_TURN_TYPES = {
    3: "rebuttal",
    4: "rebuttal",
    5: "summary",
    6: "summary",
    7: "final_focus",
    8: "final_focus",
}
_SEMANTIC_CONTRACT_FIELDS = (
    "turn_type",
    "speaker",
    "evidence_ids",
    "fact_units",
    "interpretation_units",
    "opponent_claim_target",
    "required_concession",
    "core_conclusion",
    "inference_edges",
    "certainty_level",
)
_OPENING_DISABLED_FIELDS = (
    "turn_goal",
    "claim_to_defend",
    "case_theory",
    "narrative_beats",
    "required_move",
    "allowed_concession",
    "required_concession",
    "core_conclusion",
    "inference_edges",
    "certainty_level",
    "end_state",
)
_CERTAINTY_LEVELS = {"low", "medium", "high", "mixed"}
_COMPACT_REBUTTAL_TURN_IDS = {3, 4}
_COMPACT_SUMMARY_TURN_IDS = {5, 6}
_REQUIRED_REASON_UNIT_ROLES = {
    "suspect_a_incriminating",
    "suspect_a_exculpatory",
    "suspect_b_incriminating",
    "suspect_b_exculpatory",
}
PAIRWISE_CONTENT_PLAN_VERSION = "pairwise_role_slots_v2"
CONTENT_PLAN_FAMILIES = {"standard", "verbosity"}


def _fallacy_subtype_from_trait_name(trait_name: str) -> str | None:
    if trait_name.startswith("fallacy_trait__"):
        return trait_name.removeprefix("fallacy_trait__")
    return None


class Planner:
    def __init__(self, client):
        self.client = client

    def build_deterministic_style_plan(
        self,
        agent_name: str,
        trait_name: str,
        variant_name: str,
        trait_rules: list[str],
    ) -> TransformationPlan:
        del agent_name, trait_rules
        return self.build_deterministic_transformation_plan(
            trait_name=trait_name,
            variant_name=variant_name,
        )

    def build_deterministic_transformation_plan(
        self,
        *,
        trait_name: str,
        variant_name: str,
    ) -> TransformationPlan:
        base_surface_operations = [
            "present the locked public facts and conclusions without changing them",
            "apply style only in argument and rebuttal reasoning sentences",
        ]
        base_inference_operations = [
            "facts, evidence IDs, claim target, stance, and conclusion come from the content plan",
            "reasoning style comes from the selected style guideline",
        ]
        fallacy_subtype = _fallacy_subtype_from_trait_name(trait_name)
        if fallacy_subtype is not None:
            subtrait_cfg = traitsV2.TRAIT_LIBRARY["fallacy_trait"]["subtraits"][fallacy_subtype]
            baseline_cfg = traitsV2.TRAIT_LIBRARY["fallacy_trait"]["baseline"]
            active_definition = str(subtrait_cfg["definition"])
            baseline_definition = str(baseline_cfg["definition"])
            branch_metadata = {
                "fallacy_subtype": fallacy_subtype,
                "bias_definition": subtrait_cfg,
                "baseline_definition": baseline_cfg,
            }
            if variant_name == "active":
                allowed_inference_operations = [
                    active_definition,
                    "change only the subtype-required reasoning, opponent-claim representation, or attention relation in wording",
                    "preserve locked content fields and content-unit IDs",
                ]
                allowed_surface_operations = [
                    "render the target fallacy naturally without naming it",
                    "keep all locked factual premises and citations",
                    "use the grounded hook_unit_ids from branch_metadata",
                ]
            else:
                allowed_inference_operations = [
                    baseline_definition,
                    "use valid proportional reasoning from the locked semantic contract",
                    "preserve locked content fields and content-unit IDs",
                ]
                allowed_surface_operations = [
                    "clear and proportionate presentation",
                    "accurate opponent-claim representation",
                    "ordinary evidence-based courtroom reasoning",
                ]
            return TransformationPlan(
                trait_name=trait_name,
                variant=variant_name,
                anchor_unit_id=None,
                fallacy_subtype=fallacy_subtype,
                hook_unit_ids=[],
                branch_metadata=branch_metadata,
                ordered_unit_ids=[],
                foreground_unit_ids=[],
                background_unit_ids=[],
                repeat_unit_ids=[],
                allowed_surface_operations=allowed_surface_operations,
                allowed_inference_operations=allowed_inference_operations,
            )
        trait_cfg = traitsV2.TRAIT_LIBRARY.get(trait_name, {})
        variant_cfg = trait_cfg.get(variant_name, {}) if isinstance(trait_cfg, dict) else {}
        definition = str(variant_cfg.get("definition", "")) if isinstance(variant_cfg, dict) else ""
        if trait_name == "pro_jargon_bias":
            surface_operations = [
                "use the baseline dialogue structure and baseline reasoning plan",
                "defer the treatment to the speaker rollout rewrite stage",
                "preserve sentence functions, sentence order, citations, approximate length, and factual wording constraints",
            ]
            inference_operations = [
                "preserve baseline reasoning relations and conclusions",
                "do not create sentence-specific jargon targets or alter diagnostic framing",
                "do not introduce new evidence, arguments, methods, standards, or reasoning",
            ]
            return TransformationPlan(
                trait_name=trait_name,
                variant=variant_name,
                anchor_unit_id=None,
                ordered_unit_ids=[],
                foreground_unit_ids=[],
                background_unit_ids=[],
                repeat_unit_ids=[],
                allowed_surface_operations=surface_operations,
                allowed_inference_operations=inference_operations,
            )
        if variant_name == "baseline":
            surface_operations = [
                *base_surface_operations,
                "use plain courtroom wording",
            ]
            inference_operations = [
                *base_inference_operations,
                definition or "use valid, proportionate reasoning",
            ]
        else:
            surface_operations = [
                *base_surface_operations,
                "render the selected trait naturally without naming it",
            ]
            inference_operations = [
                *base_inference_operations,
                definition or "apply the selected trait to the reasoning relation",
            ]
        return TransformationPlan(
            trait_name=trait_name,
            variant=variant_name,
            anchor_unit_id=None,
            ordered_unit_ids=[],
            foreground_unit_ids=[],
            background_unit_ids=[],
            repeat_unit_ids=[],
            allowed_surface_operations=surface_operations,
            allowed_inference_operations=inference_operations,
        )

    def _validate_turn_count(self, payload: dict, turns_per_agent: int) -> None:
        expected_turns = turns_per_agent * 2
        actual_turns = len(payload.get("turns", []))
        if actual_turns != expected_turns:
            raise ValueError(
                f"Planner returned {actual_turns} turns; expected {expected_turns}."
            )

    def _validate_content_attention_question(
        self,
        payload: dict,
        evidence_bank: list[dict],
    ) -> ContentAttentionQuestion:
        return validate_content_attention_question_payload(
            payload.get("content_attention_question"),
            evidence_bank=evidence_bank,
        )

    def _validate_suspect_evidence_map(
        self,
        payload: dict,
        *,
        evidence_bank: list[dict],
        correct_answer: str | None,
        all_suspects: list[str] | None,
    ) -> None:
        suspect_map = payload.get("suspect_evidence_map")
        if not isinstance(suspect_map, list) or not suspect_map:
            raise ValueError("Missing suspect_evidence_map list.")

        valid_evidence_ids = {int(item["index"]) for item in evidence_bank if "index" in item}
        expected_names = {
            str(name).strip().lower()
            for name in (all_suspects or [])
            if str(name).strip()
        }
        seen_names: set[str] = set()
        required_id_fields = ("incriminating_evidence_ids", "exculpatory_evidence_ids")
        optional_id_fields = ("ambiguous_evidence_ids",)

        for entry in suspect_map:
            if not isinstance(entry, dict):
                raise ValueError("suspect_evidence_map entries must be objects.")
            name = str(entry.get("suspect_name") or "").strip()
            if not name:
                raise ValueError("suspect_evidence_map entry missing suspect_name.")
            seen_names.add(name.lower())
            for field in (*required_id_fields, *optional_id_fields):
                values = entry.get(field, [])
                if values is None and field in optional_id_fields:
                    continue
                if not isinstance(values, list):
                    raise ValueError(f"{name}.{field} must be a list.")
                unknown_ids = [
                    evidence_id
                    for evidence_id in values
                    if not isinstance(evidence_id, int) or evidence_id not in valid_evidence_ids
                ]
                if unknown_ids:
                    raise ValueError(
                        f"{name}.{field} contains unknown evidence id(s): {unknown_ids}"
                    )

        if correct_answer and str(correct_answer).strip().lower() not in seen_names:
            raise ValueError("suspect_evidence_map must include the canonical culprit.")
        missing_names = sorted(expected_names - seen_names)
        if missing_names:
            raise ValueError(
                "suspect_evidence_map missing answer-option suspect(s): "
                + ", ".join(missing_names)
            )

    def _validate_reason_units(
        self,
        payload: dict,
        *,
        evidence_bank: list[dict],
    ) -> None:
        reason_units = payload.get("reason_units")
        if not isinstance(reason_units, list) or len(reason_units) != 4:
            raise ValueError("reason_units must contain exactly four required reason unit objects.")

        valid_evidence_ids = {int(item["index"]) for item in evidence_bank if "index" in item}
        seen_roles: set[str] = set()
        for unit in reason_units:
            if not isinstance(unit, dict):
                raise ValueError("reason_units entries must be objects.")
            role = str(unit.get("role") or "").strip()
            if role not in _REQUIRED_REASON_UNIT_ROLES:
                raise ValueError(f"reason_units contains invalid role: {role!r}.")
            if role in seen_roles:
                raise ValueError(f"reason_units contains duplicate role: {role}.")
            seen_roles.add(role)
            suspect_name = str(unit.get("suspect_name") or "").strip()
            if not suspect_name:
                raise ValueError(f"reason unit {role} missing suspect_name.")
            evidence_ids = unit.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"reason unit {role} evidence_ids must be a non-empty list.")
            unknown_ids = [
                evidence_id
                for evidence_id in evidence_ids
                if not isinstance(evidence_id, int) or evidence_id not in valid_evidence_ids
            ]
            if unknown_ids:
                raise ValueError(
                    f"reason unit {role} evidence_ids contains unknown id(s): {unknown_ids}"
                )
            fact_units = unit.get("fact_units")
            if not isinstance(fact_units, list) or not fact_units or not all(
                isinstance(item, str) and item.strip() for item in fact_units
            ):
                raise ValueError(f"reason unit {role} fact_units must be a non-empty list of strings.")
            unit["fact_units"] = observable_fact_units(fact_units)
            reasoning = unit.get("neutral_reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError(f"reason unit {role} missing neutral_reasoning.")
            intended_turn_ids = unit.get("intended_turn_ids")
            if not isinstance(intended_turn_ids, list) or not all(
                isinstance(turn_id, int) for turn_id in intended_turn_ids
            ):
                raise ValueError(f"reason unit {role} intended_turn_ids must be a list of integers.")

        missing_roles = sorted(_REQUIRED_REASON_UNIT_ROLES - seen_roles)
        if missing_roles:
            raise ValueError("reason_units missing required role(s): " + ", ".join(missing_roles))

    def _validate_opening_factual_reconstruction_schema(
        self,
        payload: dict,
        *,
        evidence_bank: list[dict],
    ) -> None:
        valid_evidence_ids = {int(item["index"]) for item in evidence_bank if "index" in item}
        turns = payload.get("turns")
        if not isinstance(turns, list):
            raise ValueError("Missing turns list.")

        by_id = {
            turn.get("turn_id"): turn
            for turn in turns
            if isinstance(turn, dict)
        }
        for turn_id, expected_speaker in ((1, "A"), (2, "B")):
            turn = by_id.get(turn_id)
            if not isinstance(turn, dict):
                raise ValueError(f"turn {turn_id} opening plan is missing.")
            if turn.get("speaker") != expected_speaker:
                raise ValueError(f"turn {turn_id} speaker must be {expected_speaker!r}.")
            turn_type = turn.get("turn_type")
            if turn_type not in (None, "opening"):
                raise ValueError(f"turn {turn_id} turn_type must be 'opening' or omitted.")

            for field in _OPENING_DISABLED_FIELDS:
                if turn.get(field) is not None:
                    raise ValueError(
                        f"turn {turn_id} opening field {field} must be null; "
                        "openings use only evidence_ids and fact_units."
                    )
            if turn.get("opponent_point_to_attack") is not None:
                raise ValueError(f"turn {turn_id} opponent_point_to_attack must be null.")
            if turn.get("attack_type") is not None:
                raise ValueError(f"turn {turn_id} attack_type must be null.")

            evidence_ids = turn.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"turn {turn_id} evidence_ids must be a non-empty list.")
            unknown_ids = [
                evidence_id
                for evidence_id in evidence_ids
                if not isinstance(evidence_id, int) or evidence_id not in valid_evidence_ids
            ]
            if unknown_ids:
                raise ValueError(
                    f"turn {turn_id} evidence_ids contains unknown id(s): {unknown_ids}"
                )

            fact_units = turn.get("fact_units")
            if (
                not isinstance(fact_units, list)
                or not 4 <= len(fact_units) <= 6
                or not all(isinstance(item, str) and item.strip() for item in fact_units)
            ):
                raise ValueError(f"turn {turn_id} fact_units must contain 4-6 non-empty strings.")

            turn["fact_units"] = observable_fact_units(fact_units)

    def _validate_post_opening_semantic_contract_schema(
        self,
        payload: dict,
        *,
        evidence_bank: list[dict],
    ) -> None:
        valid_evidence_ids = {int(item["index"]) for item in evidence_bank if "index" in item}
        turns = payload.get("turns")
        if not isinstance(turns, list):
            raise ValueError("Missing turns list.")

        turns_by_id: dict[int, dict] = {}
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError("turn entries must be objects.")
            turn_id = turn.get("turn_id")
            if isinstance(turn_id, int):
                turns_by_id[turn_id] = turn
            if turn_id not in _POST_OPENING_TURN_TYPES:
                continue

            missing_fields = [field for field in _SEMANTIC_CONTRACT_FIELDS if field not in turn]
            if missing_fields:
                raise ValueError(
                    f"turn {turn_id} missing semantic contract field(s): "
                    + ", ".join(missing_fields)
                )

            expected_turn_type = _POST_OPENING_TURN_TYPES[turn_id]
            if turn.get("turn_type") != expected_turn_type:
                raise ValueError(
                    f"turn {turn_id} turn_type must be {expected_turn_type!r}."
                )

            evidence_ids = turn.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"turn {turn_id} evidence_ids must be a non-empty list.")
            unknown_ids = [
                evidence_id
                for evidence_id in evidence_ids
                if not isinstance(evidence_id, int) or evidence_id not in valid_evidence_ids
            ]
            if unknown_ids:
                raise ValueError(
                    f"turn {turn_id} evidence_ids contains unknown id(s): {unknown_ids}"
                )

            for field in ("fact_units", "interpretation_units"):
                values = turn.get(field)
                if not isinstance(values, list) or not values or not all(isinstance(item, str) and item.strip() for item in values):
                    raise ValueError(f"turn {turn_id} {field} must be a non-empty list of strings.")

            for field in ("opponent_claim_target", "core_conclusion"):
                value = turn.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"turn {turn_id} {field} must be a non-empty string.")

            concession = turn.get("required_concession")
            if concession is not None and (not isinstance(concession, str) or not concession.strip()):
                raise ValueError(f"turn {turn_id} required_concession must be null or a non-empty string.")

            certainty_level = str(turn.get("certainty_level") or "").strip().lower()
            if certainty_level not in _CERTAINTY_LEVELS:
                raise ValueError(
                    f"turn {turn_id} certainty_level must be one of {sorted(_CERTAINTY_LEVELS)}."
                )

            inference_edges = turn.get("inference_edges")
            if not isinstance(inference_edges, list) or not inference_edges:
                raise ValueError(f"turn {turn_id} inference_edges must be a non-empty list.")
            if turn_id in _COMPACT_REBUTTAL_TURN_IDS | _COMPACT_SUMMARY_TURN_IDS:
                if len(evidence_ids) > 3:
                    raise ValueError(f"turn {turn_id} evidence_ids must contain at most 3 ids.")
                if len(turn.get("fact_units") or []) > 3:
                    raise ValueError(f"turn {turn_id} fact_units must contain at most 3 facts.")
                if len(turn.get("interpretation_units") or []) > 2:
                    raise ValueError(f"turn {turn_id} interpretation_units must contain at most 2 entries.")
                if len(inference_edges) > 2:
                    raise ValueError(f"turn {turn_id} inference_edges must contain at most 2 entries.")
            for edge in inference_edges:
                if not isinstance(edge, dict):
                    raise ValueError(f"turn {turn_id} inference_edges entries must be objects.")
                for field in ("source_fact_units", "interpretation_unit", "supports_conclusion"):
                    if field not in edge:
                        raise ValueError(f"turn {turn_id} inference edge missing {field}.")
                source_units = edge.get("source_fact_units")
                if not isinstance(source_units, list) or not source_units or not all(isinstance(item, str) and item.strip() for item in source_units):
                    raise ValueError(f"turn {turn_id} inference edge source_fact_units must be a non-empty list of strings.")
                for field in ("interpretation_unit", "supports_conclusion"):
                    value = edge.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"turn {turn_id} inference edge {field} must be a non-empty string.")

    def generate_content_attention_question(
        self,
        content_plan: ContentPlan,
        evidence_bank: list[dict],
    ) -> ContentAttentionQuestion:
        """Generate and validate one factual attention check for a content plan."""
        last_error = None
        validation_feedback = None
        content_plan_payload = content_plan.model_dump(exclude={"content_attention_question"})

        for attempt in range(1, 4):
            prompt = build_content_attention_question_prompt(
                content_plan=content_plan_payload,
                evidence_bank=evidence_bank,
                validation_feedback=validation_feedback,
            )
            payload = self.client.complete_json(
                model=MODEL_CONFIGS["planner"]["model"],
                prompt=prompt,
                temperature=MODEL_CONFIGS["planner"]["temperature"],
                max_tokens=_ATTENTION_QUESTION_MAX_TOKENS,
            )
            try:
                return self._validate_content_attention_question(payload, evidence_bank)
            except ValueError as exc:
                last_error = exc
                validation_feedback = str(exc)
                print(f"[planner] attempt {attempt}/3 invalid content attention question: {exc}")

        raise last_error

    @staticmethod
    def _ensure_evidence_utility_classification(
        payload: dict,
        *,
        correct_answer: str | None,
    ) -> None:
        if payload.get("evidence_utility_classification"):
            return
        suspect_map = payload.get("suspect_evidence_map")
        correct_name = str(correct_answer or "").strip().lower()
        if not isinstance(suspect_map, list) or not correct_name:
            return

        utility_by_id: dict[int, dict] = {}

        def add(evidence_id: int, utility: str, rationale: str) -> None:
            existing = utility_by_id.get(evidence_id)
            if existing is None:
                utility_by_id[evidence_id] = {
                    "evidence_id": evidence_id,
                    "utility_for_motion": utility,
                    "side_allowed": (
                        ["ground_truth_supporting"]
                        if utility == "supports_motion"
                        else ["non_ground_truth"]
                        if utility == "opposes_motion"
                        else ["both_neutral"]
                    ),
                    "rationale_from_outcome": rationale,
                }
                return
            if existing["utility_for_motion"] != utility:
                existing["utility_for_motion"] = "double_edged"
                existing["side_allowed"] = ["ground_truth_supporting", "non_ground_truth"]
                existing["rationale_from_outcome"] = (
                    existing["rationale_from_outcome"] + "; also " + rationale
                )

        for entry in suspect_map:
            if not isinstance(entry, dict):
                continue
            suspect_name = str(entry.get("suspect_name") or "").strip()
            is_culprit = suspect_name.lower() == correct_name
            for evidence_id in entry.get("incriminating_evidence_ids") or []:
                if not isinstance(evidence_id, int):
                    continue
                add(
                    evidence_id,
                    "supports_motion" if is_culprit else "opposes_motion",
                    (
                        f"incriminates canonical culprit {suspect_name}"
                        if is_culprit
                        else f"incriminates alternative suspect {suspect_name}"
                    ),
                )
            for evidence_id in entry.get("exculpatory_evidence_ids") or []:
                if not isinstance(evidence_id, int):
                    continue
                add(
                    evidence_id,
                    "opposes_motion" if is_culprit else "supports_motion",
                    (
                        f"exculpates canonical culprit {suspect_name}"
                        if is_culprit
                        else f"exculpates alternative suspect {suspect_name}"
                    ),
                )
            for evidence_id in entry.get("ambiguous_evidence_ids") or []:
                if not isinstance(evidence_id, int):
                    continue
                add(evidence_id, "double_edged", f"ambiguous clue for {suspect_name}")

        if utility_by_id:
            payload["evidence_utility_classification"] = [
                utility_by_id[evidence_id]
                for evidence_id in sorted(utility_by_id)
            ]

    def generate_evidence_content_plan(self, topic_cfg: dict, turns_per_agent: int) -> ContentPlan:
        """Generate a content plan for an evidence-grounded detective debate.

        topic_cfg must contain an 'evidence_bank' key (list of {"index", "text"} dicts).
        """
        raise ValueError(
            "The legacy detective evidence content planner is disabled. "
            "Use build_pairwise_content_plan() with static reasoning_finding from the converted case JSON."
        )

    @staticmethod
    def _reasoning_suspect_map(reasoning_finding: dict) -> dict[str, dict]:
        suspects = reasoning_finding.get("suspects")
        if not isinstance(suspects, list):
            raise ValueError("reasoning finding must contain a suspects list.")
        out: dict[str, dict] = {}
        for entry in suspects:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("suspect_name") or "").strip()
            if name:
                out[name] = entry
        return out

    @staticmethod
    def _canonical_reason_statement(reason: dict) -> str:
        statements = reason.get("reason_statements")
        if isinstance(statements, list) and statements:
            first = statements[0]
            if isinstance(first, dict):
                return str(first.get("statement") or "")
        return str(reason.get("anchor_reason") or reason.get("reasoning") or "")

    @staticmethod
    def _parse_interpretation_set_id(
        *,
        reasoning_finding: dict,
        content_plan_family: str,
        interpretation_set_id: str | None,
    ) -> tuple[str, int, int | None]:
        if content_plan_family not in CONTENT_PLAN_FAMILIES:
            raise ValueError(
                f"content_plan_family must be one of {sorted(CONTENT_PLAN_FAMILIES)}; "
                f"got {content_plan_family!r}."
            )
        primary_count, additional_count = interpretation_dimensions(reasoning_finding)
        if interpretation_set_id is None:
            interpretation_set_id = "primary-1" if content_plan_family == "standard" else "primary-1__additional-1"
        primary_match = re.fullmatch(r"primary-(\d+)", interpretation_set_id)
        additional_match = re.fullmatch(r"additional-(\d+)", interpretation_set_id)
        verbosity_match = re.fullmatch(r"primary-(\d+)__additional-(\d+)", interpretation_set_id)
        if content_plan_family == "standard" and primary_match:
            primary_index = int(primary_match.group(1))
            if not 1 <= primary_index <= primary_count:
                raise ValueError(f"Unknown standard interpretation_set_id {interpretation_set_id!r}; primary count is {primary_count}.")
            return interpretation_set_id, primary_index, None
        if content_plan_family == "standard" and additional_match:
            additional_index = int(additional_match.group(1))
            if not 1 <= additional_index <= additional_count:
                raise ValueError(f"Unknown standard interpretation_set_id {interpretation_set_id!r}; additional count is {additional_count}.")
            return interpretation_set_id, 0, additional_index
        if content_plan_family == "verbosity" and verbosity_match:
            primary_index = int(verbosity_match.group(1))
            additional_index = int(verbosity_match.group(2))
            if not 1 <= primary_index <= primary_count or not 1 <= additional_index <= additional_count:
                raise ValueError(
                    f"Unknown verbosity interpretation_set_id {interpretation_set_id!r}; "
                    f"primary count is {primary_count}, additional count is {additional_count}."
                )
            return interpretation_set_id, primary_index, additional_index
        raise ValueError(
            f"interpretation_set_id {interpretation_set_id!r} is not valid for {content_plan_family} content plans."
        )

    @staticmethod
    def _selected_interpretation(reason: dict, interpretation_index: int) -> dict:
        statements = reason.get("reason_statements")
        if not isinstance(statements, list) or not statements:
            raise ValueError(f"reason {reason.get('reason_id')!r} missing reason_statements list.")
        if not 1 <= interpretation_index <= len(statements):
            raise ValueError(
                f"reason {reason.get('reason_id')!r} has {len(statements)} interpretation(s), "
                f"cannot select index {interpretation_index}."
            )
        entry = statements[interpretation_index - 1]
        statement = normalize_anchor_reason(entry.get("statement"))
        if not statement:
            raise ValueError(f"reason {reason.get('reason_id')!r} interpretation {interpretation_index} has empty statement.")
        return {
            "reason_id": str(reason["reason_id"]),
            "interpretation_index": interpretation_index,
            "interpretation_id": str(entry.get("interpretation_id") or "").strip(),
            "statement": statement,
        }

    @staticmethod
    def _all_reason_statement_text(reason: dict) -> str:
        statements = reason.get("reason_statements")
        if not isinstance(statements, list):
            return Planner._canonical_reason_statement(reason)
        return " ".join(
            str(entry.get("statement") or "")
            for entry in statements
            if isinstance(entry, dict)
        )

    @staticmethod
    def _copy_reason(
        entry: dict,
        direction: str,
        suspect_name: str,
        *,
        source_reason: str,
        interpretation_index: int,
        additional_interpretation_index: int | None = None,
    ) -> tuple[dict, dict]:
        raw = entry.get(direction)
        if not isinstance(raw, dict):
            raise ValueError(f"reasoning finding missing {suspect_name}.{direction}.")
        selected_raw = raw
        if source_reason == "additional":
            selected_raw = raw.get("additional_reason")
            if not isinstance(selected_raw, dict):
                raise ValueError(f"reasoning finding missing {suspect_name}.{direction}.additional_reason.")
        if source_reason not in {"primary", "additional"}:
            raise ValueError(f"Unknown source_reason {source_reason!r}.")
        selected = Planner._selected_interpretation(selected_raw, interpretation_index)
        out = {
            "reason_id": str(selected_raw["reason_id"]),
            "suspect_name": suspect_name,
            "evidence_ids": list(selected_raw["evidence_ids"]),
            "evidence_text": list(selected_raw.get("evidence_text") or []),
            "fact_units": observable_fact_units(list(selected_raw["fact_units"])),
            "direction": direction,
            "source_reason": source_reason,
            "interpretation_index": selected["interpretation_index"],
            "interpretation_id": selected["interpretation_id"],
            "anchor_reason": selected["statement"],
        }
        interpretation_meta = {
            "source_reason": source_reason,
            **selected,
        }
        if additional_interpretation_index is not None:
            additional = raw.get("additional_reason")
            if not isinstance(additional, dict):
                raise ValueError(f"reasoning finding missing {suspect_name}.{direction}.additional_reason.")
            additional_selected = Planner._selected_interpretation(additional, additional_interpretation_index)
            out["additional_reason"] = {
                "reason_id": str(additional["reason_id"]),
                "suspect_name": suspect_name,
                "evidence_ids": list(additional["evidence_ids"]),
                "evidence_text": list(additional.get("evidence_text") or []),
                "fact_units": observable_fact_units(list(additional["fact_units"])),
                "direction": direction,
                "source_reason": "additional",
                "interpretation_index": additional_selected["interpretation_index"],
                "interpretation_id": additional_selected["interpretation_id"],
                "anchor_reason": additional_selected["statement"],
            }
            interpretation_meta = {
                "primary": {
                    "source_reason": "primary",
                    **selected,
                },
                "additional": {
                    "source_reason": "additional",
                    **additional_selected,
                },
            }
        return out, interpretation_meta

    @staticmethod
    def _score_rival_candidate(entry: dict) -> tuple[int, int, int]:
        incr = entry.get("incriminating") or {}
        exculp = entry.get("exculpatory") or {}
        incr_ids = incr.get("evidence_ids") or []
        exculp_ids = exculp.get("evidence_ids") or []
        incr_text = " ".join([
            *(incr.get("fact_units") or []),
            Planner._all_reason_statement_text(incr),
        ]).lower()
        clue_priority = sum(
            1
            for term in (
                "note",
                "handwriting",
                "printed",
                "knight",
                "chess",
                "tree",
                "found",
                "access",
                "could",
            )
            if term in incr_text
        )
        return (clue_priority, len(incr_ids), len(exculp_ids))

    @staticmethod
    def _public_story_text(topic_cfg: dict) -> str:
        context = str(topic_cfg.get("case_context") or "").strip()
        if context:
            return context
        evidence_lines = [
            str(item.get("text") or "").strip()
            for item in topic_cfg.get("evidence_bank") or []
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        return "\n".join(evidence_lines)

    def _generate_claim_realization_bundle(
        self,
        *,
        topic_cfg: dict,
        neutral_wrongdoing_event: str,
        culprit: str,
        rival: str,
    ) -> dict:
        existing = topic_cfg.get("claim_realization_bundle")
        if isinstance(existing, dict):
            return claim_realization_bundle_to_dict(
                validate_claim_realization_payload(
                    existing,
                    culprit_name=culprit,
                    rival_name=rival,
                    neutral_wrongdoing_event=neutral_wrongdoing_event,
                )
            )
        if self.client is None:
            return claim_realization_bundle_to_dict(
                deterministic_claim_realization_bundle(
                    neutral_wrongdoing_event=neutral_wrongdoing_event,
                    culprit_name=culprit,
                    rival_name=rival,
                )
            )

        public_story_text = self._public_story_text(topic_cfg)
        last_error: Exception | None = None
        validation_feedback: str | None = None
        for attempt in range(1, 4):
            prompt = build_claim_realization_prompt(
                public_story_text=public_story_text,
                neutral_wrongdoing_event=neutral_wrongdoing_event,
                culprit_name=culprit,
                rival_name=rival,
                validation_feedback=validation_feedback,
            )
            payload = self.client.complete_json(
                model=MODEL_CONFIGS["planner"]["model"],
                prompt=prompt,
                temperature=MODEL_CONFIGS["planner"]["temperature"],
                max_tokens=900,
            )
            try:
                bundle = validate_claim_realization_payload(
                    payload,
                    culprit_name=culprit,
                    rival_name=rival,
                    neutral_wrongdoing_event=neutral_wrongdoing_event,
                )
                return claim_realization_bundle_to_dict(bundle)
            except ValueError as exc:
                last_error = exc
                validation_feedback = str(exc)
                print(f"[planner] attempt {attempt}/3 invalid claim realization bundle: {exc}")
        raise last_error or ValueError("claim realization generation failed.")

    def build_pairwise_content_plan(
        self,
        *,
        topic_cfg: dict,
        reasoning_finding: dict,
        reasoning_finding_source_path: str | None = None,
        rival_suspect: str | None = None,
        content_plan_family: str = "standard",
        interpretation_set_id: str | None = None,
    ) -> ContentPlan:
        interpretation_set_id, primary_interpretation_index, additional_interpretation_index = self._parse_interpretation_set_id(
            reasoning_finding=reasoning_finding,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
        )
        suspect_map = self._reasoning_suspect_map(reasoning_finding)
        answer_options = [
            str(item.get("name") or "").strip()
            for item in topic_cfg.get("suspect_options") or []
            if str(item.get("name") or "").strip()
        ] or [
            str(name).strip()
            for name in topic_cfg.get("all_suspects") or []
            if str(name).strip()
        ]
        culprit = str(topic_cfg.get("correct_answer") or reasoning_finding.get("correct_answer") or "").strip()
        if culprit not in answer_options:
            raise ValueError(f"correct answer {culprit!r} must be one of the answer options.")
        if culprit not in suspect_map:
            raise ValueError(f"reasoning finding missing culprit suspect {culprit!r}.")

        if rival_suspect:
            rival = str(rival_suspect).strip()
            if rival not in answer_options:
                raise ValueError(f"rival suspect {rival!r} must be one of the answer options.")
            if rival == culprit:
                raise ValueError("rival suspect cannot equal the culprit.")
            if rival not in suspect_map:
                raise ValueError(f"reasoning finding missing rival suspect {rival!r}.")
        else:
            candidates = [
                name
                for name in answer_options
                if name != culprit and name in suspect_map
            ]
            if not candidates:
                raise ValueError("No non-culprit answer-option suspect is available as rival.")
            rival = max(candidates, key=lambda name: self._score_rival_candidate(suspect_map[name]))

        wrongdoing_event = normalize_wrongdoing_event(topic_cfg.get("wrongdoing_event"))
        claim_realization_bundle = self._generate_claim_realization_bundle(
            topic_cfg=topic_cfg,
            neutral_wrongdoing_event=wrongdoing_event,
            culprit=culprit,
            rival=rival,
        )
        neutral_event_description = normalize_wrongdoing_event(
            claim_realization_bundle.get("neutral_event_description") or wrongdoing_event
        )
        question = format_motion_from_claim_bundle(
            claim_realization_bundle,
            wrongdoing_event=wrongdoing_event,
            first_suspect=culprit,
            second_suspect=rival,
        )
        culprit_stance = accusation_clause_for_suspect(claim_realization_bundle, culprit, wrongdoing_event)
        rival_stance = accusation_clause_for_suspect(claim_realization_bundle, rival, wrongdoing_event)
        debate_setup = {
            "wrongdoing_event": wrongdoing_event,
            "neutral_event_description": neutral_event_description,
            "claim_realization_bundle": claim_realization_bundle,
            "culprit_name": culprit,
            "rival_suspect_name": rival,
            "question": question,
            "culprit_stance": culprit_stance,
            "rival_stance": rival_stance,
        }

        culprit_entry = suspect_map[culprit]
        rival_entry = suspect_map[rival]
        if content_plan_family == "standard":
            source_reason = "additional" if primary_interpretation_index == 0 else "primary"
            standard_interpretation_index = (
                additional_interpretation_index
                if source_reason == "additional"
                else primary_interpretation_index
            )
            copied_reasons = {
                "culprit_incriminating": self._copy_reason(
                    culprit_entry,
                    "incriminating",
                    culprit,
                    source_reason=source_reason,
                    interpretation_index=standard_interpretation_index,
                ),
                "culprit_exculpatory": self._copy_reason(
                    culprit_entry,
                    "exculpatory",
                    culprit,
                    source_reason=source_reason,
                    interpretation_index=standard_interpretation_index,
                ),
                "rival_incriminating": self._copy_reason(
                    rival_entry,
                    "incriminating",
                    rival,
                    source_reason=source_reason,
                    interpretation_index=standard_interpretation_index,
                ),
                "rival_exculpatory": self._copy_reason(
                    rival_entry,
                    "exculpatory",
                    rival,
                    source_reason=source_reason,
                    interpretation_index=standard_interpretation_index,
                ),
            }
        else:
            copied_reasons = {
                "culprit_incriminating": self._copy_reason(
                    culprit_entry,
                    "incriminating",
                    culprit,
                    source_reason="primary",
                    interpretation_index=primary_interpretation_index,
                    additional_interpretation_index=additional_interpretation_index,
                ),
                "culprit_exculpatory": self._copy_reason(
                    culprit_entry,
                    "exculpatory",
                    culprit,
                    source_reason="primary",
                    interpretation_index=primary_interpretation_index,
                    additional_interpretation_index=additional_interpretation_index,
                ),
                "rival_incriminating": self._copy_reason(
                    rival_entry,
                    "incriminating",
                    rival,
                    source_reason="primary",
                    interpretation_index=primary_interpretation_index,
                    additional_interpretation_index=additional_interpretation_index,
                ),
                "rival_exculpatory": self._copy_reason(
                    rival_entry,
                    "exculpatory",
                    rival,
                    source_reason="primary",
                    interpretation_index=primary_interpretation_index,
                    additional_interpretation_index=additional_interpretation_index,
                ),
            }
        selected_reasons = {
            key: value[0]
            for key, value in copied_reasons.items()
        }
        selected_interpretations = {
            key: value[1]
            for key, value in copied_reasons.items()
        }
        reason_by_key = selected_reasons
        culprit_opening_ids = list(dict.fromkeys([
            *reason_by_key["culprit_incriminating"]["evidence_ids"],
            *reason_by_key["rival_exculpatory"]["evidence_ids"],
        ]))
        rival_opening_ids = list(dict.fromkeys([
            *reason_by_key["rival_incriminating"]["evidence_ids"],
            *reason_by_key["culprit_exculpatory"]["evidence_ids"],
        ]))

        role_turn_templates = {
            "culprit_opening": {
                "speaker_role": "culprit",
                "turn_type": "opening",
                "evidence_ids": culprit_opening_ids,
                "fact_units": [
                    *reason_by_key["culprit_incriminating"]["fact_units"],
                    *reason_by_key["rival_exculpatory"]["fact_units"],
                ],
            },
            "rival_opening": {
                "speaker_role": "rival",
                "turn_type": "opening",
                "evidence_ids": rival_opening_ids,
                "fact_units": [
                    *reason_by_key["rival_incriminating"]["fact_units"],
                    *reason_by_key["culprit_exculpatory"]["fact_units"],
                ],
            },
            "culprit_argument": {
                "speaker_role": "culprit",
                "turn_type": "argument",
                "subject_role": "culprit",
                "subject_suspect": culprit,
                "reason_refs": [reason_by_key["culprit_incriminating"]["reason_id"]],
                "sentence_jobs": [
                    {
                        "sentence_index": 1,
                        "job": "state_guilt_claim_with_incriminating_reason",
                        "reason_id": reason_by_key["culprit_incriminating"]["reason_id"],
                    },
                    {
                        "sentence_index": 2,
                        "job": "fill_reasoning_slot",
                        "slot_id": "culprit_support_reasoning",
                        "goal": "Explain why the fixed incriminating reason supports guilt.",
                    },
                ],
                "evidence_ids": reason_by_key["culprit_incriminating"]["evidence_ids"],
                "fact_units": reason_by_key["culprit_incriminating"]["fact_units"],
                "accusation_clause": accusation_clause_for_suspect(claim_realization_bundle, culprit, wrongdoing_event),
                "core_conclusion": guilt_conclusion_for_suspect(claim_realization_bundle, culprit, wrongdoing_event),
            },
            "rival_argument": {
                "speaker_role": "rival",
                "turn_type": "argument",
                "subject_role": "rival",
                "subject_suspect": rival,
                "reason_refs": [reason_by_key["rival_incriminating"]["reason_id"]],
                "sentence_jobs": [
                    {
                        "sentence_index": 1,
                        "job": "state_guilt_claim_with_incriminating_reason",
                        "reason_id": reason_by_key["rival_incriminating"]["reason_id"],
                    },
                    {
                        "sentence_index": 2,
                        "job": "fill_reasoning_slot",
                        "slot_id": "rival_support_reasoning",
                        "goal": "Explain why the fixed incriminating reason supports guilt.",
                    },
                ],
                "evidence_ids": reason_by_key["rival_incriminating"]["evidence_ids"],
                "fact_units": reason_by_key["rival_incriminating"]["fact_units"],
                "accusation_clause": accusation_clause_for_suspect(claim_realization_bundle, rival, wrongdoing_event),
                "core_conclusion": guilt_conclusion_for_suspect(claim_realization_bundle, rival, wrongdoing_event),
            },
            "culprit_rebuttal": {
                "speaker_role": "culprit",
                "turn_type": "rebuttal",
                "subject_role": "rival",
                "subject_suspect": rival,
                "reason_refs": [
                    reason_by_key["rival_incriminating"]["reason_id"],
                    reason_by_key["rival_exculpatory"]["reason_id"],
                ],
                "sentence_jobs": [
                    {
                        "sentence_index": 1,
                        "job": "reject_opponent_incriminating_reason",
                        "reason_id": reason_by_key["rival_incriminating"]["reason_id"],
                        "slot_id": "culprit_rebuttal_of_rival_reasoning",
                        "goal": "Explain why the fixed incriminating reason is insufficient to establish guilt.",
                    },
                    {
                        "sentence_index": 2,
                        "job": "state_innocence_with_exculpatory_reason",
                        "reason_id": reason_by_key["rival_exculpatory"]["reason_id"],
                        "slot_id": "culprit_exculpatory_reasoning",
                        "goal": "Explain why the fixed exculpatory reason supports innocence.",
                    },
                ],
                "evidence_ids": list(dict.fromkeys([
                    *reason_by_key["rival_incriminating"]["evidence_ids"],
                    *reason_by_key["rival_exculpatory"]["evidence_ids"],
                ])),
                "fact_units": [
                    *reason_by_key["rival_incriminating"]["fact_units"],
                    *reason_by_key["rival_exculpatory"]["fact_units"],
                ],
                "accusation_clause": accusation_clause_for_suspect(claim_realization_bundle, rival, wrongdoing_event),
                "opponent_claim_target": accusation_clause_for_suspect(claim_realization_bundle, rival, wrongdoing_event),
                "core_conclusion": innocence_conclusion_for_suspect(claim_realization_bundle, rival, wrongdoing_event),
            },
            "rival_rebuttal": {
                "speaker_role": "rival",
                "turn_type": "rebuttal",
                "subject_role": "culprit",
                "subject_suspect": culprit,
                "reason_refs": [
                    reason_by_key["culprit_incriminating"]["reason_id"],
                    reason_by_key["culprit_exculpatory"]["reason_id"],
                ],
                "sentence_jobs": [
                    {
                        "sentence_index": 1,
                        "job": "reject_opponent_incriminating_reason",
                        "reason_id": reason_by_key["culprit_incriminating"]["reason_id"],
                        "slot_id": "rival_rebuttal_of_culprit_reasoning",
                        "goal": "Explain why the fixed incriminating reason is insufficient to establish guilt.",
                    },
                    {
                        "sentence_index": 2,
                        "job": "state_innocence_with_exculpatory_reason",
                        "reason_id": reason_by_key["culprit_exculpatory"]["reason_id"],
                        "slot_id": "rival_exculpatory_reasoning",
                        "goal": "Explain why the fixed exculpatory reason supports innocence.",
                    },
                ],
                "evidence_ids": list(dict.fromkeys([
                    *reason_by_key["culprit_incriminating"]["evidence_ids"],
                    *reason_by_key["culprit_exculpatory"]["evidence_ids"],
                ])),
                "fact_units": [
                    *reason_by_key["culprit_incriminating"]["fact_units"],
                    *reason_by_key["culprit_exculpatory"]["fact_units"],
                ],
                "accusation_clause": accusation_clause_for_suspect(claim_realization_bundle, culprit, wrongdoing_event),
                "opponent_claim_target": accusation_clause_for_suspect(claim_realization_bundle, culprit, wrongdoing_event),
                "core_conclusion": innocence_conclusion_for_suspect(claim_realization_bundle, culprit, wrongdoing_event),
            },
        }
        turn_dependencies = {
            "culprit_opening": [],
            "rival_opening": [],
            "culprit_argument": ["culprit_opening"],
            "rival_argument": ["rival_opening"],
            "culprit_rebuttal": ["rival_argument"],
            "rival_rebuttal": ["culprit_argument"],
        }

        return ContentPlan(
            content_plan_version=PAIRWISE_CONTENT_PLAN_VERSION,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
            selected_interpretations=selected_interpretations,
            topic=question,
            debate_question=question,
            debate_setup=debate_setup,
            claim_realization_bundle=claim_realization_bundle,
            reasoning_finding_source_path=reasoning_finding_source_path,
            selected_reasons=selected_reasons,
            private_truth_used=True,
            role_turn_templates=role_turn_templates,
            turn_dependencies=turn_dependencies,
            content_attention_question=build_default_content_attention_question(
                wrongdoing_event,
                topic_cfg.get("evidence_bank") or [],
            ),
            turns=[],
        )

    def generate_style_plan(self, agent_name: str, trait_name: str, variant_name: str, trait_rules: list[str], eval_feedback: str | None = None) -> TransformationPlan:
        prompt = build_style_prompt(agent_name, trait_name, variant_name, trait_rules, eval_feedback=eval_feedback)
        payload = self.client.complete_json(
            model=MODEL_CONFIGS["planner"]["model"],
            prompt=prompt,
            temperature=MODEL_CONFIGS["planner"]["temperature"],
            max_tokens=700,
        )
        return TransformationPlan(**payload)
