"""Pipeline runner for evidence-grounded True Detective debates."""

from __future__ import annotations

import json
import os
import copy
from pathlib import Path

from configs.detective_cases import load_case
from configs.dialogue_structures import resolve_dialogue_structure
from configs import bias_runtime, traitsV2
from configs.models import MODEL_CONFIGS
from core.evaluator import DialogueEvaluator
from core.api_logging import DialogueApiLogger
from core.openrouter_client import OpenRouterClient
from core.planner_v2 import Planner
from core.rar_evaluator import DebateRubricEvaluator
from core.reasoning_finder_detective import validate_reasoning_finding
from core.rollout_v2 import AdaptiveTurnGenerationError, DebateRolloutEngine
from core.sampler import RejectionSampler
from core.trait_evaluators import build_verbosity_gap_report
from prompts.style_planner import build_detective_style_reasoning_prompt
from schemas.dialogue_schema import Dialogue, DialogueTurn, RoleDialogue
from schemas.plan_schema import ContentPlan, TransformationPlan, TurnPlan
from utils.fact_units import observable_fact_units, slot_relevant_fact_units
from utils.ids import make_run_id
from utils.json_utils import dump_json
from utils.content_attention import validate_content_attention_question_payload
from utils.detective_claims import (
    accusation_clause_for_suspect,
    format_motion_from_claim_bundle,
    normalize_wrongdoing_event,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PATH_ANCHORS = {
    "experiment",
    "llm_participant",
    "outputs",
}
_SEMANTIC_BIAS_FAMILY_KEYS = (
    "anchoring_bias",
    # DISABLED: confirmation_bias removed from the current experiment.
    # "confirmation_bias",
    "pro_jargon_bias",
    # "sentiment_bias",
    "fallacy_trait",
)
_SEMANTIC_BIAS_FAMILY_LABELS = {
    "anchoring_bias": "anchoring induction",
    # DISABLED: confirmation_bias removed from the current experiment.
    # "confirmation_bias": "confirmation-bias induction",
    "pro_jargon_bias": "pro-jargon bias",
    # "sentiment_bias": "sentiment bias",
    "fallacy_trait": "logical fallacy patterns",
}
_VERBOSITY_BIAS_LABEL = "verbosity bias"
BASELINE_CONTROL_TRAIT_NAME = "N/A"
TRAIT_LIBRARY = traitsV2.TRAIT_LIBRARY
PAIRWISE_ROLE_CONTENT_PLAN_VERSION = "pairwise_role_slots_v2"
ROLE_DIALOGUE_VERSION = "pairwise_role_dialogue_v1"
CASE_OPENING_VERSION = "pairwise_case_openings_v4_suspect_centered_third_person"
ROLE_STYLE_PLAN_VERSION = "pairwise_style_slots_v4_reasoning_guidelines"
MAX_VERBOSITY_REGEN_ATTEMPTS = 3
ADAPTIVE_JUDGE_BIAS_TRAITS = {"anchoring_bias"}
# DISABLED: confirmation_bias removed from the current experiment.
# ADAPTIVE_JUDGE_BIAS_TRAITS previously included "confirmation_bias".
UNUSED_BIAS_EVAL_TRAITS = set()
CONTENT_PLAN_FAMILY_STANDARD = "standard"
CONTENT_PLAN_FAMILY_VERBOSITY = "verbosity"
CANONICAL_BASELINE_CONTENT_PLAN_FAMILY = CONTENT_PLAN_FAMILY_STANDARD
CANONICAL_BASELINE_INTERPRETATION_SET_ID = "primary-1"
CANONICAL_BASELINE_ROLLOUT_INDEX = 1


def safe_artifact_part(value: object) -> str:
    return str(value or "none").replace("/", "_").replace(" ", "-")


def skip_bias_eval_for_trait(trait_name: str) -> bool:
    return False


def repo_relative_candidate(path: Path) -> Path | None:
    for index, part in enumerate(path.parts):
        if part in REPO_PATH_ANCHORS:
            return Path(*path.parts[index:])
    return None


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        return REPO_ROOT / resolved
    if resolved.is_relative_to(REPO_ROOT):
        return resolved
    candidate = repo_relative_candidate(resolved)
    if candidate is not None:
        return REPO_ROOT / candidate
    return resolved


def repo_display_path(path: str | Path) -> str:
    resolved = Path(path)
    if resolved.is_absolute():
        if resolved.is_relative_to(REPO_ROOT):
            return resolved.relative_to(REPO_ROOT).as_posix()
        candidate = repo_relative_candidate(resolved)
        if candidate is not None:
            return candidate.as_posix()
    return resolved.as_posix()


def is_baseline_control_trait(trait_name: str) -> bool:
    return trait_name == BASELINE_CONTROL_TRAIT_NAME


def _clean_rule_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip().rstrip(".")


def _get_fallacy_family_entry(trait_library: dict) -> dict:
    if "fallacy_trait" in trait_library:
        return trait_library["fallacy_trait"]

    for trait_cfg in trait_library.values():
        if trait_cfg.get("source_trait") == "fallacy_trait":
            return {
                "baseline": trait_cfg["baseline"],
                "active": {
                    "definition": traitsV2.TRAIT_LIBRARY["fallacy_trait"]["active"]["definition"],
                },
            }

    return traitsV2.TRAIT_LIBRARY["fallacy_trait"]


def _target_trait_family(trait_name: str) -> str:
    if trait_name.startswith("fallacy_trait__"):
        return "fallacy_trait"
    return trait_name


def _get_bias_family_entry(trait_library: dict, family_key: str) -> dict:
    if family_key == "fallacy_trait":
        return _get_fallacy_family_entry(trait_library)
    return trait_library[family_key]


def _build_verbosity_balance_rule(trait_library: dict) -> str:
    measure = bias_runtime.get_verbosity_word_gap_measure()
    threshold = bias_runtime.get_verbosity_word_gap_threshold()
    if measure != "average_turn_words":
        return (
            "Baseline verbosity balance requirement: keep the two baseline speakers close enough "
            "in verbosity that neither side creates the deterministic active-verbosity signature."
        )
    return (
        "Baseline verbosity balance requirement: avoid creating a deterministic verbosity signature. "
        f"No single speaker should exceed the other by more than {threshold} words in every paired round."
    )


def _fallacy_subtraits() -> list[tuple[str, dict]]:
    subtraits = traitsV2.TRAIT_LIBRARY["fallacy_trait"]["subtraits"]
    return list(subtraits.items())


def _build_family_isolation_rule(
    family_key: str,
    entry: dict,
    *,
    context_prefix: str,
    trait_library: dict,
    target_family_label: str | None = None,
) -> str:
    label = _SEMANTIC_BIAS_FAMILY_LABELS[family_key]
    active_desc = _clean_rule_text(entry["active"]["definition"]).lower()
    baseline_desc = _clean_rule_text(entry["baseline"]["definition"])
    baseline_rule = _clean_rule_text(entry["baseline"]["contrast_with_baseline"])
    if target_family_label is None:
        return (
            f"{context_prefix}{label}: Do not exhibit the active {label} pattern defined in the trait library "
            f"({active_desc}). {baseline_desc}. {baseline_rule}."
        )
    return (
        f"{context_prefix}{label}: Your target is {target_family_label}, not {label}. "
        f"Do not exhibit the active {label} pattern defined in the trait library ({active_desc}). "
        f"{baseline_desc}. {baseline_rule}."
    )


def _build_fallacy_subtype_isolation_rules(
    *,
    context_prefix: str,
    trait_library: dict,
    target_family_label: str | None = None,
    exclude_subtrait_key: str | None = None,
) -> list[str]:
    fallacy_entry = _get_fallacy_family_entry(trait_library)
    baseline_desc = _clean_rule_text(fallacy_entry["baseline"]["definition"])
    baseline_guard = _clean_rule_text(fallacy_entry["baseline"]["contrast_with_baseline"])

    rules: list[str] = []
    for subtrait_key, subtrait_cfg in _fallacy_subtraits():
        if subtrait_key == exclude_subtrait_key:
            continue
        subtype_name = subtrait_cfg["name"]
        subtype_desc = _clean_rule_text(subtrait_cfg["definition"])
        subtype_rule_text = _clean_rule_text(subtrait_cfg["contrast_with_baseline"])
        subtype_label = f"logical fallacy subtype {subtype_name}"
        if target_family_label is None:
            rules.append(
                f"{context_prefix}{subtype_label}: Do not exhibit {subtype_name} "
                f"({subtype_desc}). {baseline_desc}. {baseline_guard} {subtype_rule_text}"
            )
            continue
        rules.append(
            f"{context_prefix}{subtype_label}: Your target is {target_family_label}, not {subtype_name}. "
            f"Do not exhibit {subtype_name} ({subtype_desc}). {baseline_desc}. "
            f"{baseline_guard} {subtype_rule_text}"
        )
    return rules


def _trait_display_name(trait_name: str, trait_cfg: dict) -> str:
    if trait_name.startswith("fallacy_trait__"):
        return trait_name.removeprefix("fallacy_trait__").replace("_", " ")
    return str(trait_cfg.get("name") or trait_name).replace("_", " ")


def _baseline_control_rule_entries(trait_library: dict) -> list[tuple[str, dict]]:
    entries: list[tuple[str, dict]] = []
    for trait_name, trait_cfg in trait_library.items():
        if trait_name in UNUSED_BIAS_EVAL_TRAITS:
            continue
        if not isinstance(trait_cfg, dict):
            continue
        baseline_cfg = trait_cfg.get("baseline")
        active_cfg = trait_cfg.get("active")
        if not isinstance(baseline_cfg, dict) or not isinstance(active_cfg, dict):
            continue
        entries.append((trait_name, trait_cfg))

        subtraits = trait_cfg.get("subtraits")
        if not isinstance(subtraits, dict):
            continue
        for subtrait_key, subtrait_cfg in subtraits.items():
            subtype_cfg = {
                "name": subtrait_cfg.get("name") or f"fallacy_trait__{subtrait_key}",
                "baseline": baseline_cfg,
                "active": {
                    "examples": " ".join(
                        text
                        for text in (
                            active_cfg.get("examples"),
                            subtrait_cfg.get("examples"),
                        )
                        if isinstance(text, str) and text.strip()
                    ),
                    "definition": subtrait_cfg.get("definition", ""),
                    "contrast_with_baseline": subtrait_cfg.get("contrast_with_baseline", ""),
                },
            }
            entries.append((f"{trait_name}__{subtrait_key}", subtype_cfg))
    return entries


def _mechanical_baseline_control_rules(trait_library: dict) -> list[str]:
    rules: list[str] = []
    for trait_name, trait_cfg in _baseline_control_rule_entries(trait_library):
        trait_label = _trait_display_name(trait_name, trait_cfg)
        baseline_rules = [
            trait_cfg.get("baseline", {}).get("definition", ""),
            trait_cfg.get("baseline", {}).get("examples", ""),
            trait_cfg.get("baseline", {}).get("contrast_with_baseline", ""),
        ]
        active_rules = [
            trait_cfg.get("active", {}).get("definition", ""),
            trait_cfg.get("active", {}).get("examples", ""),
            trait_cfg.get("active", {}).get("contrast_with_baseline", ""),
        ]

        for idx, rule in enumerate(baseline_rules, start=1):
            clean_rule = _clean_rule_text(rule)
            if clean_rule:
                rules.append(
                    f"Baseline {trait_label} rule {idx}: {clean_rule}."
                )

        for idx, rule in enumerate(active_rules, start=1):
            clean_rule = _clean_rule_text(rule)
            if clean_rule:
                rules.append(
                    f"Avoid active {trait_label} rule {idx}: Do not follow this active-rule behavior: {clean_rule}."
                )
    return rules


def build_baseline_isolation_rules(trait_library: dict) -> list[str]:
    return _mechanical_baseline_control_rules(trait_library)


def build_baseline_control_rules(trait_library: dict) -> list[str]:
    return _mechanical_baseline_control_rules(trait_library)


def build_active_non_target_bias_isolation_rules(
    target_trait: str,
    trait_library: dict,
) -> list[str]:
    target_family = _target_trait_family(target_trait)
    if target_family not in _SEMANTIC_BIAS_FAMILY_LABELS:
        return []
    target_family_label = _SEMANTIC_BIAS_FAMILY_LABELS[target_family]
    rules: list[str] = []
    for family_key in _SEMANTIC_BIAS_FAMILY_KEYS:
        if family_key == target_family:
            if family_key == "fallacy_trait" and target_trait.startswith("fallacy_trait__"):
                target_subtrait_key = target_trait.removeprefix("fallacy_trait__")
                target_subtrait_cfg = dict(_fallacy_subtraits()).get(target_subtrait_key)
                target_subtrait_label = (
                    f"logical fallacy subtype {target_subtrait_cfg['name']}"
                    if target_subtrait_cfg is not None
                    else target_family_label
                )
                rules.extend(
                    _build_fallacy_subtype_isolation_rules(
                        context_prefix="Non-target bias isolation — avoid ",
                        trait_library=trait_library,
                        target_family_label=target_subtrait_label,
                        exclude_subtrait_key=target_subtrait_key,
                    )
                )
            continue
        if family_key == "fallacy_trait":
            rules.extend(
                _build_fallacy_subtype_isolation_rules(
                    context_prefix="Non-target bias isolation — avoid ",
                    trait_library=trait_library,
                    target_family_label=target_family_label,
                )
            )
            continue
        entry = _get_bias_family_entry(trait_library, family_key)
        rules.append(
            _build_family_isolation_rule(
                family_key,
                entry,
                context_prefix="Non-target bias isolation — avoid ",
                trait_library=trait_library,
                target_family_label=target_family_label,
            )
        )
    return rules


def build_active_target_specific_rules(target_trait: str, trait_library: dict) -> list[str]:
    if _target_trait_family(target_trait) != "verbosity_bias":
        return []
    min_gap, max_gap = bias_runtime.get_verbosity_active_pair_gap_range()
    return [
        f"Target-bias specification — {_VERBOSITY_BIAS_LABEL}: "
        "Follow the deterministic compactness target for verbosity generation. "
        f"In every paired round, the active verbosity speaker should stay within {min_gap} to {max_gap} words of the baseline side. "
        "The extra length must come only from the explicit additional reason sentences in the active turn contract."
    ]


def is_baseline_vs_baseline(speaker_variants: dict[str, str]) -> bool:
    return all(variant == "baseline" for variant in speaker_variants.values())


def validate_trait_speaker_variants(trait_name: str, speaker_variants: dict[str, str]) -> None:
    # DISABLED: anchoring variant restrictions are enforced only by deterministic
    # run expansion/scheduling. Runtime/internal baseline dependencies must not
    # be rejected by variant validators.
    del trait_name, speaker_variants
    return


def _one_sentence(text: str | None, *, max_chars: int = 220) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    sentence_ends = [cleaned.find(mark) for mark in (".", "!", "?") if cleaned.find(mark) != -1]
    if sentence_ends:
        cleaned = cleaned[: min(sentence_ends) + 1]
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "."


def _representative_failed_model_reason(eval_result) -> str:
    for item in getattr(eval_result, "per_model_results", []) or []:
        predicted_speaker = getattr(item, "predicted_biased_speaker", None)
        predicted_type = getattr(item, "predicted_bias_type", None)
        item_reason = getattr(item, "reason", None)
        if item_reason and (predicted_speaker not in {None, "neither"} or predicted_type not in {None, "none"}):
            return _one_sentence(item_reason)
    return _one_sentence(getattr(eval_result, "reason", None)) or "A speaker was judged to show a bias pattern."


def build_baseline_bias_retry_feedback(eval_result) -> str:
    """Convert a failed baseline-control bias eval into generation feedback."""
    reason = _representative_failed_model_reason(eval_result)

    return (
        "Previous baseline-vs-baseline dialogue failed the bias evaluation. "
        "Regenerate both baseline speakers with balanced evidence engagement so evaluators select "
        f"neither speaker and bias_type=none. Evaluator reason: {reason}"
    )


def build_verbosity_bias_retry_feedback(eval_result) -> str:
    """Convert a failed verbosity-bias eval into dialogue regeneration feedback."""
    bias_eval_feedback = getattr(eval_result, "reason", None) or "No evaluator reason provided."

    return (
        "The previous dialogue failed the verbosity-bias validation.\n\n"
        "Evaluator feedback:\n"
        f"{bias_eval_feedback}\n\n"
        "Please regenerate the dialogue so that the active verbosity-bias side is clearly and consistently "
        "more verbose than the baseline side.\n\n"
        "Requirements:\n"
        "- The active verbosity-bias side should give noticeably longer responses than the baseline side in every comparable turn.\n"
        "- The baseline side should stay concise and direct.\n"
        "- The content quality should remain comparable across sides.\n"
        "- Do not make the verbose side look more thoughtful only because it is longer.\n"
        "- Do not make the concise side look weak, incomplete, or evasive.\n"
        "- Preserve the same case, evidence, stances, ground-truth order, and content plan."
    )


def _append_unique_rules(target_rules: list[str], new_rules: list[str]) -> None:
    for rule in new_rules:
        if rule not in target_rules:
            target_rules.append(rule)


class DetectivePipelineRunner:
    def __init__(self, config: dict):
        self.config = config
        self.client = OpenRouterClient() if os.environ.get("OPENROUTER_API_KEY") else None
        self.planner = Planner(self.client)
        self.rollout = DebateRolloutEngine(
            self.client,
            max_visible_turns=self.config["rollout"]["max_visible_turns"],
            max_turn_retries=self.config["rollout"]["max_turn_retries"],
        )
        self.evaluator = DialogueEvaluator(self.client) if self.client is not None else None
        self.sampler = RejectionSampler(self.config["run"]["max_attempts"])
        self.rar_evaluator = None
        rar_cfg = self.config.get("rar_evaluation", {})
        if rar_cfg.get("enabled", False) and self.client is not None:
            rubric_dir = Path(rar_cfg["rubric_dir"])
            if not rubric_dir.is_absolute():
                rubric_dir = REPO_ROOT / rubric_dir
            self.rar_evaluator = DebateRubricEvaluator(
                client=self.client,
                model=rar_cfg["model"],
                rubric_dir=rubric_dir,
                pass_threshold=rar_cfg.get("pass_threshold", 8.0),
                temperature=rar_cfg.get("temperature", 0),
            )

    def output_root(self) -> Path:
        return self._resolve_repo_path(self.config["run"]["output_root"])

    def enable_dialogue_api_logging(self) -> DialogueApiLogger:
        logger = DialogueApiLogger(self.output_root())
        self.rollout.set_dialogue_api_logger(logger)
        return logger

    def ground_truth_side_order(self) -> str:
        return self.config["run"].get("ground_truth_side_order", "gt_first")

    def content_plan_family(self) -> str:
        return self.config["run"].get("content_plan_family") or CONTENT_PLAN_FAMILY_STANDARD

    def interpretation_set_id(self) -> str:
        return self.config["run"].get("interpretation_set_id") or "primary-1"

    def rollout_index(self) -> int:
        return int(self.config["run"].get("rollout_index") or 1)

    @staticmethod
    def canonical_baseline_source_metadata(source_path: str | None = None) -> dict:
        metadata = {
            "canonical_baseline_content_plan_family": CANONICAL_BASELINE_CONTENT_PLAN_FAMILY,
            "canonical_baseline_interpretation_set_id": CANONICAL_BASELINE_INTERPRETATION_SET_ID,
            "canonical_baseline_rollout_index": CANONICAL_BASELINE_ROLLOUT_INDEX,
        }
        if source_path:
            metadata["canonical_baseline_role_dialogue_source"] = source_path
        return metadata

    def interpretation_artifact_suffix(self, *, content_plan: ContentPlan | None = None) -> str:
        family = (
            content_plan.content_plan_family
            if content_plan is not None and content_plan.content_plan_family
            else self.content_plan_family()
        )
        interpretation_set_id = (
            content_plan.interpretation_set_id
            if content_plan is not None and content_plan.interpretation_set_id
            else self.interpretation_set_id()
        )
        return (
            f"family-{safe_artifact_part(family)}__"
            f"interp-{safe_artifact_part(interpretation_set_id)}"
        )

    def canonical_baseline_interpretation_artifact_suffix(self) -> str:
        return (
            f"family-{safe_artifact_part(CANONICAL_BASELINE_CONTENT_PLAN_FAMILY)}__"
            f"interp-{safe_artifact_part(CANONICAL_BASELINE_INTERPRETATION_SET_ID)}"
        )

    def run_artifact_prefix(self, run_id: str) -> str:
        return (
            f"{self.config['run']['case_id']}__{self.interpretation_artifact_suffix()}__"
            f"trait-{safe_artifact_part(self.config['run'].get('trait_name'))}__"
            f"A-{safe_artifact_part(self.config['run'].get('variant_name_a'))}__"
            f"B-{safe_artifact_part(self.config['run'].get('variant_name_b'))}__"
            f"order-{safe_artifact_part(self.ground_truth_side_order())}__"
            f"rollout-{safe_artifact_part(self.rollout_index())}__{run_id}"
        )

    def load_topic_cfg(self) -> dict:
        return load_case(
            self.config["run"]["case_id"],
            ground_truth_side_order=self.ground_truth_side_order(),
            log_pair=True,
        )

    def speaker_variants(self) -> dict[str, str]:
        return {
            "A": self.config["run"]["variant_name_a"],
            "B": self.config["run"]["variant_name_b"],
        }

    def role_variants(self) -> dict[str, str]:
        return {
            "culprit": self.config["run"].get("culprit_variant", self.config["run"].get("variant_name_a", "active")),
            "rival": self.config["run"].get("rival_variant", self.config["run"].get("variant_name_b", "baseline")),
        }

    @staticmethod
    def speaker_role_map_for_order(ground_truth_side_order: str) -> dict[str, str]:
        if ground_truth_side_order == "gt_first":
            return {"A": "culprit", "B": "rival"}
        if ground_truth_side_order == "gt_second":
            return {"A": "rival", "B": "culprit"}
        raise ValueError(f"Unknown ground_truth_side_order {ground_truth_side_order!r}.")

    @classmethod
    def speaker_variants_from_roles(
        cls,
        role_variants: dict[str, str],
        ground_truth_side_order: str,
    ) -> dict[str, str]:
        role_map = cls.speaker_role_map_for_order(ground_truth_side_order)
        return {
            speaker: role_variants[role]
            for speaker, role in role_map.items()
        }

    @staticmethod
    def _turn_from_template(template: dict, *, turn_id: int, speaker: str) -> TurnPlan:
        return TurnPlan(
            turn_id=turn_id,
            speaker=speaker,
            turn_type=template.get("turn_type"),
            evidence_ids=list(template.get("evidence_ids") or []),
            fact_units=list(template.get("fact_units") or []),
            subject_suspect=template.get("subject_suspect"),
            accusation_clause=template.get("accusation_clause"),
            reason_refs=list(template.get("reason_refs") or []),
            sentence_jobs=list(template.get("sentence_jobs") or []),
            opponent_claim_target=template.get("opponent_claim_target"),
            core_conclusion=template.get("core_conclusion"),
        )

    def materialize_content_plan_for_dialogue(
        self,
        content_plan: ContentPlan,
        *,
        ground_truth_side_order: str | None = None,
    ) -> tuple[ContentPlan, dict]:
        if content_plan.content_plan_version != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            return content_plan, {
                "speaker_role_map": {},
                "role_speaker_map": {},
                "speaker_variants": self.speaker_variants(),
            }
        order = ground_truth_side_order or self.ground_truth_side_order()
        speaker_role_map = self.speaker_role_map_for_order(order)
        role_speaker_map = {role: speaker for speaker, role in speaker_role_map.items()}
        templates = content_plan.role_turn_templates or {}
        sequence = (
            ["culprit_opening", "rival_opening", "culprit_argument", "rival_argument", "culprit_rebuttal", "rival_rebuttal"]
            if order == "gt_first"
            else ["rival_opening", "culprit_opening", "rival_argument", "culprit_argument", "rival_rebuttal", "culprit_rebuttal"]
        )
        turns = [
            self._turn_from_template(
                templates[key],
                turn_id=index,
                speaker=role_speaker_map[str(templates[key]["speaker_role"])],
            )
            for index, key in enumerate(sequence, start=1)
        ]
        setup = dict(content_plan.debate_setup or {})
        culprit = setup.get("culprit_name")
        rival = setup.get("rival_suspect_name")
        role_variants = self.role_variants()
        speaker_variants = self.speaker_variants_from_roles(role_variants, order)
        materialized_setup = {
            **setup,
            "ground_truth_side_order": order,
            "speaker_role_map": speaker_role_map,
            "role_speaker_map": role_speaker_map,
            "side_a_suspect": culprit if speaker_role_map["A"] == "culprit" else rival,
            "side_b_suspect": culprit if speaker_role_map["B"] == "culprit" else rival,
            "side_a_is_ground_truth": speaker_role_map["A"] == "culprit",
            "side_b_is_ground_truth": speaker_role_map["B"] == "culprit",
            "agent_a_stance": setup["culprit_stance"] if speaker_role_map["A"] == "culprit" else setup["rival_stance"],
            "agent_b_stance": setup["culprit_stance"] if speaker_role_map["B"] == "culprit" else setup["rival_stance"],
        }
        claim_realization_bundle = (
            content_plan.claim_realization_bundle
            or setup.get("claim_realization_bundle")
        )
        materialized_question = format_motion_from_claim_bundle(
            claim_realization_bundle,
            wrongdoing_event=setup.get("wrongdoing_event") or "the wrongdoing",
            first_suspect=materialized_setup["side_a_suspect"],
            second_suspect=materialized_setup["side_b_suspect"],
        )
        runtime_plan = content_plan.model_copy(update={
            "topic": materialized_question,
            "debate_question": materialized_question,
            "agent_a_stance": materialized_setup["agent_a_stance"],
            "agent_b_stance": materialized_setup["agent_b_stance"],
            "ground_truth_side_order": order,
            "claim_realization_bundle": claim_realization_bundle,
            "debate_setup": materialized_setup,
            "turns": turns,
        })
        return runtime_plan, {
            "speaker_role_map": speaker_role_map,
            "role_speaker_map": role_speaker_map,
            "speaker_variants": speaker_variants,
            "culprit_variant": role_variants["culprit"],
            "rival_variant": role_variants["rival"],
        }

    def load_content_plan_from_path(self, plan_path: str | Path) -> ContentPlan:
        path = self._resolve_repo_path(plan_path)
        return ContentPlan(**json.loads(path.read_text()))

    @staticmethod
    def validate_content_plan_pair(content_plan: ContentPlan, topic_cfg: dict) -> None:
        if content_plan.content_plan_version != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            return
        setup = content_plan.debate_setup or {}
        expected_correct = str(topic_cfg.get("correct_answer") or "").strip()
        expected_rival = str(topic_cfg.get("rival_suspect") or "").strip()
        actual_correct = str(setup.get("culprit_name") or "").strip()
        actual_rival = str(setup.get("rival_suspect_name") or "").strip()
        if not expected_correct or not expected_rival:
            raise ValueError(
                f"Data-driven suspect pair is missing for case={topic_cfg.get('case_id')}; "
                "refusing to use pairwise content plan."
            )
        if actual_correct != expected_correct or actual_rival != expected_rival:
            raise ValueError(
                f"Content plan suspect pair for case={topic_cfg.get('case_id')} is stale or manual: "
                f"plan correct={actual_correct!r} rival={actual_rival!r}; "
                f"data correct={expected_correct!r} rival={expected_rival!r}. "
                "Regenerate the content plan from the CSV-derived pair."
            )

    @staticmethod
    def validate_role_dialogue_pair(role_dialogue: RoleDialogue, content_plan: ContentPlan) -> None:
        if content_plan.content_plan_version != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            return
        setup = content_plan.debate_setup or {}
        expected_correct = str(setup.get("culprit_name") or "").strip()
        expected_rival = str(setup.get("rival_suspect_name") or "").strip()
        actual_correct = str(role_dialogue.culprit_name or "").strip()
        actual_rival = str(role_dialogue.rival_suspect_name or "").strip()
        if actual_correct != expected_correct or actual_rival != expected_rival:
            raise ValueError(
                f"Role dialogue suspect pair for case={role_dialogue.case_id} is stale or manual: "
                f"role_dialogue correct={actual_correct!r} rival={actual_rival!r}; "
                f"content_plan correct={expected_correct!r} rival={expected_rival!r}. "
                "Regenerate the role dialogue from the CSV-derived pair."
            )
        expected_family = content_plan.content_plan_family
        expected_interpretation = content_plan.interpretation_set_id
        if expected_family and role_dialogue.content_plan_family != expected_family:
            raise ValueError(
                f"Role dialogue family {role_dialogue.content_plan_family!r} does not match "
                f"content plan family {expected_family!r}."
            )
        if expected_interpretation and role_dialogue.interpretation_set_id != expected_interpretation:
            raise ValueError(
                f"Role dialogue interpretation_set_id {role_dialogue.interpretation_set_id!r} does not match "
                f"content plan interpretation_set_id {expected_interpretation!r}."
            )

    def _resolve_repo_path(self, path: str | Path) -> Path:
        return resolve_repo_path(path)

    def repo_display_path(self, path: str | Path) -> str:
        return repo_display_path(path)

    def save_content_plan(self, content_plan: ContentPlan, run_id: str) -> Path:
        case_id = self.config["run"]["case_id"]
        if content_plan.content_attention_question is None:
            raise ValueError(
                f"Content plan for case={case_id} is missing content_attention_question; "
                "generate or backfill it before saving."
            )
        if content_plan.content_plan_version == PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            family = content_plan.content_plan_family or self.content_plan_family()
            interpretation_set_id = content_plan.interpretation_set_id or self.interpretation_set_id()
            path = self.output_root() / "plans" / (
                f"{case_id}__{safe_artifact_part(family)}__{safe_artifact_part(interpretation_set_id)}.json"
            )
            payload = content_plan.model_dump(exclude_none=True)
        else:
            ground_truth_side_order = self.ground_truth_side_order()
            path = self.output_root() / "plans" / f"{case_id}_{ground_truth_side_order}_{run_id}.json"
            payload = content_plan.model_dump()
        payload["case_id"] = case_id
        payload.setdefault("content_plan_family", content_plan.content_plan_family or self.content_plan_family())
        payload.setdefault("interpretation_set_id", content_plan.interpretation_set_id or self.interpretation_set_id())
        if content_plan.content_plan_version != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            payload["ground_truth_side_order"] = self.ground_truth_side_order()
        validate_content_attention_question_payload(
            payload.get("content_attention_question"),
            evidence_bank=self.load_topic_cfg()["evidence_bank"],
            context=f"case={case_id} plan={path.name} content_attention_question",
        )
        dump_json(path, payload)
        return path

    def generate_role_opening_turns(
        self,
        *,
        topic_cfg: dict,
        content_plan: ContentPlan,
    ) -> dict[int, DialogueTurn]:
        if self.client is None:
            raise ValueError("OPENROUTER_API_KEY is required for opening generation.")
        runtime_plan, _metadata = self.materialize_content_plan_for_dialogue(
            content_plan,
            ground_truth_side_order="gt_first",
        )
        topic_cfg = dict(topic_cfg)
        topic_cfg["agent_a_stance"] = runtime_plan.agent_a_stance
        topic_cfg["agent_b_stance"] = runtime_plan.agent_b_stance
        topic_cfg["ground_truth_side_order"] = "gt_first"
        topic_cfg = self.rollout._topic_cfg_for_content_plan(topic_cfg, runtime_plan)
        memories = self.rollout._build_memories(topic_cfg, {})
        for memory in memories.values():
            memory.style_rules = []
        role_turns = {
            "culprit": runtime_plan.turns[0],
            "rival": runtime_plan.turns[1],
        }
        speaker_for_role = {"culprit": "A", "rival": "B"}
        cached_turns: dict[int, DialogueTurn] = {}
        for role in ("culprit", "rival"):
            turn_plan = role_turns[role]
            side_evidence = self.rollout._pairwise_opening_side_evidence(
                runtime_plan,
                speaker=speaker_for_role[role],
                evidence_bank=topic_cfg.get("evidence_bank"),
            )
            topic_cfg_for_turn = self.rollout._topic_cfg_with_opening_side_evidence(
                topic_cfg,
                side_evidence,
            )
            turn_plan_for_generation = self.rollout._opening_side_evidence_turn_plan(
                turn_plan,
                side_evidence,
            )
            turn = self.rollout._generate_turn(
                topic_cfg=topic_cfg_for_turn,
                content_plan=runtime_plan,
                turn_plan=turn_plan_for_generation,
                memory=memories[speaker_for_role[role]],
                trait_evaluator=None,
                trait_name=BASELINE_CONTROL_TRAIT_NAME,
                speaker_variant="baseline",
                eval_feedback=None,
                locked_content_contract=None,
                transformation_plan=None,
                visible_history_override="No dialogue history is provided. This turn is generated independently.",
            )
            cached_turns[turn_plan.turn_id] = turn.model_copy(update={
                "generated_from_role_opening": role,
            })
        return cached_turns

    def role_dialogue_path(
        self,
        *,
        trait_name: str | None = None,
        culprit_variant: str | None = None,
        rival_variant: str | None = None,
        rollout_index: int | None = None,
        seed: int | None = None,
    ) -> Path:
        role_variants = self.role_variants()
        requested_trait = trait_name or self.config["run"]["trait_name"]
        culprit = culprit_variant or role_variants["culprit"]
        rival = rival_variant or role_variants["rival"]
        is_canonical_baseline = culprit == "baseline" and rival == "baseline"
        safe_trait = (BASELINE_CONTROL_TRAIT_NAME if is_canonical_baseline else requested_trait).replace("/", "_")
        rollout = (
            CANONICAL_BASELINE_ROLLOUT_INDEX
            if is_canonical_baseline
            else (self.rollout_index() if rollout_index is None else int(rollout_index or 1))
        )
        seed_value = None if is_canonical_baseline else (self.config["run"].get("seed") if seed is None else seed)
        seed_part = f"seed-{seed_value}" if seed_value is not None else "seed-none"
        interpretation_suffix = (
            self.canonical_baseline_interpretation_artifact_suffix()
            if is_canonical_baseline
            else self.interpretation_artifact_suffix()
        )
        return self.output_root() / "role_dialogues" / (
            f"{self.config['run']['case_id']}__{interpretation_suffix}__{safe_trait}__"
            f"culprit-{culprit}__rival-{rival}__rollout-{rollout}__{seed_part}.json"
        )

    def save_role_dialogue(self, role_dialogue: RoleDialogue | dict, run_id: str) -> Path:
        del run_id
        path = self.role_dialogue_path(
            trait_name=role_dialogue.trait_name if isinstance(role_dialogue, RoleDialogue) else role_dialogue.get("trait_name"),
            culprit_variant=role_dialogue.culprit_variant if isinstance(role_dialogue, RoleDialogue) else role_dialogue.get("culprit_variant"),
            rival_variant=role_dialogue.rival_variant if isinstance(role_dialogue, RoleDialogue) else role_dialogue.get("rival_variant"),
            rollout_index=role_dialogue.rollout_index if isinstance(role_dialogue, RoleDialogue) else role_dialogue.get("rollout_index"),
            seed=role_dialogue.seed if isinstance(role_dialogue, RoleDialogue) else role_dialogue.get("seed"),
        )
        payload = (
            role_dialogue.model_dump(exclude_none=True)
            if isinstance(role_dialogue, RoleDialogue)
            else dict(role_dialogue)
        )
        if (
            payload.get("culprit_variant") == "baseline"
            and payload.get("rival_variant") == "baseline"
        ):
            payload["trait_name"] = BASELINE_CONTROL_TRAIT_NAME
            payload["content_plan_family"] = CANONICAL_BASELINE_CONTENT_PLAN_FAMILY
            payload["interpretation_set_id"] = CANONICAL_BASELINE_INTERPRETATION_SET_ID
            payload["rollout_index"] = CANONICAL_BASELINE_ROLLOUT_INDEX
            payload["seed"] = None
            payload.update(self.canonical_baseline_source_metadata())
        else:
            payload.setdefault("rollout_index", self.rollout_index())
        dump_json(path, payload)
        return path

    def load_role_dialogue_from_path(self, path: str | Path) -> RoleDialogue:
        resolved = self._resolve_repo_path(path)
        return RoleDialogue(**json.loads(resolved.read_text()))

    @staticmethod
    def _is_all_baseline_role_variants(role_variants: dict[str, str]) -> bool:
        return role_variants.get("culprit") == "baseline" and role_variants.get("rival") == "baseline"

    @staticmethod
    def _baseline_turn_ids_for_role_variants(role_variants: dict[str, str], trait_name: str | None = None) -> set[int]:
        # if trait_name == "anchoring_bias":
        #     if role_variants.get("culprit") == "active" and role_variants.get("rival") == "baseline":
        #         return {2, 4, 6}
        #     if role_variants.get("culprit") == "baseline" and role_variants.get("rival") == "active":
        #         return {1, 3, 5}
        # if trait_name == "pro_jargon_bias":
        #     if role_variants.get("culprit") == "active" and role_variants.get("rival") == "baseline":
        #         return {2, 4, 6}
        #     if role_variants.get("culprit") == "baseline" and role_variants.get("rival") == "active":
        #         return {1, 3, 5}
        turn_ids = {1, 2}
        if role_variants.get("culprit") == "baseline":
            turn_ids.update({3, 5})
        if role_variants.get("rival") == "baseline":
            turn_ids.update({4, 6})
        return turn_ids

    def _ensure_baseline_role_dialogue(
        self,
        *,
        topic_cfg: dict,
        content_plan: ContentPlan,
        role_style_bundle: dict,
        trait_name: str,
        run_id: str,
        content_plan_source_path: str | None,
        style_plan_source_path: str | None,
    ) -> tuple[RoleDialogue, Path]:
        baseline_trait_name = BASELINE_CONTROL_TRAIT_NAME
        canonical_content_plan, canonical_plan_path = self.load_or_generate_canonical_baseline_content_plan(
            topic_cfg,
            run_id,
        )
        baseline_path = self.role_dialogue_path(
            trait_name=baseline_trait_name,
            culprit_variant="baseline",
            rival_variant="baseline",
            rollout_index=CANONICAL_BASELINE_ROLLOUT_INDEX,
            seed=None,
        )
        if baseline_path.exists():
            candidate = self.load_role_dialogue_from_path(baseline_path)
            if (
                candidate.role_dialogue_version == ROLE_DIALOGUE_VERSION
                and candidate.case_opening_version == CASE_OPENING_VERSION
                and candidate.trait_name == BASELINE_CONTROL_TRAIT_NAME
                and self._is_all_baseline_role_variants({
                    "culprit": candidate.culprit_variant,
                    "rival": candidate.rival_variant,
                })
                and all(
                    role_turn_id in candidate.role_turns
                    for role_turn_id in (
                        "culprit_opening",
                        "rival_opening",
                        "culprit_argument",
                        "rival_argument",
                        "culprit_rebuttal",
                        "rival_rebuttal",
                    )
                )
            ):
                self.validate_role_dialogue_pair(candidate, canonical_content_plan)
                print(f"[detective-pipeline] loading canonical baseline role dialogue: {baseline_path}")
                return candidate, baseline_path
            print(f"[detective-pipeline] ignoring incompatible baseline role dialogue cache: {baseline_path}")

        print(f"[detective-pipeline] canonical baseline role dialogue not found; generating: {baseline_path}")
        run_cfg = self.config["run"]
        old_culprit = run_cfg.get("culprit_variant")
        old_rival = run_cfg.get("rival_variant")
        old_variant_a = run_cfg.get("variant_name_a")
        old_variant_b = run_cfg.get("variant_name_b")
        old_family = run_cfg.get("content_plan_family")
        old_interpretation = run_cfg.get("interpretation_set_id")
        old_rollout = run_cfg.get("rollout_index")
        try:
            run_cfg["culprit_variant"] = "baseline"
            run_cfg["rival_variant"] = "baseline"
            run_cfg["variant_name_a"] = "baseline"
            run_cfg["variant_name_b"] = "baseline"
            run_cfg["content_plan_family"] = CANONICAL_BASELINE_CONTENT_PLAN_FAMILY
            run_cfg["interpretation_set_id"] = CANONICAL_BASELINE_INTERPRETATION_SET_ID
            run_cfg["rollout_index"] = CANONICAL_BASELINE_ROLLOUT_INDEX
            baseline_style = self.style_bundle_for_role_variants(
                self.generate_role_style_bundle(
                    content_plan=canonical_content_plan,
                    trait_name=baseline_trait_name,
                ),
                {"culprit": "baseline", "rival": "baseline"},
            )
            return self.generate_role_dialogue(
                topic_cfg=topic_cfg,
                content_plan=canonical_content_plan,
                role_style_bundle=baseline_style,
                trait_name=baseline_trait_name,
                run_id=run_id,
                content_plan_source_path=self.repo_display_path(canonical_plan_path),
                style_plan_source_path=None,
            )
        finally:
            if old_culprit is None:
                run_cfg.pop("culprit_variant", None)
            else:
                run_cfg["culprit_variant"] = old_culprit
            if old_rival is None:
                run_cfg.pop("rival_variant", None)
            else:
                run_cfg["rival_variant"] = old_rival
            if old_variant_a is None:
                run_cfg.pop("variant_name_a", None)
            else:
                run_cfg["variant_name_a"] = old_variant_a
            if old_variant_b is None:
                run_cfg.pop("variant_name_b", None)
            else:
                run_cfg["variant_name_b"] = old_variant_b
            if old_family is None:
                run_cfg.pop("content_plan_family", None)
            else:
                run_cfg["content_plan_family"] = old_family
            if old_interpretation is None:
                run_cfg.pop("interpretation_set_id", None)
            else:
                run_cfg["interpretation_set_id"] = old_interpretation
            if old_rollout is None:
                run_cfg.pop("rollout_index", None)
            else:
                run_cfg["rollout_index"] = old_rollout

    def find_reusable_case_openings(self, content_plan: ContentPlan) -> tuple[dict[int, DialogueTurn], Path] | None:
        case_id = self.config["run"]["case_id"]
        search_dir = self.output_root() / "role_dialogues"
        candidates = sorted(
            search_dir.glob("*.json") if search_dir.exists() else [],
            key=lambda path: path.stat().st_mtime,
        )
        for path in candidates:
            try:
                role_dialogue = self.load_role_dialogue_from_path(path)
            except Exception:
                continue
            if role_dialogue.case_id != case_id:
                continue
            if (role_dialogue.rollout_index or 1) != self.rollout_index():
                continue
            if role_dialogue.case_opening_version != CASE_OPENING_VERSION:
                continue
            try:
                self.validate_role_dialogue_pair(role_dialogue, content_plan)
            except ValueError:
                continue
            culprit_opening = role_dialogue.role_turns.get("culprit_opening")
            rival_opening = role_dialogue.role_turns.get("rival_opening")
            if not culprit_opening or not rival_opening:
                continue
            if not culprit_opening.utterance or not rival_opening.utterance:
                continue
            return {
                1: DialogueTurn(
                    turn_id=1,
                    speaker="A",
                    utterance=culprit_opening.utterance,
                    opponent_claim_targeted=culprit_opening.opponent_claim_targeted,
                    attack_move_used=culprit_opening.attack_move_used,
                    evidence_citations=list(culprit_opening.evidence_citations or []),
                    content_fidelity_note=culprit_opening.content_fidelity_note,
                    content_preservation_note=culprit_opening.content_preservation_note,
                    treatment_realization_note=culprit_opening.treatment_realization_note,
                    generated_from_role_opening="culprit",
                ),
                2: DialogueTurn(
                    turn_id=2,
                    speaker="B",
                    utterance=rival_opening.utterance,
                    opponent_claim_targeted=rival_opening.opponent_claim_targeted,
                    attack_move_used=rival_opening.attack_move_used,
                    evidence_citations=list(rival_opening.evidence_citations or []),
                    content_fidelity_note=rival_opening.content_fidelity_note,
                    content_preservation_note=rival_opening.content_preservation_note,
                    treatment_realization_note=rival_opening.treatment_realization_note,
                    generated_from_role_opening="rival",
                ),
            }, path
        return None

    @staticmethod
    def _role_dialogue_turn_from_dialogue_turn(
        turn: DialogueTurn,
        *,
        speaker_role: str,
        role_turn_id: str,
    ) -> dict:
        replies_to = None
        if role_turn_id == "culprit_rebuttal":
            replies_to = "rival_argument"
        elif role_turn_id == "rival_rebuttal":
            replies_to = "culprit_argument"
        return {
            "speaker_role": speaker_role,
            "role_turn_id": role_turn_id,
            "utterance": turn.utterance,
            "opponent_claim_targeted": turn.opponent_claim_targeted,
            "attack_move_used": turn.attack_move_used,
            "content_fidelity_note": turn.content_fidelity_note,
            "evidence_citations": list(turn.evidence_citations or []),
            "content_preservation_note": turn.content_preservation_note,
            "treatment_realization_note": turn.treatment_realization_note,
            "generated_from_role_opening": turn.generated_from_role_opening,
            "replies_to_role_turn_id": replies_to,
        }

    def generate_role_dialogue(
        self,
        *,
        topic_cfg: dict,
        content_plan: ContentPlan,
        role_style_bundle: dict,
        trait_name: str,
        run_id: str,
        content_plan_source_path: str | None,
        style_plan_source_path: str | None,
    ) -> tuple[RoleDialogue, Path]:
        if self.client is None:
            raise ValueError("OPENROUTER_API_KEY is required for role dialogue generation.")
        role_variants = self.role_variants()
        is_all_baseline = self._is_all_baseline_role_variants(role_variants)
        if is_all_baseline:
            content_plan, canonical_plan_path = self.load_or_generate_canonical_baseline_content_plan(
                topic_cfg,
                run_id,
            )
            content_plan_source_path = self.repo_display_path(canonical_plan_path)
            style_plan_source_path = None
            role_style_bundle = self.style_bundle_for_role_variants(
                self.generate_role_style_bundle(
                    content_plan=content_plan,
                    trait_name=BASELINE_CONTROL_TRAIT_NAME,
                ),
                {"culprit": "baseline", "rival": "baseline"},
            )
        generation_trait_name = (
            BASELINE_CONTROL_TRAIT_NAME
            if is_all_baseline
            else trait_name
        )
        runtime_plan, _metadata = self.materialize_content_plan_for_dialogue(
            content_plan,
            ground_truth_side_order="gt_first",
        )
        speaker_variants = self.speaker_variants_from_roles(role_variants, "gt_first")
        style_bundle = self.role_style_bundle_to_speaker_bundle(
            role_style_bundle,
            ground_truth_side_order="gt_first",
        )
        baseline_role_dialogue = None
        baseline_source_role_dialogue = None
        baseline_fixed_dialogue = None
        baseline_fixed_turn_ids: set[int] = set()
        if not self._is_all_baseline_role_variants(role_variants):
            baseline_role_dialogue, baseline_path = self._ensure_baseline_role_dialogue(
                topic_cfg=topic_cfg,
                content_plan=content_plan,
                role_style_bundle=role_style_bundle,
                trait_name=trait_name,
                run_id=run_id,
                content_plan_source_path=content_plan_source_path,
                style_plan_source_path=style_plan_source_path,
            )
            baseline_source_role_dialogue = self.repo_display_path(baseline_path)
            baseline_fixed_dialogue = self.materialize_role_dialogue(
                baseline_role_dialogue,
                ground_truth_side_order="gt_first",
            )
            baseline_fixed_turn_ids = self._baseline_turn_ids_for_role_variants(
                role_variants,
                trait_name=trait_name,
            )
            print(
                "[detective-pipeline] reusing baseline role turns from: "
                f"{baseline_path}; fixed turn ids={sorted(baseline_fixed_turn_ids)}"
            )
        reusable_openings = None if baseline_role_dialogue is not None else self.find_reusable_case_openings(content_plan)
        opening_source_role_dialogue = None
        if baseline_role_dialogue is not None:
            cached_openings = {}
            opening_source_role_dialogue = baseline_source_role_dialogue
        elif reusable_openings is None:
            print("[detective-pipeline] no reusable case openings found; generating Turn 1/2 once")
            cached_openings = self.generate_role_opening_turns(
                topic_cfg=topic_cfg,
                content_plan=content_plan,
            )
        else:
            cached_openings, opening_source_path = reusable_openings
            opening_source_role_dialogue = self.repo_display_path(opening_source_path)
            print(f"[detective-pipeline] reusing case-level Turn 1/2 from: {opening_source_path}")
        runtime_topic_cfg = dict(topic_cfg)
        runtime_topic_cfg["agent_a_stance"] = runtime_plan.agent_a_stance
        runtime_topic_cfg["agent_b_stance"] = runtime_plan.agent_b_stance
        runtime_topic_cfg["ground_truth_side_order"] = "gt_first"
        runtime_topic_cfg["trait_name"] = generation_trait_name
        runtime_topic_cfg["culprit_variant"] = role_variants["culprit"]
        runtime_topic_cfg["rival_variant"] = role_variants["rival"]
        dialogue = self.generate_dialogue(
            topic_cfg=runtime_topic_cfg,
            content_plan=runtime_plan,
            style_bundle=style_bundle,
            trait_name=generation_trait_name,
            speaker_variants=speaker_variants,
            cached_opening_turns=cached_openings,
            fixed_dialogue=baseline_fixed_dialogue,
            fixed_turn_ids=baseline_fixed_turn_ids,
        )
        role_ids_by_turn_id = {
            1: ("culprit", "culprit_opening"),
            2: ("rival", "rival_opening"),
            3: ("culprit", "culprit_argument"),
            4: ("rival", "rival_argument"),
            5: ("culprit", "culprit_rebuttal"),
            6: ("rival", "rival_rebuttal"),
        }
        turns_by_id = {turn.turn_id: turn for turn in dialogue.turns}
        role_turns = {}
        for turn_id, (role, role_turn_id) in role_ids_by_turn_id.items():
            role_turns[role_turn_id] = self._role_dialogue_turn_from_dialogue_turn(
                turns_by_id[turn_id],
                speaker_role=role,
                role_turn_id=role_turn_id,
            )
        setup = content_plan.debate_setup or {}
        is_canonical_baseline = is_all_baseline
        role_dialogue_family = (
            CANONICAL_BASELINE_CONTENT_PLAN_FAMILY
            if is_canonical_baseline
            else (content_plan.content_plan_family or self.content_plan_family())
        )
        role_dialogue_interpretation = (
            CANONICAL_BASELINE_INTERPRETATION_SET_ID
            if is_canonical_baseline
            else (content_plan.interpretation_set_id or self.interpretation_set_id())
        )
        role_dialogue_rollout = (
            CANONICAL_BASELINE_ROLLOUT_INDEX
            if is_canonical_baseline
            else self.rollout_index()
        )
        role_dialogue = RoleDialogue(
            role_dialogue_version=ROLE_DIALOGUE_VERSION,
            case_opening_version=CASE_OPENING_VERSION,
            case_id=self.config["run"]["case_id"],
            content_plan_family=role_dialogue_family,
            interpretation_set_id=role_dialogue_interpretation,
            trait_name=generation_trait_name,
            culprit_variant=role_variants["culprit"],
            rival_variant=role_variants["rival"],
            rollout_index=role_dialogue_rollout,
            seed=None if is_canonical_baseline else self.config["run"].get("seed"),
            topic=content_plan.topic,
            culprit_name=setup.get("culprit_name"),
            rival_suspect_name=setup.get("rival_suspect_name"),
            culprit_stance=setup.get("culprit_stance"),
            rival_stance=setup.get("rival_stance"),
            content_plan_source=content_plan_source_path,
            style_plan_source=style_plan_source_path,
            opening_source_role_dialogue=opening_source_role_dialogue,
            baseline_source_role_dialogue=baseline_source_role_dialogue,
            baseline_canonical_source=self.canonical_baseline_source_metadata(baseline_source_role_dialogue),
            role_turns=role_turns,
        )
        return role_dialogue, self.save_role_dialogue(role_dialogue, run_id)

    def materialize_role_dialogue(
        self,
        role_dialogue: RoleDialogue,
        *,
        ground_truth_side_order: str,
    ) -> Dialogue:
        speaker_role_map = self.speaker_role_map_for_order(ground_truth_side_order)
        speaker_for_role = {role: speaker for speaker, role in speaker_role_map.items()}
        sequence = (
            ["culprit_opening", "rival_opening", "culprit_argument", "rival_argument", "culprit_rebuttal", "rival_rebuttal"]
            if ground_truth_side_order == "gt_first"
            else ["rival_opening", "culprit_opening", "rival_argument", "culprit_argument", "rival_rebuttal", "culprit_rebuttal"]
        )
        role_variants = {
            "culprit": role_dialogue.culprit_variant,
            "rival": role_dialogue.rival_variant,
        }
        speaker_variants = self.speaker_variants_from_roles(role_variants, ground_truth_side_order)
        turns = []
        for turn_id, role_turn_id in enumerate(sequence, start=1):
            role_turn = role_dialogue.role_turns[role_turn_id]
            turns.append(DialogueTurn(
                turn_id=turn_id,
                speaker=speaker_for_role[role_turn.speaker_role],
                speaker_role=role_turn.speaker_role,
                role_turn_id=role_turn.role_turn_id,
                utterance=role_turn.utterance,
                opponent_claim_targeted=role_turn.opponent_claim_targeted,
                attack_move_used=role_turn.attack_move_used,
                content_fidelity_note=role_turn.content_fidelity_note,
                evidence_citations=list(role_turn.evidence_citations or []),
                content_preservation_note=role_turn.content_preservation_note,
                treatment_realization_note=role_turn.treatment_realization_note,
                generated_from_role_opening=role_turn.generated_from_role_opening,
            ))
        agent_a_stance = (
            role_dialogue.culprit_stance
            if speaker_role_map["A"] == "culprit"
            else role_dialogue.rival_stance
        )
        agent_b_stance = (
            role_dialogue.culprit_stance
            if speaker_role_map["B"] == "culprit"
            else role_dialogue.rival_stance
        )
        return Dialogue(
            topic=role_dialogue.topic or "",
            case_id=role_dialogue.case_id,
            content_plan_family=role_dialogue.content_plan_family,
            interpretation_set_id=role_dialogue.interpretation_set_id,
            trait_name=role_dialogue.trait_name,
            variant_name_a=speaker_variants["A"],
            variant_name_b=speaker_variants["B"],
            ground_truth_side_order=ground_truth_side_order,
            agent_a_stance=agent_a_stance,
            agent_b_stance=agent_b_stance,
            turns=turns,
        )

    def content_plan_dialogue_metadata(self, content_plan: ContentPlan) -> dict:
        attention_question = content_plan.content_attention_question
        metadata: dict = {}
        if attention_question is not None:
            metadata["content_attention_question"] = attention_question.model_dump()
        if content_plan.private_truth_used:
            metadata["private_truth_used_in_planning"] = True
        if content_plan.official_solution_public_evidence_map:
            metadata["official_solution_public_evidence_map"] = content_plan.official_solution_public_evidence_map
        if content_plan.outcome_only_revelations_to_exclude:
            metadata["outcome_only_revelations_to_exclude"] = content_plan.outcome_only_revelations_to_exclude
        if content_plan.suspect_evidence_map:
            metadata["suspect_evidence_map"] = content_plan.suspect_evidence_map
        if content_plan.reason_units:
            metadata["reason_units"] = content_plan.reason_units
        if content_plan.content_plan_version:
            metadata["content_plan_version"] = content_plan.content_plan_version
        if content_plan.content_plan_family:
            metadata["content_plan_family"] = content_plan.content_plan_family
        if content_plan.interpretation_set_id:
            metadata["interpretation_set_id"] = content_plan.interpretation_set_id
        if content_plan.selected_interpretations:
            metadata["selected_interpretations"] = content_plan.selected_interpretations
        if content_plan.debate_setup:
            metadata["debate_setup"] = content_plan.debate_setup
        if content_plan.reasoning_finding_source_path:
            metadata["reasoning_finding_source"] = content_plan.reasoning_finding_source_path
        if content_plan.selected_reasons:
            metadata["selected_reasons"] = content_plan.selected_reasons
        if content_plan.role_turn_templates:
            metadata["role_turn_templates"] = content_plan.role_turn_templates
        if content_plan.turn_dependencies:
            metadata["turn_dependencies"] = content_plan.turn_dependencies
        if content_plan.evidence_utility_classification:
            metadata["evidence_utility_classification"] = content_plan.evidence_utility_classification
        return metadata

    def dialogue_runtime_metadata(self) -> dict:
        return dict(self.rollout.last_opening_coverage or {})

    def backfill_content_attention_question(
        self,
        plan_path: str | Path,
        topic_cfg: dict,
        *,
        force: bool = False,
    ) -> tuple[Path, bool]:
        path = self._resolve_repo_path(plan_path)
        payload = json.loads(path.read_text())
        content_plan = ContentPlan(**payload)
        if content_plan.content_attention_question is not None and not force:
            return path, False

        attention_question = self.planner.generate_content_attention_question(
            content_plan=content_plan,
            evidence_bank=topic_cfg["evidence_bank"],
        )
        payload["content_attention_question"] = attention_question.model_dump()
        payload.setdefault("case_id", self.config["run"]["case_id"])
        if payload.get("content_plan_version") != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            payload.setdefault("ground_truth_side_order", self.ground_truth_side_order())
        dump_json(path, payload)
        return path, True

    def generate_content_plan(
        self,
        topic_cfg: dict,
        run_id: str,
        *,
        reasoning_finding: dict | None = None,
        reasoning_finding_source_path: str | None = None,
        rival_suspect: str | None = None,
    ) -> tuple[ContentPlan, Path]:
        static_reasoning_finding = topic_cfg.get("reasoning_finding")
        if static_reasoning_finding is None:
            raise ValueError(
                "Current detective pipeline requires a complete static reasoning_finding in the converted case JSON; "
                "rerun `python code/convert_true_detective.py`. "
                "The legacy planner and live reason generation are disabled."
            )
        if reasoning_finding is not None and reasoning_finding != static_reasoning_finding:
            raise ValueError(
                "Content-plan generation uses only topic_cfg['reasoning_finding'] from the converted case JSON; "
                "remove the external reasoning_finding override."
            )
        static_reasoning_finding = validate_reasoning_finding(static_reasoning_finding, topic_cfg=topic_cfg)
        selected_rival = str(topic_cfg.get("rival_suspect") or "").strip()
        if not selected_rival:
            raise ValueError(
                "Data-driven rival suspect is missing from topic_cfg; refusing to generate a pairwise plan."
            )
        if rival_suspect is not None and str(rival_suspect).strip() != selected_rival:
            raise ValueError(
                f"Manual rival_suspect={rival_suspect!r} conflicts with data-driven rival "
                f"{selected_rival!r} for case={topic_cfg.get('case_id')}. "
                "Remove the manual rival override."
            )
        content_plan = self.planner.build_pairwise_content_plan(
            topic_cfg=topic_cfg,
            reasoning_finding=static_reasoning_finding,
            reasoning_finding_source_path=(
                reasoning_finding_source_path
                or topic_cfg.get("converted_case_source")
            ),
            rival_suspect=selected_rival,
            content_plan_family=self.content_plan_family(),
            interpretation_set_id=self.interpretation_set_id(),
        )
        return content_plan, self.save_content_plan(content_plan, run_id)

    def load_or_generate_canonical_baseline_content_plan(
        self,
        topic_cfg: dict,
        run_id: str,
    ) -> tuple[ContentPlan, Path]:
        canonical_path = self.output_root() / "plans" / (
            f"{self.config['run']['case_id']}__"
            f"{safe_artifact_part(CANONICAL_BASELINE_CONTENT_PLAN_FAMILY)}__"
            f"{safe_artifact_part(CANONICAL_BASELINE_INTERPRETATION_SET_ID)}.json"
        )
        if canonical_path.exists():
            content_plan = self.load_content_plan_from_path(canonical_path)
            self.validate_content_plan_pair(content_plan, topic_cfg)
            if (
                content_plan.content_plan_family == CANONICAL_BASELINE_CONTENT_PLAN_FAMILY
                and content_plan.interpretation_set_id == CANONICAL_BASELINE_INTERPRETATION_SET_ID
            ):
                return content_plan, canonical_path
            raise ValueError(
                f"Canonical baseline content plan path {canonical_path} contains "
                f"family={content_plan.content_plan_family!r} "
                f"interp={content_plan.interpretation_set_id!r}."
            )

        static_reasoning_finding = topic_cfg.get("reasoning_finding")
        if static_reasoning_finding is None:
            raise ValueError("Cannot generate canonical baseline content plan without topic_cfg['reasoning_finding'].")
        static_reasoning_finding = validate_reasoning_finding(static_reasoning_finding, topic_cfg=topic_cfg)
        selected_rival = str(topic_cfg.get("rival_suspect") or "").strip()
        if not selected_rival:
            raise ValueError("Data-driven rival suspect is missing from topic_cfg; refusing to generate canonical baseline plan.")
        content_plan = self.planner.build_pairwise_content_plan(
            topic_cfg=topic_cfg,
            reasoning_finding=static_reasoning_finding,
            reasoning_finding_source_path=topic_cfg.get("converted_case_source"),
            rival_suspect=selected_rival,
            content_plan_family=CANONICAL_BASELINE_CONTENT_PLAN_FAMILY,
            interpretation_set_id=CANONICAL_BASELINE_INTERPRETATION_SET_ID,
        )
        return content_plan, self.save_content_plan(content_plan, run_id)

    def build_style_bundle(
        self,
        trait_name: str,
        speaker_variants: dict[str, str],
        eval_feedback: str | None = None,
    ) -> dict:
        del trait_name, speaker_variants, eval_feedback
        raise ValueError(
            "Detective style plans must be generated from a pairwise content plan "
            "with generate_role_style_bundle()."
        )

    def build_role_style_bundle(
        self,
        trait_name: str,
        role_variants: dict[str, str],
    ) -> dict:
        del trait_name, role_variants
        raise ValueError(
            "Detective role style plans must be generated from a pairwise content plan "
            "with generate_role_style_bundle()."
        )

    @staticmethod
    def _reasoning_slot_ids_for_speaker(content_plan: ContentPlan, speaker: str) -> list[str]:
        slot_ids: list[str] = []
        for turn in content_plan.turns:
            if turn.speaker != speaker:
                continue
            if turn.turn_type not in {"argument", "rebuttal", "summary"}:
                continue
            for job in turn.sentence_jobs or []:
                if isinstance(job, dict) and job.get("slot_id"):
                    slot_id = str(job["slot_id"])
                    if slot_id not in slot_ids:
                        slot_ids.append(slot_id)
        return slot_ids

    @staticmethod
    def _reasoning_slot_ids_for_role(content_plan: ContentPlan, role: str) -> list[str]:
        slot_ids: list[str] = []
        templates = content_plan.role_turn_templates or {}
        for template in templates.values():
            if not isinstance(template, dict) or template.get("speaker_role") != role:
                continue
            if template.get("turn_type") not in {"argument", "rebuttal", "summary"}:
                continue
            for job in template.get("sentence_jobs") or []:
                if isinstance(job, dict) and job.get("slot_id"):
                    slot_id = str(job["slot_id"])
                    if slot_id not in slot_ids:
                        slot_ids.append(slot_id)
        return slot_ids

    @staticmethod
    def _reasoning_slot_contexts_for_role(content_plan: ContentPlan, role: str) -> dict[str, dict]:
        contexts: dict[str, dict] = {}
        templates = content_plan.role_turn_templates or {}
        setup = content_plan.debate_setup or {}
        wrongdoing_event = normalize_wrongdoing_event(setup.get("wrongdoing_event"))
        claim_realization_bundle = content_plan.claim_realization_bundle or setup.get("claim_realization_bundle")
        selected_reasons = content_plan.selected_reasons or {}
        reasons_by_id = {
            str(reason.get("reason_id")): reason
            for reason in selected_reasons.values()
            if isinstance(reason, dict) and reason.get("reason_id")
        }
        for role_turn_id, template in templates.items():
            if not isinstance(template, dict) or template.get("speaker_role") != role:
                continue
            if template.get("turn_type") not in {"argument", "rebuttal", "summary"}:
                continue
            for job in template.get("sentence_jobs") or []:
                if not isinstance(job, dict) or not job.get("slot_id"):
                    continue
                slot_id = str(job["slot_id"])
                reason_refs = list(template.get("reason_refs") or [])
                reason_id = job.get("reason_id") or (reason_refs[0] if len(reason_refs) == 1 else None)
                reason = reasons_by_id.get(str(reason_id or "")) or {}
                evidence_ids = list(reason.get("evidence_ids") or template.get("evidence_ids") or [])
                local_purpose = DetectivePipelineRunner._slot_local_purpose({
                    "turn_type": template.get("turn_type"),
                    "job": job.get("job"),
                })
                subject_suspect = template.get("subject_suspect")
                fact_units = slot_relevant_fact_units(
                    list(reason.get("fact_units") or []),
                    subject_suspect=subject_suspect,
                    local_purpose=local_purpose,
                )
                if not fact_units:
                    fact_units = observable_fact_units(list(template.get("fact_units") or []))
                required_conclusion = template.get("core_conclusion")
                opponent_source_role = None
                accusation_being_weakened = None
                if local_purpose == "weaken_opponent":
                    opponent_source_role = "rival" if role == "culprit" else "culprit"
                    accusation_being_weakened = (
                        accusation_clause_for_suspect(
                            claim_realization_bundle,
                            subject_suspect,
                            wrongdoing_event,
                        )
                        if subject_suspect
                        else None
                    )
                claim_target = template.get("opponent_claim_target")
                if not claim_target:
                    claim_target = accusation_being_weakened or required_conclusion
                contexts[slot_id] = {
                    "slot_id": slot_id,
                    "local_purpose": local_purpose,
                    "speaker_role": role,
                    "role_turn_id": role_turn_id,
                    "turn_type": template.get("turn_type"),
                    "sentence_index": job.get("sentence_index"),
                    "job": job.get("job"),
                    "reason_id": reason_id,
                    "goal": job.get("goal"),
                    "subject_suspect": subject_suspect,
                    "opponent_source_role": opponent_source_role,
                    "accusation_being_weakened": accusation_being_weakened,
                    "reason_refs": reason_refs,
                    "evidence_ids": evidence_ids,
                    "fact_units": fact_units,
                    "reason_direction": reason.get("direction"),
                    "claim_target": claim_target,
                    "required_conclusion": required_conclusion,
                    "core_conclusion": required_conclusion,
                }
        return contexts

    @staticmethod
    def _slot_local_purpose(slot_context: dict) -> str:
        if slot_context.get("turn_type") == "argument":
            return "support_guilt"
        if slot_context.get("job") == "reject_opponent_incriminating_reason":
            return "weaken_opponent"
        return "support_innocence"

    @staticmethod
    def _slot_fact_text(slot_context: dict) -> str:
        facts = [
            str(item).strip()
            for item in slot_context.get("fact_units") or []
            if str(item).strip()
        ]
        return "; ".join(facts) if facts else "the locked public facts"

    @staticmethod
    def _is_fallacy_style_trait(trait_name: str | None) -> bool:
        return trait_name == "fallacy_trait" or str(trait_name or "").startswith("fallacy_trait__")

    @staticmethod
    def _style_plan_trait_name(trait_name: str | None) -> str:
        if trait_name == "pro_jargon_bias":
            return BASELINE_CONTROL_TRAIT_NAME
        return str(trait_name or BASELINE_CONTROL_TRAIT_NAME)

    @staticmethod
    def _deterministic_guideline_for_purpose(local_purpose: str | None) -> str:
        if local_purpose == "support_guilt":
            return "Explain how the fixed incriminating facts support the assigned guilt conclusion."
        if local_purpose == "weaken_opponent":
            return "Explain why the opponent's fixed incriminating facts do not sufficiently establish the accusation."
        if local_purpose == "support_innocence":
            return "Explain how the fixed exculpatory facts weaken the accusation and support the assigned innocence conclusion."
        return "Explain how the fixed facts support the assigned conclusion."

    @staticmethod
    def _style_bias_keys(trait_name: str) -> list[str]:
        fallacy_subtype = trait_name.removeprefix("fallacy_trait__") if trait_name.startswith("fallacy_trait__") else None
        return [fallacy_subtype] if fallacy_subtype else [trait_name]

    @staticmethod
    def _slot_role_from_id(slot_id: str) -> str:
        if str(slot_id).startswith("culprit_"):
            return "culprit"
        if str(slot_id).startswith("rival_"):
            return "rival"
        raise ValueError(f"Cannot infer role from reasoning slot id {slot_id!r}.")

    @staticmethod
    def _local_purpose_from_slot_id(slot_id: str) -> str:
        text = str(slot_id)
        if text.endswith("_support_reasoning"):
            return "support_guilt"
        if "_rebuttal_of_" in text:
            return "weaken_opponent"
        if text.endswith("_exculpatory_reasoning"):
            return "support_innocence"
        return "support_guilt"

    @staticmethod
    def _runtime_style_guideline_entry(
        *,
        context: dict,
        baseline_guideline: str,
        biased_guideline: dict[str, str],
    ) -> dict:
        entry = {
            "speaker_role": context.get("speaker_role"),
            "role_turn_id": context.get("role_turn_id"),
            "sentence_index": context.get("sentence_index"),
            "local_purpose": context.get("local_purpose"),
            "reason_id": context.get("reason_id"),
            "subject_suspect": context.get("subject_suspect"),
            "claim_target": context.get("claim_target") or context.get("required_conclusion"),
            "required_conclusion": context.get("required_conclusion") or context.get("core_conclusion"),
            "evidence_ids": list(context.get("evidence_ids") or []),
            "fact_units": list(context.get("fact_units") or []),
            "baseline_guideline": baseline_guideline,
            "biased_guideline": dict(biased_guideline),
        }
        return {
            key: value
            for key, value in entry.items()
            if value is not None and value != "" and value != {}
        }

    def _validate_style_reasoning_payload(
        self,
        payload: dict,
        *,
        contexts: dict[str, dict],
        bias_keys: list[str],
    ) -> dict[str, dict]:
        by_slot = payload.get("guidelines")
        if not isinstance(by_slot, dict):
            raise ValueError("style reasoning payload must contain guidelines object.")
        missing = sorted(set(contexts) - set(by_slot))
        extra = sorted(set(by_slot) - set(contexts))
        if missing or extra:
            raise ValueError(
                "style guideline slot mismatch: "
                f"missing={missing or []} extra={extra or []}"
            )
        out: dict[str, dict] = {}
        for slot_id, context in contexts.items():
            entry = by_slot[slot_id]
            if not isinstance(entry, dict):
                raise ValueError(f"{slot_id} guideline entry must be an object.")
            if set(entry) != {"baseline", "biased"}:
                raise ValueError(f"{slot_id} guideline entry must contain only baseline and biased.")
            baseline = str(entry.get("baseline") or "").strip()
            if not baseline:
                raise ValueError(f"{slot_id}.baseline guideline must be non-empty.")
            raw_biased = entry.get("biased")
            biased_text = str(raw_biased or "").strip()
            if not biased_text:
                raise ValueError(f"{slot_id}.biased guideline must be non-empty.")
            biased = {bias_key: biased_text for bias_key in bias_keys}
            out[slot_id] = self._runtime_style_guideline_entry(
                context=context,
                baseline_guideline=baseline,
                biased_guideline=biased,
            )
        return out

    def _generate_style_reasoning_payload(
        self,
        *,
        content_plan: ContentPlan,
        role: str,
        trait_name: str,
        contexts: dict[str, dict],
        bias_keys: list[str],
    ) -> dict[str, dict]:
        client = getattr(self, "client", None)
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for API style-plan generation.")

        last_error: Exception | None = None
        feedback = ""
        for attempt in range(1, 4):
            print(f"[detective-pipeline] calling style-plan API for role={role} attempt={attempt}/3")
            prompt_contexts = {
                slot_id: {
                    **context,
                    "slot_local_purpose": self._slot_local_purpose(context),
                }
                for slot_id, context in contexts.items()
            }
            prompt = build_detective_style_reasoning_prompt(
                content_plan=content_plan,
                role=role,
                trait_name=trait_name,
                contexts=prompt_contexts,
                bias_keys=bias_keys,
                trait_library=traitsV2.TRAIT_LIBRARY,
            )
            if feedback:
                prompt = f"{prompt}\n\nPrevious attempt failed validation: {feedback}"
            payload = client.complete_json(
                model=MODEL_CONFIGS["planner"]["model"],
                prompt=prompt,
                temperature=MODEL_CONFIGS["planner"]["temperature"],
                max_tokens=2400,
            )
            try:
                return self._validate_style_reasoning_payload(
                    payload,
                    contexts=contexts,
                    bias_keys=bias_keys,
                )
            except ValueError as exc:
                last_error = exc
                feedback = str(exc)
                print(f"[detective-pipeline] style reasoning attempt {attempt}/3 failed: {exc}")
        raise last_error or ValueError("style reasoning generation failed.")

    def _reasoning_slot_realizations_for_role(
        self,
        *,
        content_plan: ContentPlan,
        role: str,
        plan: TransformationPlan,
    ) -> dict[str, dict]:
        contexts = self._reasoning_slot_contexts_for_role(content_plan, role)
        if not contexts:
            return {}
        trait_name = plan.trait_name
        bias_keys = self._style_bias_keys(trait_name)
        return self._generate_style_reasoning_payload(
            content_plan=content_plan,
            role=role,
            trait_name=trait_name,
            contexts=contexts,
            bias_keys=bias_keys,
        )

    def _base_api_transformation_plan(
        self,
        *,
        trait_name: str,
        variant: str,
        slot_ids: list[str],
    ) -> TransformationPlan:
        return self._base_role_transformation_plan(
            trait_name=trait_name,
            variant=variant,
            slot_ids=slot_ids,
            style_plan_generation="api",
        )

    def _base_role_transformation_plan(
        self,
        *,
        trait_name: str,
        variant: str,
        slot_ids: list[str],
        style_plan_generation: str,
    ) -> TransformationPlan:
        fallacy_subtype = trait_name.removeprefix("fallacy_trait__") if trait_name.startswith("fallacy_trait__") else None
        base_plan = self.planner.build_deterministic_transformation_plan(
            trait_name=trait_name,
            variant_name=variant,
        )
        structure = resolve_dialogue_structure(
            trait_name=trait_name,
            variant_name=variant,
        )
        branch_metadata: dict = {
            **(base_plan.branch_metadata or {}),
            "style_plan_generation": style_plan_generation,
            "target_reasoning_slot_ids": slot_ids,
            "hook_unit_ids": slot_ids,
            "dialogue_structure_id": structure.structure_id,
        }
        if self._is_fallacy_style_trait(trait_name):
            branch_metadata["fallacy_target_sentence_slots"] = {
                "argument": ["S3"],
                "rebuttal": ["S2", "S5"],
            }
            branch_metadata["fixed_sentence_slots_not_transformable"] = {
                "argument": ["S1", "S2"],
                "rebuttal": ["S1", "S3", "S4"],
            }
        if fallacy_subtype:
            branch_metadata.update({
                "fallacy_subtype": fallacy_subtype,
            })
        update = {
            "variant": variant,
            "fallacy_subtype": fallacy_subtype,
            "hook_unit_ids": slot_ids,
            "branch_metadata": branch_metadata,
        }
        if self._is_fallacy_style_trait(trait_name):
            update["allowed_surface_operations"] = [
                "apply the fallacy only in the target reasoning slots: Argument S3, Rebuttal S2, and Rebuttal S5",
                "do not alter fixed fact sentences, conclusion-only sentences, evidence citations, or sentence functions",
                "preserve the baseline dialogue structure while changing only designated reasoning wording",
            ]
            update["allowed_inference_operations"] = [
                "use API-generated case-specific guidelines only for target_reasoning_slot_ids",
                "preserve locked facts, citations, conclusions, and non-reasoning sentence content",
            ]
        return base_plan.model_copy(update=update)

    def _deterministic_reasoning_payload(
        self,
        *,
        content_plan: ContentPlan,
        role: str,
        trait_name: str,
        contexts: dict[str, dict],
    ) -> dict[str, dict]:
        bias_keys = self._style_bias_keys(trait_name)
        out: dict[str, dict] = {}
        for slot_id, context in contexts.items():
            guideline = self._deterministic_guideline_for_purpose(context.get("local_purpose"))
            biased_guideline = {bias_key: guideline for bias_key in bias_keys}
            out[slot_id] = self._runtime_style_guideline_entry(
                context=context,
                baseline_guideline=guideline,
                biased_guideline=biased_guideline,
            )
        return out

    def generate_api_role_style_bundle(
        self,
        *,
        content_plan: ContentPlan,
        trait_name: str,
    ) -> dict:
        if not self._is_fallacy_style_trait(trait_name):
            raise ValueError(
                f"API style-plan generation is only enabled for fallacy traits; got {trait_name!r}."
            )
        if getattr(self, "client", None) is None:
            raise ValueError("OPENROUTER_API_KEY is required for API style-plan generation.")
        if content_plan.content_plan_version != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            raise ValueError(
                f"API style-plan generation requires {PAIRWISE_ROLE_CONTENT_PLAN_VERSION}; "
                f"got {content_plan.content_plan_version!r}."
            )
        style_bundle: dict[str, TransformationPlan] = {}
        for role in ("culprit", "rival"):
            slot_ids = self._reasoning_slot_ids_for_role(content_plan, role)
            plan = self._base_api_transformation_plan(
                trait_name=trait_name,
                variant="active",
                slot_ids=slot_ids,
            )
            slot_realizations = self._reasoning_slot_realizations_for_role(
                content_plan=content_plan,
                role=role,
                plan=plan,
            )
            metadata = dict(plan.branch_metadata or {})
            metadata["reasoning_slot_realizations"] = slot_realizations
            style_bundle[role] = plan.model_copy(update={"branch_metadata": metadata})
        self.validate_api_style_bundle(style_bundle, content_plan=content_plan, trait_name=trait_name)
        return style_bundle

    def generate_deterministic_role_style_bundle(
        self,
        *,
        content_plan: ContentPlan,
        trait_name: str,
    ) -> dict:
        if content_plan.content_plan_version != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            raise ValueError(
                f"Deterministic style-plan generation requires {PAIRWISE_ROLE_CONTENT_PLAN_VERSION}; "
                f"got {content_plan.content_plan_version!r}."
            )
        style_bundle: dict[str, TransformationPlan] = {}
        for role in ("culprit", "rival"):
            slot_ids = self._reasoning_slot_ids_for_role(content_plan, role)
            plan = self._base_role_transformation_plan(
                trait_name=trait_name,
                variant="active",
                slot_ids=slot_ids,
                style_plan_generation="deterministic",
            )
            contexts = self._reasoning_slot_contexts_for_role(content_plan, role)
            slot_realizations = self._deterministic_reasoning_payload(
                content_plan=content_plan,
                role=role,
                trait_name=trait_name,
                contexts=contexts,
            )
            metadata = dict(plan.branch_metadata or {})
            metadata["reasoning_slot_realizations"] = slot_realizations
            style_bundle[role] = plan.model_copy(update={"branch_metadata": metadata})
        self.validate_api_style_bundle(style_bundle, content_plan=content_plan, trait_name=trait_name)
        return style_bundle

    def generate_role_style_bundle(
        self,
        *,
        content_plan: ContentPlan,
        trait_name: str,
    ) -> dict:
        if self._is_fallacy_style_trait(trait_name):
            return self.generate_api_role_style_bundle(
                content_plan=content_plan,
                trait_name=trait_name,
            )
        return self.generate_deterministic_role_style_bundle(
            content_plan=content_plan,
            trait_name=self._style_plan_trait_name(trait_name),
        )

    def validate_api_style_bundle(
        self,
        style_bundle: dict,
        *,
        content_plan: ContentPlan | None = None,
        trait_name: str | None = None,
    ) -> None:
        if isinstance(style_bundle, dict) and "guidelines" in style_bundle:
            if style_bundle.get("style_plan_version") != ROLE_STYLE_PLAN_VERSION:
                raise ValueError(
                    f"Style plan must use version {ROLE_STYLE_PLAN_VERSION!r}; "
                    f"got {style_bundle.get('style_plan_version')!r}."
                )
            if style_bundle.get("style_plan_generation") not in {"deterministic", "api"}:
                raise ValueError("Style plan must declare style_plan_generation as deterministic or api.")
            expected_trait_name = self._style_plan_trait_name(trait_name) if trait_name is not None else None
            if expected_trait_name is not None and style_bundle.get("trait_name") != expected_trait_name:
                raise ValueError(
                    f"Style plan has trait {style_bundle.get('trait_name')!r}; expected {expected_trait_name!r}."
                )
            if self._is_fallacy_style_trait(style_bundle.get("trait_name")) and style_bundle.get("style_plan_generation") != "api":
                raise ValueError("Fallacy style plans must be API-generated.")
            if not self._is_fallacy_style_trait(style_bundle.get("trait_name")) and style_bundle.get("style_plan_generation") != "deterministic":
                raise ValueError("Non-fallacy style plans must be deterministic.")
            guidelines = style_bundle.get("guidelines")
            if not isinstance(guidelines, dict) or not guidelines:
                raise ValueError("Style plan must contain non-empty guidelines.")
            if content_plan is not None and content_plan.content_plan_version == PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
                for role in ("culprit", "rival"):
                    slot_ids = [slot_id for slot_id in guidelines if self._slot_role_from_id(slot_id) == role]
                    if content_plan is not None and content_plan.content_plan_version == PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
                        expected_slots = self._reasoning_slot_ids_for_role(content_plan, role)
                        missing = sorted(set(expected_slots) - set(slot_ids))
                        extra = sorted(set(slot_ids) - set(expected_slots))
                        if missing or extra:
                            raise ValueError(
                                f"style plan role {role} reasoning slots do not match content plan: "
                                f"missing={missing or []} extra={extra or []}"
                            )
            for slot_id, entry in guidelines.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"style plan guideline {slot_id} must be an object.")
                if set(entry) != {"baseline", "biased"}:
                    raise ValueError(f"style plan guideline {slot_id} must contain only baseline and biased.")
                for key in ("baseline", "biased"):
                    value = str(entry.get(key) or "").strip()
                    if not value:
                        raise ValueError(f"style plan guideline {slot_id}.{key} must be non-empty.")
            return

        if set(style_bundle) != {"culprit", "rival"}:
            raise ValueError("Style plan must contain exactly culprit and rival role plans.")
        expected_trait_name = self._style_plan_trait_name(trait_name) if trait_name is not None else None
        for role, plan in style_bundle.items():
            if expected_trait_name is not None and plan.trait_name != expected_trait_name:
                raise ValueError(f"Style plan role {role} has trait {plan.trait_name!r}; expected {expected_trait_name!r}.")
            metadata = dict(plan.branch_metadata or {})
            if metadata.get("style_plan_generation") not in {"deterministic", "api"}:
                raise ValueError(f"style plan role {role} is missing valid generation metadata.")
            if self._is_fallacy_style_trait(plan.trait_name) and metadata.get("style_plan_generation") != "api":
                raise ValueError(f"fallacy style plan role {role} must be API-generated.")
            if not self._is_fallacy_style_trait(plan.trait_name) and metadata.get("style_plan_generation") != "deterministic":
                raise ValueError(f"non-fallacy style plan role {role} must be deterministic.")
            realizations = metadata.get("reasoning_slot_realizations")
            if not isinstance(realizations, dict) or not realizations:
                raise ValueError(f"style plan role {role} lacks complete reasoning_slot_realizations.")
            expected_slots = (
                self._reasoning_slot_ids_for_role(content_plan, role)
                if content_plan is not None and content_plan.content_plan_version == PAIRWISE_ROLE_CONTENT_PLAN_VERSION
                else list(realizations)
            )
            missing = sorted(set(expected_slots) - set(realizations))
            extra = sorted(set(realizations) - set(expected_slots))
            if missing or extra:
                raise ValueError(
                    f"style plan role {role} reasoning slots do not match content plan: "
                    f"missing={missing or []} extra={extra or []}"
                )
            bias_keys = self._style_bias_keys(plan.trait_name)
            for slot_id in expected_slots:
                entry = realizations.get(slot_id)
                if not isinstance(entry, dict):
                    raise ValueError(f"style plan role {role} slot {slot_id} must be an object.")
                required_fields = {"local_purpose", "baseline_guideline", "biased_guideline"}
                missing_fields = sorted(field for field in required_fields if field not in entry)
                if missing_fields:
                    raise ValueError(f"style plan role {role} slot {slot_id} missing {missing_fields}.")
                if not str(entry.get("baseline_guideline") or "").strip():
                    raise ValueError(f"style plan role {role} slot {slot_id} has empty baseline_guideline.")
                biased = entry.get("biased_guideline")
                if not isinstance(biased, dict):
                    raise ValueError(f"style plan role {role} slot {slot_id} biased_guideline must be an object.")
                for bias_key in bias_keys:
                    if not str(biased.get(bias_key) or "").strip():
                        raise ValueError(f"style plan role {role} slot {slot_id} missing biased_guideline[{bias_key}].")

    def bind_style_bundle_to_content_plan(self, style_bundle: dict, content_plan: ContentPlan) -> dict:
        if content_plan.content_plan_version not in {"pairwise_reason_slots_v1", PAIRWISE_ROLE_CONTENT_PLAN_VERSION}:
            return style_bundle
        if content_plan.content_plan_version == PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            self.validate_api_style_bundle(
                style_bundle,
                content_plan=content_plan,
                trait_name=self._style_plan_trait_name(self.config["run"].get("trait_name")),
            )
            return style_bundle
        bound = {}
        for key, plan in style_bundle.items():
            if content_plan.content_plan_version == PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
                slot_ids = self._reasoning_slot_ids_for_role(content_plan, key)
                slot_realizations = self._reasoning_slot_realizations_for_role(
                    content_plan=content_plan,
                    role=key,
                    plan=plan,
                )
            else:
                slot_ids = self._reasoning_slot_ids_for_speaker(content_plan, key)
                slot_realizations = {}
            metadata = dict(getattr(plan, "branch_metadata", None) or {})
            if slot_realizations:
                metadata["reasoning_slot_realizations"] = slot_realizations
            if plan.trait_name.startswith("fallacy_trait__") and slot_ids:
                subtype = plan.trait_name.removeprefix("fallacy_trait__")
                metadata["target_reasoning_slot_ids"] = slot_ids
                metadata["hook_unit_ids"] = slot_ids
                metadata["fallacy_subtype"] = subtype
            bound[key] = plan.model_copy(update={
                "hook_unit_ids": slot_ids if plan.trait_name.startswith("fallacy_trait__") else plan.hook_unit_ids,
                "branch_metadata": metadata or plan.branch_metadata,
            })
        return bound

    def _style_bundle_guidelines(self, style_bundle: dict) -> dict[str, dict]:
        out: dict[str, dict] = {}
        trait_name = next(
            (
                getattr(plan, "trait_name", None)
                for plan in style_bundle.values()
                if getattr(plan, "trait_name", None)
            ),
            self._style_plan_trait_name(self.config["run"]["trait_name"]),
        )
        bias_keys = self._style_bias_keys(trait_name)
        for plan in style_bundle.values():
            metadata = dict(getattr(plan, "branch_metadata", None) or {})
            realizations = metadata.get("reasoning_slot_realizations") or {}
            if not isinstance(realizations, dict):
                continue
            for slot_id, entry in realizations.items():
                if isinstance(entry, dict):
                    biased_guidelines = entry.get("biased_guideline") or {}
                    if not isinstance(biased_guidelines, dict):
                        continue
                    biased = ""
                    for bias_key in bias_keys:
                        biased = str(biased_guidelines.get(bias_key) or "").strip()
                        if biased:
                            break
                    if not biased and biased_guidelines:
                        biased = str(next(iter(biased_guidelines.values())) or "").strip()
                    out[str(slot_id)] = {
                        "baseline": str(entry.get("baseline_guideline") or "").strip(),
                        "biased": biased,
                    }
        return out

    def _minimal_style_payload_from_bundle(self, style_bundle: dict) -> dict:
        guidelines = self._style_bundle_guidelines(style_bundle)
        generations = {
            (getattr(plan, "branch_metadata", None) or {}).get("style_plan_generation")
            for plan in style_bundle.values()
        }
        generations.discard(None)
        style_plan_generation = generations.pop() if len(generations) == 1 else "deterministic"
        return {
            "style_plan_version": ROLE_STYLE_PLAN_VERSION,
            "style_plan_generation": style_plan_generation,
            "case_id": self.config["run"]["case_id"],
            "trait_name": next(
                (
                    getattr(plan, "trait_name", None)
                    for plan in style_bundle.values()
                    if getattr(plan, "trait_name", None)
                ),
                self._style_plan_trait_name(self.config["run"]["trait_name"]),
            ),
            "guidelines": guidelines,
        }

    def _style_bundle_from_minimal_payload(self, payload: dict) -> dict[str, TransformationPlan]:
        trait_name = str(payload.get("trait_name") or "").strip()
        style_plan_generation = str(payload.get("style_plan_generation") or "deterministic").strip()
        guidelines = payload.get("guidelines") or {}
        bias_keys = self._style_bias_keys(trait_name)
        primary_bias_key = bias_keys[0] if bias_keys else trait_name
        out: dict[str, TransformationPlan] = {}
        for role in ("culprit", "rival"):
            slot_ids = [
                str(slot_id)
                for slot_id in guidelines
                if self._slot_role_from_id(str(slot_id)) == role
            ]
            plan = self._base_role_transformation_plan(
                trait_name=trait_name,
                variant="active",
                slot_ids=slot_ids,
                style_plan_generation=style_plan_generation,
            )
            realizations = {
                slot_id: {
                    "local_purpose": self._local_purpose_from_slot_id(slot_id),
                    "baseline_guideline": str(guidelines[slot_id].get("baseline") or "").strip(),
                    "biased_guideline": {
                        primary_bias_key: str(guidelines[slot_id].get("biased") or "").strip()
                    },
                }
                for slot_id in slot_ids
                if slot_id in guidelines
            }
            metadata = dict(plan.branch_metadata or {})
            metadata["reasoning_slot_realizations"] = realizations
            out[role] = plan.model_copy(update={"branch_metadata": metadata})
        return out

    @staticmethod
    def role_style_bundle_to_speaker_bundle(
        role_style_bundle: dict,
        *,
        ground_truth_side_order: str,
    ) -> dict:
        role_map = DetectivePipelineRunner.speaker_role_map_for_order(ground_truth_side_order)
        return {
            speaker: role_style_bundle[role]
            for speaker, role in role_map.items()
        }


    def save_style_bundle(self, style_bundle: dict, run_id: str) -> Path:
        topic_cfg = self.load_topic_cfg()
        if set(style_bundle) == {"culprit", "rival"}:
            self.validate_api_style_bundle(
                style_bundle,
                trait_name=self._style_plan_trait_name(self.config["run"].get("trait_name")),
            )
            safe_trait = self._style_plan_trait_name(self.config["run"]["trait_name"]).replace("/", "_")
            payload = self._minimal_style_payload_from_bundle(style_bundle)
            payload["content_plan_family"] = self.content_plan_family()
            payload["interpretation_set_id"] = self.interpretation_set_id()
            path = self.output_root() / "style_plans" / (
                f"{self.config['run']['case_id']}__{self.interpretation_artifact_suffix()}__{safe_trait}.json"
            )
        else:
            payload = {
                "case_id": self.config["run"]["case_id"],
                "content_plan_family": self.content_plan_family(),
                "interpretation_set_id": self.interpretation_set_id(),
                "trait_name": self.config["run"]["trait_name"],
                "variant_name_a": self.config["run"]["variant_name_a"],
                "variant_name_b": self.config["run"]["variant_name_b"],
                "ground_truth_side_order": self.ground_truth_side_order(),
                "agent_a_stance": topic_cfg["agent_a_stance"],
                "agent_b_stance": topic_cfg["agent_b_stance"],
                "transformation_plans": {
                    speaker: plan.model_dump()
                    for speaker, plan in style_bundle.items()
                },
            }
            path = self.output_root() / "style_plans" / f"{self.run_artifact_prefix(run_id)}_style_plans.json"
        dump_json(path, payload)
        return path

    def load_style_bundle_from_path(self, path: str | Path) -> dict:
        raw = json.loads(Path(path).read_text())
        if raw.get("style_plan_version") != ROLE_STYLE_PLAN_VERSION or raw.get("style_plan_generation") not in {"deterministic", "api"}:
            raise ValueError(
                f"Style plan {path} is not a complete role style plan "
                f"with version {ROLE_STYLE_PLAN_VERSION!r}."
            )
        if "guidelines" not in raw:
            raise ValueError(
                f"Style plan {path} uses an obsolete schema; expected minimal "
                "guidelines."
            )
        obsolete_fields = {
            "reasoning_slots",
            "role_slot_ids",
            "role_transformation_plans",
            "reasoning_slot_realizations",
            "branch_metadata",
            "transformation_plans",
            "style_plans",
        }
        present_obsolete = sorted(field for field in obsolete_fields if field in raw)
        if present_obsolete:
            raise ValueError(
                f"Style plan {path} contains obsolete duplicated fields: {present_obsolete}."
            )
        self.validate_api_style_bundle(raw, trait_name=raw.get("trait_name"))
        out = self._style_bundle_from_minimal_payload(raw)
        self.validate_api_style_bundle(out, trait_name=raw.get("trait_name"))
        print(f"[detective-pipeline] reusing validated {raw.get('style_plan_generation')} style plan: {path}")
        return out

    @staticmethod
    def style_bundle_for_role_variants(style_bundle: dict, role_variants: dict[str, str]) -> dict:
        out = {}
        for role, plan in style_bundle.items():
            variant = role_variants.get(role, plan.variant)
            metadata = dict(plan.branch_metadata or {})
            structure = resolve_dialogue_structure(
                trait_name=plan.trait_name,
                variant_name=variant,
            )
            metadata["dialogue_structure_id"] = structure.structure_id
            out[role] = plan.model_copy(update={"variant": variant, "branch_metadata": metadata})
        return out

    def fixed_speakers_for_baseline_reuse(self) -> set[str]:
        if self.config["run"].get("trait_name") == "pro_jargon_bias":
            return set()
        variant_name_a = self.config["run"]["variant_name_a"]
        variant_name_b = self.config["run"]["variant_name_b"]
        if variant_name_a == "active" and variant_name_b == "baseline":
            return {"B"}
        if variant_name_a == "baseline" and variant_name_b == "active":
            return {"A"}
        return set()

    def fixed_turn_ids_for_baseline_reuse(self) -> set[int]:
        if self.config["run"].get("trait_name") == "anchoring_bias":
            return set()
        if self.config["run"].get("trait_name") == "pro_jargon_bias":
            variant_name_a = self.config["run"]["variant_name_a"]
            variant_name_b = self.config["run"]["variant_name_b"]
            if variant_name_a == "active" and variant_name_b == "baseline":
                return {2, 4, 6}
            if variant_name_a == "baseline" and variant_name_b == "active":
                return {1, 3, 5}
            return set()
        if int(self.config["run"].get("turns_per_agent", 4)) == 3:
            return {1, 2} if self.fixed_speakers_for_baseline_reuse() else set()
        if self.fixed_speakers_for_baseline_reuse():
            return {1, 2, 7, 8}
        return set()

    def baseline_verbosity_gap_report(self, dialogue: Dialogue, speaker_variants: dict[str, str]) -> dict | None:
        if not all(variant == "baseline" for variant in speaker_variants.values()):
            return None
        return build_verbosity_gap_report(dialogue, speaker_variants)

    def load_reuse_baseline_dialogue(self) -> tuple[Dialogue | None, set[str], set[int], Path | None]:
        reuse_baseline_dialogue_path = self.config["run"].get("reuse_baseline_dialogue_path")
        fixed_speakers = self.fixed_speakers_for_baseline_reuse()
        fixed_turn_ids = self.fixed_turn_ids_for_baseline_reuse()
        if not reuse_baseline_dialogue_path:
            return None, set(), set(), None
        if not fixed_speakers and not fixed_turn_ids:
            print(
                "[detective-pipeline] reuse_baseline_dialogue_path provided, "
                "but the variant pair is not anchored; generating normally."
            )
            return None, set(), set(), None
        path = self._resolve_repo_path(reuse_baseline_dialogue_path)
        dialogue = self.load_dialogue_from_path(path)
        print(
            f"[detective-pipeline] Reusing fixed baseline turns from {path} "
            f"for speaker(s) {', '.join(sorted(fixed_speakers)) or 'none'} "
            f"and turn_id(s) {', '.join(str(i) for i in sorted(fixed_turn_ids)) or 'none'}"
        )
        return dialogue, fixed_speakers, fixed_turn_ids, path

    def adaptive_treatment_metadata(
        self,
        *,
        fixed_dialogue: Dialogue | None,
        speaker_variants: dict[str, str],
        topic_cfg: dict | None = None,
    ) -> dict:
        return self.rollout._adaptive_treatment_metadata(
            trait_name=self.config["run"]["trait_name"],
            speaker_variants=speaker_variants,
            fixed_dialogue=fixed_dialogue,
            topic_cfg=topic_cfg,
        )

    def generate_dialogue(
        self,
        topic_cfg: dict,
        content_plan: ContentPlan,
        style_bundle: dict,
        trait_name: str,
        speaker_variants: dict[str, str],
        eval_feedback: str | None = None,
        fixed_dialogue: Dialogue | None = None,
        fixed_speakers: set[str] | None = None,
        fixed_turn_ids: set[int] | None = None,
        cached_opening_turns: dict[int, DialogueTurn] | None = None,
        treatment_metadata: dict | None = None,
    ) -> Dialogue:
        if self.client is None:
            raise ValueError("OPENROUTER_API_KEY is required for dialogue generation.")
        validate_trait_speaker_variants(trait_name, speaker_variants)
        fixed_speakers = set(fixed_speakers or set())
        fixed_turn_ids = set(fixed_turn_ids or set())
        if fixed_dialogue is not None and (fixed_speakers or fixed_turn_ids):
            return self.rollout.generate_dialogue_with_fixed_turns(
                topic_cfg=topic_cfg,
                content_plan=content_plan,
                style_bundle=style_bundle,
                trait_name=trait_name,
                speaker_variants=speaker_variants,
                fixed_dialogue=fixed_dialogue,
                fixed_speakers=fixed_speakers,
                fixed_turn_ids=fixed_turn_ids,
                cached_opening_turns=cached_opening_turns,
                treatment_metadata=treatment_metadata,
                eval_feedback=eval_feedback,
            )
        return self.rollout.generate_dialogue(
            topic_cfg=topic_cfg,
            content_plan=content_plan,
            style_bundle=style_bundle,
            trait_name=trait_name,
            speaker_variants=speaker_variants,
            eval_feedback=eval_feedback,
            cached_opening_turns=cached_opening_turns,
        )

    def save_dialogue(
        self,
        dialogue: Dialogue,
        run_id: str,
        attempt: int | None = None,
        extra_metadata: dict | None = None,
    ) -> Path:
        role_variants = self.role_variants()
        seed = self.config["run"].get("seed")
        prefix = self.run_artifact_prefix(run_id)
        filename = f"{prefix}_dialogue.json" if attempt is None else f"{prefix}_attempt{attempt}_dialogue.json"
        path = self.output_root() / "dialogues" / filename
        payload = dialogue.model_dump()
        payload["case_id"] = self.config["run"]["case_id"]
        payload["content_plan_family"] = self.content_plan_family()
        payload["interpretation_set_id"] = self.interpretation_set_id()
        payload["trait_name"] = self.config["run"].get("trait_name")
        payload["culprit_variant"] = role_variants.get("culprit")
        payload["rival_variant"] = role_variants.get("rival")
        payload["ground_truth_side_order"] = self.ground_truth_side_order()
        payload["variant_name_a"] = self.config["run"].get("variant_name_a")
        payload["variant_name_b"] = self.config["run"].get("variant_name_b")
        payload["rollout_index"] = self.rollout_index()
        payload["seed"] = seed
        if extra_metadata:
            payload.update(extra_metadata)
        dump_json(path, payload)
        return path

    def load_dialogue_from_path(self, path: str | Path) -> Dialogue:
        resolved = self._resolve_repo_path(path)
        return Dialogue(**json.loads(resolved.read_text()))

    def evaluate_computational_verbosity(
        self,
        dialogue: Dialogue,
        trait_name: str,
        speaker_variants: dict[str, str],
    ) -> dict:
        return self.evaluator.evaluate_computational_verbosity(
            dialogue=dialogue,
            speaker_variants=speaker_variants,
            trait_name=trait_name,
        )

    def evaluate_bias(
        self,
        dialogue: Dialogue,
        content_plan: ContentPlan,
        topic_cfg: dict,
        trait_name: str,
        speaker_variants: dict[str, str],
        include_computational_verbosity: bool = True,
    ):
        return self.evaluator.evaluate_evidence_debate_v2(
            dialogue=dialogue,
            content_plan=content_plan,
            evidence_bank=topic_cfg["evidence_bank"],
            trait_name=trait_name,
            speaker_variants=speaker_variants,
            include_computational_verbosity=include_computational_verbosity,
        )

    def save_bias_eval(self, eval_result, run_id: str, attempt: int | None = None) -> Path:
        if eval_result is None:
            raise ValueError("cannot save an empty bias eval result")
        prefix = self.run_artifact_prefix(run_id)
        filename = f"{prefix}_eval.json" if attempt is None else f"{prefix}_attempt{attempt}_eval.json"
        path = self.output_root() / "evals" / filename
        topic_cfg = self.load_topic_cfg()
        payload = eval_result.model_dump()
        payload.update({
            "case_id": self.config["run"]["case_id"],
            "content_plan_family": self.content_plan_family(),
            "interpretation_set_id": self.interpretation_set_id(),
            "trait_name": self.config["run"]["trait_name"],
            "variant_name_a": self.config["run"]["variant_name_a"],
            "variant_name_b": self.config["run"]["variant_name_b"],
            "ground_truth_side_order": self.ground_truth_side_order(),
            "rollout_index": self.rollout_index(),
            "agent_a_stance": topic_cfg["agent_a_stance"],
            "agent_b_stance": topic_cfg["agent_b_stance"],
            "topic": topic_cfg["topic"],
        })
        dump_json(path, payload)
        return path

    def evaluate_rar(self, dialogue: Dialogue) -> dict:
        if self.rar_evaluator is None:
            raise ValueError("RaR evaluation is not enabled in config.")
        return self.rar_evaluator.evaluate_debate_rubric_quality(dialogue)

    def save_rar_eval(self, rar_result: dict, run_id: str) -> Path:
        path = self.output_root() / "rar_evals" / f"{self.run_artifact_prefix(run_id)}_rar_eval.json"
        topic_cfg = self.load_topic_cfg()
        payload = dict(rar_result)
        payload.update({
            "case_id": self.config["run"]["case_id"],
            "content_plan_family": self.content_plan_family(),
            "interpretation_set_id": self.interpretation_set_id(),
            "trait_name": self.config["run"].get("trait_name"),
            "variant_name_a": self.config["run"]["variant_name_a"],
            "variant_name_b": self.config["run"]["variant_name_b"],
            "ground_truth_side_order": self.ground_truth_side_order(),
            "rollout_index": self.rollout_index(),
            "agent_a_stance": topic_cfg["agent_a_stance"],
            "agent_b_stance": topic_cfg["agent_b_stance"],
        })
        dump_json(path, payload)
        return path

    def save_accepted_dialogue(
        self,
        *,
        dialogue: Dialogue,
        content_plan: ContentPlan,
        topic_cfg: dict,
        plan_source: str,
        dialogue_metadata: dict,
        output_root: Path,
        run_id: str,
    ) -> Path:
        accepted_payload = dialogue.model_dump()
        accepted_payload["content_plan_source"] = plan_source
        accepted_payload["case_meta"] = {
            "case_id": topic_cfg["case_id"],
            "case_name": topic_cfg["case_name"],
            "motion": topic_cfg["topic"],
            "question": topic_cfg["question"],
            "correct_answer": topic_cfg["correct_answer"],
            "rival_suspect": topic_cfg.get("rival_suspect"),
            "pairwise_suspect_pair": topic_cfg.get("pairwise_suspect_pair"),
            "solve_rate": topic_cfg["solve_rate"],
            "evidence_count": len(topic_cfg["evidence_bank"]),
            "ground_truth_side_order": topic_cfg["ground_truth_side_order"],
            "ground_truth_supporting_speaker": topic_cfg["ground_truth_supporting_speaker"],
            "ground_truth_opposing_speaker": topic_cfg["ground_truth_opposing_speaker"],
            "agent_a_stance": topic_cfg["agent_a_stance"],
            "agent_b_stance": topic_cfg["agent_b_stance"],
        }
        accepted_payload["evidence_bank"] = topic_cfg["evidence_bank"]
        accepted_payload["case_context"] = topic_cfg.get("case_context")
        accepted_payload.update(self.content_plan_dialogue_metadata(content_plan))
        accepted_payload.update(dialogue_metadata)
        accepted_payload["case_id"] = topic_cfg["case_id"]
        accepted_payload["content_plan_family"] = content_plan.content_plan_family or self.content_plan_family()
        accepted_payload["interpretation_set_id"] = content_plan.interpretation_set_id or self.interpretation_set_id()
        accepted_payload["trait_name"] = self.config["run"].get("trait_name")
        accepted_payload["variant_name_a"] = self.config["run"].get("variant_name_a")
        accepted_payload["variant_name_b"] = self.config["run"].get("variant_name_b")
        accepted_payload["ground_truth_side_order"] = self.ground_truth_side_order()
        accepted_payload["rollout_index"] = self.rollout_index()
        accepted_dialogue_path = output_root / "accepted" / f"{self.run_artifact_prefix(run_id)}_accepted_dialogue.json"
        dump_json(accepted_dialogue_path, accepted_payload)
        return accepted_dialogue_path

    def save_verbosity_retry_trace(self, run_id: str, payload: dict) -> Path:
        path = self.output_root() / "evals" / "verbosity_retries" / f"{self.run_artifact_prefix(run_id)}_verbosity_retry.json"
        dump_json(path, payload)
        return path

    def run(self) -> dict:
        run_id = make_run_id()
        case_id = self.config["run"]["case_id"]
        trait_name = self.config["run"]["trait_name"]
        variant_name_a = self.config["run"]["variant_name_a"]
        variant_name_b = self.config["run"]["variant_name_b"]
        ground_truth_side_order = self.ground_truth_side_order()
        speaker_variants = self.speaker_variants()

        print(f"[{run_id}] Starting detective pipeline — case: {case_id}")
        print(f"[{run_id}] Trait: {trait_name}, A: {variant_name_a}, B: {variant_name_b}")
        print(f"[{run_id}] Content plan: family={self.content_plan_family()} interpretation_set={self.interpretation_set_id()}")

        topic_cfg = self.load_topic_cfg()
        output_root = self.output_root()

        print(
            f"[{run_id}] GT side order: {ground_truth_side_order} "
            f"(A: {topic_cfg['agent_a_stance']}; B: {topic_cfg['agent_b_stance']})"
        )
        print(f"[{run_id}] Motion: {topic_cfg['topic']}")
        print(f"[{run_id}] Evidence items: {len(topic_cfg['evidence_bank'])}")

        reuse_plan_path = self.config["run"].get("reuse_plan_path")
        if reuse_plan_path:
            plan_path = self._resolve_repo_path(reuse_plan_path)
            print(f"[{run_id}] Reusing content plan from: {plan_path}")
            content_plan = self.load_content_plan_from_path(plan_path)
            self.validate_content_plan_pair(content_plan, topic_cfg)
            plan_source = plan_path.name
        else:
            print(f"[{run_id}] Generating pairwise role-based content plan...")
            content_plan, plan_path = self.generate_content_plan(
                topic_cfg,
                run_id,
            )
            self.validate_content_plan_pair(content_plan, topic_cfg)
            plan_source = plan_path.name
            print(f"[{run_id}] Content plan saved.")

        reuse_baseline_dialogue, fixed_speakers, fixed_turn_ids, baseline_dialogue_source = self.load_reuse_baseline_dialogue()
        dialogue_extra_metadata = self.content_plan_dialogue_metadata(content_plan)
        if baseline_dialogue_source is not None:
            baseline_source = self.repo_display_path(baseline_dialogue_source)
            dialogue_extra_metadata["baseline_dialogue_source"] = baseline_source
            dialogue_extra_metadata.update(self.canonical_baseline_source_metadata(baseline_source))
        if fixed_speakers:
            dialogue_extra_metadata["fixed_speakers"] = sorted(fixed_speakers)
        if fixed_turn_ids:
            dialogue_extra_metadata["fixed_turn_ids"] = sorted(fixed_turn_ids)
        treatment_metadata = self.adaptive_treatment_metadata(
            fixed_dialogue=reuse_baseline_dialogue,
            speaker_variants=speaker_variants,
            topic_cfg=topic_cfg,
        )
        dialogue_extra_metadata.update(treatment_metadata)

        if content_plan.content_plan_version != PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            raise ValueError(
                f"Role style-plan generation requires {PAIRWISE_ROLE_CONTENT_PLAN_VERSION}; "
                f"got {content_plan.content_plan_version!r}."
            )
        role_variants = self.role_variants()
        style_generation = "api" if self._is_fallacy_style_trait(trait_name) else "deterministic"
        reuse_style_plan_path = self.config["run"].get("reuse_style_plan_path")
        if reuse_style_plan_path:
            style_path = self._resolve_repo_path(reuse_style_plan_path)
            print(f"[{run_id}] Reusing style plan from: {style_path}")
            role_style_bundle = self.load_style_bundle_from_path(style_path)
            self.validate_api_style_bundle(role_style_bundle, content_plan=content_plan, trait_name=trait_name)
        else:
            print(f"[{run_id}] Generating {style_generation} style plan...")
            role_style_bundle = self.generate_role_style_bundle(
                content_plan=content_plan,
                trait_name=trait_name,
            )
            style_path = self.save_style_bundle(role_style_bundle, run_id)
            print(f"[{run_id}] {style_generation.capitalize()} style plan saved to: {style_path}")
        runtime_style_bundle = self.style_bundle_for_role_variants(role_style_bundle, role_variants)
        style_bundle = self.role_style_bundle_to_speaker_bundle(
            runtime_style_bundle,
            ground_truth_side_order=ground_truth_side_order,
        )
        dialogue_extra_metadata["style_plan_source"] = self.repo_display_path(style_path)

        accepted = None
        last_eval = None
        eval_feedback = None
        configured_max_attempts = int(self.config["run"]["max_attempts"])
        max_attempts = (
            min(configured_max_attempts, MAX_VERBOSITY_REGEN_ATTEMPTS)
            if trait_name == "verbosity_bias"
            else configured_max_attempts
        )
        last_dialogue_metadata = dict(dialogue_extra_metadata)
        verbosity_retry_attempts: list[dict] = []
        verbosity_retry_trace_path = None
        final_selected_dialogue_path = None
        final_selected_eval_path = None

        for attempt in range(1, max_attempts + 1):
            print(f"[{run_id}] Attempt {attempt}/{max_attempts}: Generating dialogue...")
            try:
                dialogue = self.generate_dialogue(
                    topic_cfg=topic_cfg,
                    content_plan=content_plan,
                    style_bundle=style_bundle,
                    trait_name=trait_name,
                    speaker_variants=speaker_variants,
                    eval_feedback=eval_feedback,
                    fixed_dialogue=reuse_baseline_dialogue,
                    fixed_speakers=fixed_speakers,
                    fixed_turn_ids=fixed_turn_ids,
                    treatment_metadata=treatment_metadata,
                )
            except AdaptiveTurnGenerationError as exc:
                eval_feedback = str(exc)
                print(
                    f"[{run_id}] Attempt {attempt}/{max_attempts}: dialogue generation failed. "
                    f"{exc}"
                )
                continue
            attempt_dialogue_metadata = dict(dialogue_extra_metadata)
            attempt_dialogue_metadata.update(self.dialogue_runtime_metadata())
            attempt_dialogue_path = self.save_dialogue(
                dialogue,
                run_id,
                attempt=attempt,
                extra_metadata=attempt_dialogue_metadata,
            )
            last_dialogue_metadata = attempt_dialogue_metadata
            if skip_bias_eval_for_trait(trait_name):
                print(
                    f"[{run_id}] Attempt {attempt}/{max_attempts}: "
                    f"Skipping bias evaluation for {trait_name}; accepting dialogue after generation checks."
                )
                accepted = dialogue
                accepted_dialogue_path = self.save_accepted_dialogue(
                    dialogue=dialogue,
                    content_plan=content_plan,
                    topic_cfg=topic_cfg,
                    plan_source=plan_source,
                    dialogue_metadata=last_dialogue_metadata,
                    output_root=output_root,
                    run_id=run_id,
                )
                final_selected_dialogue_path = accepted_dialogue_path
                final_selected_eval_path = None
                print(f"[{run_id}] Dialogue accepted and saved!")
                print(f"[{run_id}] Bias evaluation skipped for {trait_name}.")

                rar_cfg = self.config.get("rar_evaluation", {})
                if self.rar_evaluator is not None:
                    print(f"[{run_id}] Running RaR debate-rubric evaluation...")
                    rar_result = self.evaluate_rar(dialogue)
                    if rar_cfg.get("save_outputs", True):
                        self.save_rar_eval(rar_result, run_id)
                        print(f"[{run_id}] RaR evaluation saved.")
                    if rar_cfg.get("fail_pipeline_on_rar_failure", False) and not rar_result.get("passed", True):
                        print(
                            f"[{run_id}] RaR evaluation failed cross-model verification; "
                            "evaluation no longer controls dialogue acceptance."
                        )
                break

            print(f"[{run_id}] Attempt {attempt}/{max_attempts}: Dialogue saved. Evaluating...")

            eval_result = self.evaluate_bias(dialogue, content_plan, topic_cfg, trait_name, speaker_variants)
            last_eval = eval_result
            attempt_eval_path = self.save_bias_eval(eval_result, run_id, attempt=attempt)
            if getattr(eval_result, "evaluation_type", "llm_semantic") == "deterministic":
                details = eval_result.deterministic_details or {}
                print(
                    f"[{run_id}] Attempt {attempt}/{max_attempts}: "
                    f"deterministic verbosity eval turns={eval_result.passed_turns}/{eval_result.total_turns} "
                    f"gap={details.get('gap')} threshold={details.get('threshold')} "
                    f"measure={details.get('measure')} passed={eval_result.passed} "
                    f"failure_modes={eval_result.failure_modes}"
                )
            else:
                print(
                    f"[{run_id}] Attempt {attempt}/{max_attempts}: "
                    f"turns={eval_result.passed_turns}/{eval_result.total_turns} "
                    f"A={eval_result.side_summaries[0].passed_turns}/{eval_result.side_summaries[0].total_turns} "
                    f"B={eval_result.side_summaries[1].passed_turns}/{eval_result.side_summaries[1].total_turns} "
                    f"models_with_all_turns_passing={eval_result.models_with_all_turns_passing}/{len(eval_result.per_model_results)} "
                    f"passed={eval_result.passed} "
                    f"failure_modes={eval_result.failure_modes}"
                )

            if trait_name == "verbosity_bias":
                verbosity_retry_attempts.append({
                    "attempt": attempt,
                    "bias_eval_passed": self.sampler.should_accept(eval_result),
                    "evaluator_feedback": getattr(eval_result, "reason", None),
                    "dialogue_path": str(attempt_dialogue_path),
                    "eval_path": str(attempt_eval_path),
                    "dialogue_id": run_id,
                    "selected_as_final": False,
                })
                final_selected_dialogue_path = attempt_dialogue_path
                final_selected_eval_path = attempt_eval_path
                verbosity_retry_trace_path = self.save_verbosity_retry_trace(
                    run_id,
                    {
                        "run_id": run_id,
                        "case_id": case_id,
                        "trait_name": trait_name,
                        "variant_name_a": variant_name_a,
                        "variant_name_b": variant_name_b,
                        "ground_truth_side_order": ground_truth_side_order,
                        "rollout_index": self.rollout_index(),
                        "max_attempts": max_attempts,
                        "attempts": verbosity_retry_attempts,
                        "final_selected_dialogue": str(final_selected_dialogue_path),
                        "final_selected_eval": str(final_selected_eval_path),
                        "accepted": False,
                    },
                )

            if trait_name == "verbosity_bias":
                print(
                    f"[verbosity-eval] attempt {attempt} saved; "
                    "evaluation no longer controls dialogue acceptance."
                )
            elif not self.sampler.should_accept(eval_result):
                print(
                    f"[{run_id}] Attempt {attempt}/{max_attempts}: "
                    "bias evaluation did not pass, but evaluation no longer controls dialogue acceptance."
                )

            accepted = dialogue
            accepted_dialogue_path = self.save_accepted_dialogue(
                dialogue=dialogue,
                content_plan=content_plan,
                topic_cfg=topic_cfg,
                plan_source=plan_source,
                dialogue_metadata=last_dialogue_metadata,
                output_root=output_root,
                run_id=run_id,
            )
            accepted_eval_path = self.save_bias_eval(eval_result, run_id)
            final_selected_dialogue_path = accepted_dialogue_path
            final_selected_eval_path = accepted_eval_path
            if trait_name == "verbosity_bias":
                verbosity_retry_attempts[-1]["selected_as_final"] = True
                verbosity_retry_trace_path = self.save_verbosity_retry_trace(
                    run_id,
                    {
                        "run_id": run_id,
                        "case_id": case_id,
                        "trait_name": trait_name,
                        "variant_name_a": variant_name_a,
                        "variant_name_b": variant_name_b,
                        "ground_truth_side_order": ground_truth_side_order,
                        "rollout_index": self.rollout_index(),
                        "max_attempts": max_attempts,
                        "attempts": verbosity_retry_attempts,
                        "final_selected_dialogue": str(final_selected_dialogue_path),
                        "final_selected_eval": str(final_selected_eval_path),
                        "accepted": True,
                    },
                )
            print(f"[{run_id}] Dialogue accepted and saved!")
            print(f"[{run_id}] Bias evaluation saved to {accepted_eval_path}")

            rar_cfg = self.config.get("rar_evaluation", {})
            if self.rar_evaluator is not None:
                print(f"[{run_id}] Running RaR debate-rubric evaluation...")
                rar_result = self.evaluate_rar(dialogue)
                if rar_cfg.get("save_outputs", True):
                    self.save_rar_eval(rar_result, run_id)
                    print(f"[{run_id}] RaR evaluation saved.")
                if rar_cfg.get("fail_pipeline_on_rar_failure", False) and not rar_result.get("passed", True):
                    print(
                        f"[{run_id}] RaR evaluation failed cross-model verification; "
                        "evaluation no longer controls dialogue acceptance."
                    )
            break

        if accepted is None:
            if trait_name == "verbosity_bias" and verbosity_retry_attempts:
                verbosity_retry_attempts[-1]["selected_as_final"] = True
                verbosity_retry_trace_path = self.save_verbosity_retry_trace(
                    run_id,
                    {
                        "run_id": run_id,
                        "case_id": case_id,
                        "trait_name": trait_name,
                        "variant_name_a": variant_name_a,
                        "variant_name_b": variant_name_b,
                        "ground_truth_side_order": ground_truth_side_order,
                        "rollout_index": self.rollout_index(),
                        "max_attempts": max_attempts,
                        "attempts": verbosity_retry_attempts,
                        "final_selected_dialogue": str(final_selected_dialogue_path) if final_selected_dialogue_path else None,
                        "final_selected_eval": str(final_selected_eval_path) if final_selected_eval_path else None,
                        "accepted": False,
                    },
                )
            print(f"[{run_id}] No dialogue accepted after {max_attempts} attempts.")
        print(f"[{run_id}] Detective pipeline run complete.")

        return {
            "run_id": run_id,
            "case_id": case_id,
            "accepted": accepted is not None,
            "last_eval": None if last_eval is None else last_eval.model_dump(),
            "output_root": self.repo_display_path(output_root),
            "plan_path": self.repo_display_path(plan_path),
            "plan_reused": bool(reuse_plan_path),
            "style_path": self.repo_display_path(style_path),
            "style_reused": bool(reuse_style_plan_path),
            "baseline_dialogue_reused": baseline_dialogue_source is not None,
            "fixed_speakers": sorted(fixed_speakers),
            "fixed_turn_ids": sorted(fixed_turn_ids),
            "verbosity_retry_trace_path": (
                None if verbosity_retry_trace_path is None else self.repo_display_path(verbosity_retry_trace_path)
            ),
        }
