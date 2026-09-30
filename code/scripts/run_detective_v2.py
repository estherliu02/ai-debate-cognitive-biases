"""Batch and stepwise detective data generation using code/configs/traitsV2.py.

Supports:
- full sweep / smoke-test execution
- rerunning individual pipeline steps against prior outputs
- metadata-based reference file lookup under outputs/<run_dir>/<subdir>/

python code/scripts/run_detective_v2.py \
  --step style-plan \
  --case our-quarterback-is-missing \
  --trait anchoring_bias \
  --variant-a active \
  --variant-b baseline \
  --gt-order gt_second \
  --reference-output 0509_detective_v2

"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime
from functools import cache
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import core.evaluator as core_evaluator
import core.pipeline_detective as pipeline_detective
from configs import traitsV2
from configs.detective_cases import (
    GROUND_TRUTH_SIDE_ORDER_CHOICES,
    load_case,
)
from core.reasoning_finder_detective import (
    standard_interpretation_set_ids,
    verbosity_interpretation_set_ids,
)
from utils.ids import make_run_id


CONFIG_PATH = CODE_ROOT / "configs" / "generation_detective.yaml"

CASE_SPECS = [
    ("The Locker Incident", "our-quarterback-is-missing"),
    ("The Diamond Necklace", "the-diamond-necklace"),
    ("The Missing Briefcase", "the-missing-briefcase"),
    ("The Missing Trophy", "the-mystery-of-the-leprechaun-s-trophy"),
]

ABLATIONS = [
    ("baseline", "baseline"),
    ("active", "baseline"),
    ("baseline", "active"),
]
ROLE_BASELINE_PAIR = ("baseline", "baseline")
ROLE_BIAS_PAIRS = [
    ("active", "baseline"),
    ("baseline", "active"),
]

STEP_CHOICES = ("content-plan", "attention-question", "style-plan", "role-dialogue", "dialogue", "bias-eval", "rar-eval", "full")
CASE_LABEL_BY_ID = {slug: label for label, slug in CASE_SPECS}
BASELINE_CONTROL_TRAIT_NAME = "N/A"
BASELINE_CONTROL_KEY = "__baseline_control__"
BASELINE_STYLE_PLAN_TRAITS = {"pro_jargon_bias"}
CANONICAL_BASELINE_CONTENT_PLAN_FAMILY = pipeline_detective.CANONICAL_BASELINE_CONTENT_PLAN_FAMILY
CANONICAL_BASELINE_INTERPRETATION_SET_ID = pipeline_detective.CANONICAL_BASELINE_INTERPRETATION_SET_ID
CANONICAL_BASELINE_ROLLOUT_INDEX = pipeline_detective.CANONICAL_BASELINE_ROLLOUT_INDEX
StepRun = tuple[str, str, str, str, str, str, str, str, int]
ATTEMPT_OUTPUT_RE = re.compile(r"^(?P<prefix>.+)_attempt\d+_(?P<kind>dialogue|eval)\.json$")
FAILURE_LOG_SUBDIR = "failures"
DialogueFailureRecord = dict[str, object]


def dialogue_generation_failure_log_path(output_root: Path) -> Path:
    return output_root / FAILURE_LOG_SUBDIR / "dialogue_generation_failures.jsonl"


def record_dialogue_generation_failure(
    *,
    output_root: Path,
    case_label: str,
    case_id: str,
    trait_name: str,
    culprit_variant: str,
    rival_variant: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int,
    seed: int,
    step_index: int,
    step_total: int,
    exc: BaseException,
) -> tuple[DialogueFailureRecord, Path]:
    failure_log = dialogue_generation_failure_log_path(output_root)
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    error_message = str(exc)
    record: DialogueFailureRecord = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "case_label": case_label,
        "case_id": case_id,
        "trait_name": trait_name,
        "culprit_variant": culprit_variant,
        "rival_variant": rival_variant,
        "ground_truth_side_order": ground_truth_side_order,
        "content_plan_family": content_plan_family,
        "interpretation_set_id": interpretation_set_id,
        "rollout_index": rollout_index,
        "seed": seed,
        "step_index": step_index,
        "step_total": step_total,
        "exception_type": type(exc).__name__,
        "error_message": error_message,
        "error": f"{type(exc).__name__}: {error_message}",
    }
    with failure_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record, failure_log


def selected_ground_truth_side_orders(
    ground_truth_side_order: str | None,
) -> list[str]:
    if ground_truth_side_order is not None:
        return [ground_truth_side_order]
    return list(GROUND_TRUTH_SIDE_ORDER_CHOICES)


def interpretation_sets_for_case(case_id: str, family: str) -> list[str]:
    cfg = load_case(case_id)
    finding = cfg["reasoning_finding"]
    if family == "standard":
        return [
            interpretation_id
            for interpretation_id in standard_interpretation_set_ids(finding)
            if re.fullmatch(r"primary-\d+", interpretation_id)
        ]
    if family == "verbosity":
        return verbosity_interpretation_set_ids(finding)
    raise ValueError(f"Unknown content_plan_family {family!r}.")


def content_plan_family_for_trait(trait_name: str | None) -> str:
    return "verbosity" if trait_name == "verbosity_bias" else "standard"


def selected_interpretation_targets(
    case_id: str,
    trait_name: str | None,
    interpretation_set: str | None,
    *,
    strict: bool = True,
) -> list[tuple[str, str]]:
    family = content_plan_family_for_trait(trait_name)
    ids = interpretation_sets_for_case(case_id, family)
    targets = [(family, interpretation_id) for interpretation_id in ids]
    if interpretation_set is None:
        return targets
    filtered = [target for target in targets if target[1] == interpretation_set]
    if not filtered:
        if not strict:
            return []
        raise ValueError(
            f"Unknown interpretation set {interpretation_set!r} for case={case_id} "
            f"family={family}. Expected one of: {', '.join(ids)}"
        )
    return filtered


def canonical_baseline_identity() -> tuple[str, str, int]:
    return (
        CANONICAL_BASELINE_CONTENT_PLAN_FAMILY,
        CANONICAL_BASELINE_INTERPRETATION_SET_ID,
        CANONICAL_BASELINE_ROLLOUT_INDEX,
    )


def is_baseline_variant_pair(variant_a: str, variant_b: str) -> bool:
    return variant_a == "baseline" and variant_b == "baseline"


def rollout_indices(rollouts_per_combo: int) -> range:
    if rollouts_per_combo < 1:
        raise ValueError("--rollouts-per-combo must be >= 1.")
    return range(1, rollouts_per_combo + 1)


def payload_rollout_index(payload: dict) -> int:
    try:
        return int(payload.get("rollout_index") or 1)
    except (TypeError, ValueError):
        return 1


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def canonical_ground_truth_side_order_for_variants(
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
) -> str:
    del trait_name, variant_a, variant_b
    return ground_truth_side_order


def is_anchoring_trait(trait_name: str | None) -> bool:
    return trait_name == "anchoring_bias"


def anchoring_speaker_variants_allowed(
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
) -> bool:
    del ground_truth_side_order
    if not is_anchoring_trait(trait_name):
        return True
    if (variant_a, variant_b) == ROLE_BASELINE_PAIR:
        return True
    return (variant_a, variant_b) in {("active", "baseline"), ("baseline", "active")}


def anchoring_role_variants_allowed(
    trait_name: str,
    culprit_variant: str,
    rival_variant: str,
    ground_truth_side_order: str,
) -> bool:
    if not is_anchoring_trait(trait_name):
        return True
    if (culprit_variant, rival_variant) == ROLE_BASELINE_PAIR:
        return True
    speaker_variants = speaker_variants_from_role_variants(
        culprit_variant,
        rival_variant,
        ground_truth_side_order,
    )
    return anchoring_speaker_variants_allowed(
        trait_name,
        speaker_variants["A"],
        speaker_variants["B"],
        ground_truth_side_order,
    )


def anchoring_invalid_request_message(
    *,
    trait_name: str,
    variant_a: str | None = None,
    variant_b: str | None = None,
    culprit_variant: str | None = None,
    rival_variant: str | None = None,
    ground_truth_side_order: str | None,
) -> str:
    order_desc = ground_truth_side_order or "default gt_first,gt_second"
    if culprit_variant is not None or rival_variant is not None:
        return (
            "Invalid anchoring_bias schedule: anchoring can only use one active speaker. "
            f"Requested culprit={culprit_variant} rival={rival_variant} order={order_desc}. "
            "Valid role conditions map to A=active/B=baseline or A=baseline/B=active."
        )
    return (
        "Invalid anchoring_bias schedule: anchoring can only use one active speaker. "
        f"Requested A={variant_a} B={variant_b} order={order_desc}. "
        "Valid speaker conditions are A=active/B=baseline or A=baseline/B=active "
        "for each generated order."
    )


def infer_ground_truth_side_order_from_payload(payload: dict) -> str:
    explicit_order = payload.get("ground_truth_side_order") or payload.get("case_meta", {}).get("ground_truth_side_order")
    if explicit_order in GROUND_TRUTH_SIDE_ORDER_CHOICES:
        return explicit_order

    supporting_speaker = payload.get("ground_truth_supporting_speaker") or payload.get("case_meta", {}).get("ground_truth_supporting_speaker")
    if supporting_speaker == "B":
        return "gt_second"
    return "gt_first"


def select_baseline_control_trait(expanded_traits: dict) -> str:
    del expanded_traits
    return BASELINE_CONTROL_TRAIT_NAME


def canonical_trait_key(trait_name: str, variant_a: str, variant_b: str) -> str:
    del variant_a, variant_b
    return trait_name


def canonical_style_plan_trait_name(trait_name: str) -> str:
    if trait_name in BASELINE_STYLE_PLAN_TRAITS:
        return BASELINE_CONTROL_TRAIT_NAME
    return trait_name


def trait_enabled_for_variants(
    trait_name: str,
    variant_a: str,
    variant_b: str,
    expanded_traits: dict,
) -> bool:
    if variant_a == "baseline" and variant_b == "baseline":
        return trait_name == BASELINE_CONTROL_TRAIT_NAME
    return trait_name in expanded_traits


def trait_variant_pair_allowed(trait_name: str, variant_a: str, variant_b: str) -> bool:
    if variant_a == "baseline" and variant_b == "baseline":
        return trait_name == BASELINE_CONTROL_TRAIT_NAME
    if trait_name == BASELINE_CONTROL_TRAIT_NAME:
        return False
    return (variant_a, variant_b) in ROLE_BIAS_PAIRS


def validate_trait_variant_pair(trait_name: str | None, variant_a: str | None, variant_b: str | None) -> None:
    if variant_a == "active" and variant_b == "active":
        raise ValueError("active/active detective rollouts are not part of the supported generation matrix.")
    if trait_name == BASELINE_CONTROL_TRAIT_NAME and (variant_a, variant_b) != ROLE_BASELINE_PAIR:
        raise ValueError("The N/A baseline control can only be generated as baseline/baseline.")
    return


def selected_ablation_pairs(
    variant_a: str | None,
    variant_b: str | None,
) -> list[tuple[str, str]]:
    if variant_a is None and variant_b is None:
        return [ROLE_BASELINE_PAIR, *ROLE_BIAS_PAIRS]
    if variant_a is None:
        return [(a, variant_b) for a, b in [ROLE_BASELINE_PAIR, *ROLE_BIAS_PAIRS] if b == variant_b]
    if variant_b is None:
        return [(variant_a, b) for a, b in [ROLE_BASELINE_PAIR, *ROLE_BIAS_PAIRS] if a == variant_a]
    return [(variant_a, variant_b)]


def canonicalize_requested_trait_variants(
    trait_name: str | None,
    variant_a: str | None,
    variant_b: str | None,
) -> tuple[str | None, str | None, str | None]:
    if variant_a == "baseline" and variant_b == "baseline":
        return BASELINE_CONTROL_TRAIT_NAME, "baseline", "baseline"
    return trait_name, variant_a, variant_b


def is_supported_rollout_condition(trait_name: str | None, variant_a: str | None, variant_b: str | None) -> bool:
    if not trait_name or not variant_a or not variant_b:
        return False
    return trait_variant_pair_allowed(trait_name, variant_a, variant_b)


def resolve_trait_ablation_targets(
    expanded_traits: dict,
    *,
    trait_name: str | None,
    ablation_pairs: list[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Resolve the intersection of requested traits and final A/B ablations."""
    trait_name, _unused_a, _unused_b = canonicalize_requested_trait_variants(
        trait_name,
        ablation_pairs[0][0] if len(ablation_pairs) == 1 else None,
        ablation_pairs[0][1] if len(ablation_pairs) == 1 else None,
    )
    if trait_name is not None and trait_name != BASELINE_CONTROL_TRAIT_NAME and trait_name not in expanded_traits:
        raise ValueError(
            f"Unknown trait_name '{trait_name}'. Expected one of: "
            f"{BASELINE_CONTROL_TRAIT_NAME}, {', '.join(expanded_traits)}"
        )

    trait_candidates = (
        [trait_name]
        if trait_name is not None
        else [BASELINE_CONTROL_TRAIT_NAME, *expanded_traits.keys()]
    )
    targets: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for variant_a, variant_b in ablation_pairs:
        if variant_a == "active" and variant_b == "active":
            continue
        for trait in trait_candidates:
            if trait is None:
                continue
            if variant_a == "baseline" and variant_b == "baseline":
                trait = BASELINE_CONTROL_TRAIT_NAME
            if not trait_variant_pair_allowed(trait, variant_a, variant_b):
                continue
            key = (trait, variant_a, variant_b)
            if key in seen:
                continue
            seen.add(key)
            targets.append(key)
    return targets


def speaker_variants_from_role_variants(
    culprit_variant: str,
    rival_variant: str,
    ground_truth_side_order: str,
) -> dict[str, str]:
    if ground_truth_side_order == "gt_first":
        return {"A": culprit_variant, "B": rival_variant}
    if ground_truth_side_order == "gt_second":
        return {"A": rival_variant, "B": culprit_variant}
    raise ValueError(f"Unknown ground_truth_side_order {ground_truth_side_order!r}.")


def make_run_key(
    case_id: str,
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
) -> tuple[str, str, str, str, str, str, str, int]:
    return (
        case_id,
        canonical_trait_key(trait_name, variant_a, variant_b),
        variant_a,
        variant_b,
        ground_truth_side_order,
        content_plan_family,
        interpretation_set_id,
        rollout_index,
    )


def fixed_baseline_speakers_for_variants(variant_a: str, variant_b: str) -> set[str]:
    if variant_a == "active" and variant_b == "baseline":
        return {"B"}
    if variant_a == "baseline" and variant_b == "active":
        return {"A"}
    return set()


def load_completed_runs(output_root: Path) -> set[tuple[str, str, str, str, str, str, str, int]]:
    accepted_dir = output_root / "accepted"
    if not accepted_dir.exists():
        return set()

    completed: set[tuple[str, str, str, str, str, str, str, int]] = set()
    for path in accepted_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        case_id = payload.get("case_meta", {}).get("case_id")
        trait_name = payload.get("trait_name")
        variant_a = payload.get("variant_name_a")
        variant_b = payload.get("variant_name_b")
        ground_truth_side_order = infer_ground_truth_side_order_from_payload(payload)
        content_plan_family = payload.get("content_plan_family")
        interpretation_set_id = payload.get("interpretation_set_id")
        if all([case_id, trait_name, variant_a, variant_b, ground_truth_side_order]):
            if not content_plan_family or not interpretation_set_id:
                continue
            if not is_supported_rollout_condition(trait_name, variant_a, variant_b):
                continue
            canonical_side_order = canonical_ground_truth_side_order_for_variants(
                trait_name,
                variant_a,
                variant_b,
                ground_truth_side_order,
            )
            if canonical_side_order != ground_truth_side_order:
                continue
            ground_truth_side_order = canonical_side_order
            completed.add(make_run_key(
                case_id,
                trait_name,
                variant_a,
                variant_b,
                ground_truth_side_order,
                content_plan_family,
                interpretation_set_id,
                payload_rollout_index(payload),
            ))
    return completed


def load_existing_plan_paths(output_root: Path) -> dict[tuple[str, str, str, str], str]:
    plans_dir = output_root / "plans"
    if not plans_dir.exists():
        return {}

    plan_paths: dict[tuple[str, str, str, str], str] = {}
    for path in sorted(plans_dir.glob("*.json")):
        payload = load_json_if_possible(path)
        case_id = infer_case_id_from_payload(payload) if payload else None

        if case_id is None:
            stem = path.stem
            if "_run_" not in stem:
                continue
            case_id = stem.split("_run_", 1)[0]

        if payload and payload.get("content_plan_version") == pipeline_detective.PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
            family = payload.get("content_plan_family")
            interpretation_set_id = payload.get("interpretation_set_id")
            if not family or not interpretation_set_id:
                continue
            for ground_truth_side_order in GROUND_TRUTH_SIDE_ORDER_CHOICES:
                key = (case_id, ground_truth_side_order, family, interpretation_set_id)
                if key not in plan_paths:
                    plan_paths[key] = str(path)
            continue

        ground_truth_side_order = infer_ground_truth_side_order_from_payload(payload) if payload else "gt_first"
        family = payload.get("content_plan_family") if payload else None
        interpretation_set_id = payload.get("interpretation_set_id") if payload else None
        if not family or not interpretation_set_id:
            continue
        key = (case_id, ground_truth_side_order, family, interpretation_set_id)
        if key not in plan_paths:
            plan_paths[key] = str(path)
    return plan_paths


def load_existing_style_paths(output_root: Path) -> dict[tuple[str, str, str, str], str]:
    style_dir = output_root / "style_plans"
    if not style_dir.exists():
        return {}

    style_paths: dict[tuple[str, str, str, str], str] = {}
    for path in sorted(style_dir.glob("*.json")):
        payload = load_json_if_possible(path)
        if not payload:
            continue
        case_id = payload.get("case_id")
        trait_name = payload.get("trait_name")
        family = payload.get("content_plan_family")
        interpretation_set_id = payload.get("interpretation_set_id")
        if not all([case_id, trait_name, family, interpretation_set_id]):
            continue
        key = (case_id, trait_name, family, interpretation_set_id)
        if key not in style_paths:
            style_paths[key] = str(path)
    return style_paths


def build_expanded_trait_library() -> dict:
    source = copy.deepcopy(traitsV2.TRAIT_LIBRARY)
    expanded: dict[str, dict] = {}

    for trait_name, trait_cfg in source.items():
        if trait_name != "fallacy_trait":
            expanded[trait_name] = trait_cfg
            continue

        baseline_cfg = copy.deepcopy(trait_cfg["baseline"])
        parent_active = trait_cfg["active"]

        for subtrait_key, subtrait_cfg in trait_cfg["subtraits"].items():
            expanded_name = f"fallacy_trait__{subtrait_key}"
            expanded[expanded_name] = {
                "baseline": copy.deepcopy(baseline_cfg),
                "active": {
                    "name": subtrait_cfg["name"],
                    "definition": subtrait_cfg["definition"],
                    "eval_guidance": subtrait_cfg["eval_guidance"],
                    "examples": subtrait_cfg["examples"],
                    "contrast_with_baseline": subtrait_cfg["contrast_with_baseline"],
                },
                "source_trait": trait_name,
                "subtrait_key": subtrait_key,
            }

    return expanded


def build_smoke_test_runs(
    expanded_traits: dict,
    case_id: str | None = None,
    trait_name: str | None = None,
    variant_a: str = "active",
    variant_b: str = "baseline",
    ground_truth_side_order: str | None = None,
    interpretation_set: str | None = None,
    rollouts_per_combo: int = 1,
) -> list[StepRun]:
    validate_trait_variant_pair(trait_name, variant_a, variant_b)
    case_lookup = {slug: label for label, slug in CASE_SPECS}

    selected_case_id = case_id or CASE_SPECS[0][1]
    if selected_case_id not in case_lookup:
        raise ValueError(
            f"Unknown smoke-test case_id '{selected_case_id}'. Expected one of: {', '.join(case_lookup)}"
        )

    targets = resolve_trait_ablation_targets(
        expanded_traits,
        trait_name=trait_name,
        ablation_pairs=[(variant_a, variant_b)],
    )
    if not targets:
        return []
    selected_trait_name, variant_a, variant_b = targets[0]
    side_orders = selected_ground_truth_side_orders(ground_truth_side_order)
    invalid_orders = [
        order
        for order in side_orders
        if not anchoring_speaker_variants_allowed(selected_trait_name, variant_a, variant_b, order)
    ]
    if invalid_orders:
        raise ValueError(
            anchoring_invalid_request_message(
                trait_name=selected_trait_name,
                variant_a=variant_a,
                variant_b=variant_b,
                ground_truth_side_order=ground_truth_side_order,
            )
        )

    interpretation_targets = selected_interpretation_targets(
        selected_case_id,
        selected_trait_name,
        interpretation_set,
    )
    if is_baseline_variant_pair(variant_a, variant_b):
        interpretation_targets = [(CANONICAL_BASELINE_CONTENT_PLAN_FAMILY, CANONICAL_BASELINE_INTERPRETATION_SET_ID)]
        rollout_targets = [CANONICAL_BASELINE_ROLLOUT_INDEX]
    else:
        rollout_targets = list(rollout_indices(rollouts_per_combo))

    return [
        (
            case_lookup[selected_case_id],
            selected_case_id,
            selected_trait_name,
            variant_a,
            variant_b,
            side_order,
            family,
            interpretation_id,
            rollout_index,
        )
        for side_order in side_orders
        for family, interpretation_id in interpretation_targets
        for rollout_index in rollout_targets
    ]


def build_step_runs(
    expanded_traits: dict,
    *,
    case_id: str | None,
    trait_name: str | None,
    variant_a: str | None,
    variant_b: str | None,
    ground_truth_side_order: str | None,
    interpretation_set: str | None = None,
    rollouts_per_combo: int = 1,
) -> list[StepRun]:
    validate_trait_variant_pair(trait_name, variant_a, variant_b)
    case_runs = [
        (label, slug)
        for label, slug in CASE_SPECS
        if case_id is None or slug == case_id
    ]
    if case_id is not None and not case_runs:
        raise ValueError(f"Unknown case_id '{case_id}'. Expected one of: {', '.join(slug for _, slug in CASE_SPECS)}")

    targets = resolve_trait_ablation_targets(
        expanded_traits,
        trait_name=trait_name,
        ablation_pairs=selected_ablation_pairs(variant_a, variant_b),
    )

    side_orders = selected_ground_truth_side_orders(ground_truth_side_order)

    runs: list[StepRun] = []
    seen_keys: set[tuple[str, str, str, str, str, str, str, int]] = set()
    invalid_anchoring_requests: list[tuple[str, str, str]] = []
    for case_label, case_slug in case_runs:
        for trait, var_a, var_b in targets:
            interpretation_targets = selected_interpretation_targets(
                case_slug,
                trait,
                interpretation_set,
                strict=False,
            )
            if not interpretation_targets:
                continue
            if is_baseline_variant_pair(var_a, var_b):
                interpretation_targets = [(CANONICAL_BASELINE_CONTENT_PLAN_FAMILY, CANONICAL_BASELINE_INTERPRETATION_SET_ID)]
                rollout_targets = [CANONICAL_BASELINE_ROLLOUT_INDEX]
            else:
                rollout_targets = list(rollout_indices(rollouts_per_combo))
            for side_order in side_orders:
                if not anchoring_speaker_variants_allowed(trait, var_a, var_b, side_order):
                    invalid_anchoring_requests.append((var_a, var_b, side_order))
                    continue
                canonical_side_order = canonical_ground_truth_side_order_for_variants(
                    trait,
                    var_a,
                    var_b,
                    side_order,
                )
                for family, interpretation_id in interpretation_targets:
                    for rollout_index in rollout_targets:
                        run_key = make_run_key(
                            case_slug,
                            trait,
                            var_a,
                            var_b,
                            canonical_side_order,
                            family,
                            interpretation_id,
                            rollout_index,
                        )
                        if run_key in seen_keys:
                            continue
                        seen_keys.add(run_key)
                        runs.append((
                            case_label,
                            case_slug,
                            trait,
                            var_a,
                            var_b,
                            canonical_side_order,
                            family,
                            interpretation_id,
                            rollout_index,
                        ))
    if not runs and invalid_anchoring_requests and trait_name == "anchoring_bias":
        first_variant_a, first_variant_b, _first_order = invalid_anchoring_requests[0]
        raise ValueError(
            anchoring_invalid_request_message(
                trait_name=trait_name,
                variant_a=first_variant_a,
                variant_b=first_variant_b,
                ground_truth_side_order=ground_truth_side_order,
            )
        )
    return runs


def resolve_output_root(reference_output: str | None) -> Path | None:
    if not reference_output:
        return None
    path = Path(reference_output)
    if path.is_absolute():
        return pipeline_detective.resolve_repo_path(path)
    if not path.is_absolute():
        if path.parts and path.parts[0] == "outputs":
            path = ROOT / path
        else:
            path = ROOT / "outputs" / path
    return path


@cache
def case_topic_by_id() -> dict[str, str]:
    return {slug: load_case(slug)["topic"] for _, slug in CASE_SPECS}


def infer_case_id_from_payload(payload: dict) -> str | None:
    payload_case = payload.get("case_meta", {}).get("case_id") or payload.get("case_id")
    if payload_case:
        return payload_case

    payload_topic = payload.get("topic")
    if not payload_topic:
        return None

    for case_id, case_topic in case_topic_by_id().items():
        if payload_topic == case_topic:
            return case_id
    return None


def metadata_matches(
    payload: dict,
    *,
    case_id: str,
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    subdir: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
) -> bool:
    trait_name, variant_a, variant_b = canonicalize_requested_trait_variants(
        trait_name,
        variant_a,
        variant_b,
    )
    ground_truth_side_order = canonical_ground_truth_side_order_for_variants(
        trait_name,
        variant_a,
        variant_b,
        ground_truth_side_order,
    )
    payload_case = infer_case_id_from_payload(payload)
    if payload_case != case_id:
        return False
    if payload.get("content_plan_family") != content_plan_family:
        return False
    if payload.get("interpretation_set_id") != interpretation_set_id:
        return False
    payload_ground_truth_side_order = infer_ground_truth_side_order_from_payload(payload)
    if subdir == "plans":
        return payload_ground_truth_side_order == ground_truth_side_order
    if subdir == "style_plans":
        return payload.get("trait_name") == canonical_style_plan_trait_name(trait_name)
    payload_trait = payload.get("trait_name")
    payload_variant_a = payload.get("variant_name_a") or payload.get("variant_a")
    payload_variant_b = payload.get("variant_name_b") or payload.get("variant_b")
    if not is_supported_rollout_condition(payload_trait, payload_variant_a, payload_variant_b):
        return False
    payload_canonical_side_order = canonical_ground_truth_side_order_for_variants(
        payload_trait,
        payload_variant_a,
        payload_variant_b,
        payload_ground_truth_side_order,
    )
    if payload_canonical_side_order != payload_ground_truth_side_order:
        return False
    payload_ground_truth_side_order = payload_canonical_side_order
    return (
        payload_trait == trait_name
        and payload_variant_a == variant_a
        and payload_variant_b == variant_b
        and payload_ground_truth_side_order == ground_truth_side_order
        and payload_rollout_index(payload) == rollout_index
    )


def load_json_if_possible(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def eval_payload_passed(payload: dict | None) -> bool:
    if not payload:
        return False
    if payload.get("final_bias_eval_passed") is not None:
        return bool(payload["final_bias_eval_passed"])
    return bool(payload.get("passed"))


def rar_payload_passed(payload: dict | None) -> bool:
    if not payload:
        return False
    return bool(payload.get("passed"))


def dialogue_eval_prefix(dialogue_path: Path) -> str | None:
    stem = dialogue_path.stem
    for suffix in ("_accepted_dialogue", "_dialogue"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def find_passing_eval_for_dialogue(root: Path, dialogue_path: Path) -> Path | None:
    prefix = dialogue_eval_prefix(dialogue_path)
    if prefix is None:
        return None
    eval_path = root / "evals" / f"{prefix}_eval.json"
    if eval_payload_passed(load_json_if_possible(eval_path)):
        return eval_path

    dialogue_payload = load_json_if_possible(dialogue_path)
    if not dialogue_payload:
        return None
    dialogue_case, dialogue_trait, dialogue_variant_a, dialogue_variant_b, dialogue_order, dialogue_family, dialogue_interpretation, dialogue_rollout = extract_run_metadata(dialogue_payload)
    if not all([dialogue_case, dialogue_trait, dialogue_variant_a, dialogue_variant_b, dialogue_order, dialogue_family, dialogue_interpretation]):
        return None
    eval_dir = root / "evals"
    if not eval_dir.exists():
        return None
    fallback_matches: list[Path] = []
    for candidate in eval_dir.glob("*.json"):
        eval_payload = load_json_if_possible(candidate)
        if not eval_payload_passed(eval_payload):
            continue
        if metadata_matches(
            eval_payload,
            case_id=dialogue_case,
            trait_name=dialogue_trait,
            variant_a=dialogue_variant_a,
            variant_b=dialogue_variant_b,
            ground_truth_side_order=dialogue_order,
            content_plan_family=dialogue_family,
            interpretation_set_id=dialogue_interpretation,
            subdir="evals",
            rollout_index=dialogue_rollout,
        ):
            fallback_matches.append(candidate)
    fallback_matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if fallback_matches:
        return fallback_matches[0]
    return None


def find_passing_rar_for_dialogue(root: Path, dialogue_path: Path) -> Path | None:
    prefix = dialogue_eval_prefix(dialogue_path)
    if prefix is not None:
        rar_path = root / "rar_evals" / f"{prefix}_rar_eval.json"
        if rar_payload_passed(load_json_if_possible(rar_path)):
            return rar_path

    dialogue_payload = load_json_if_possible(dialogue_path)
    if not dialogue_payload:
        return None
    dialogue_case, dialogue_trait, dialogue_variant_a, dialogue_variant_b, dialogue_order, dialogue_family, dialogue_interpretation, dialogue_rollout = extract_run_metadata(dialogue_payload)
    if not all([dialogue_case, dialogue_trait, dialogue_variant_a, dialogue_variant_b, dialogue_order, dialogue_family, dialogue_interpretation]):
        return None
    rar_dir = root / "rar_evals"
    if not rar_dir.exists():
        return None
    fallback_matches: list[Path] = []
    for candidate in rar_dir.glob("*.json"):
        rar_payload = load_json_if_possible(candidate)
        if not rar_payload_passed(rar_payload):
            continue
        if metadata_matches(
            rar_payload,
            case_id=dialogue_case,
            trait_name=dialogue_trait,
            variant_a=dialogue_variant_a,
            variant_b=dialogue_variant_b,
            ground_truth_side_order=dialogue_order,
            content_plan_family=dialogue_family,
            interpretation_set_id=dialogue_interpretation,
            subdir="rar_evals",
            rollout_index=dialogue_rollout,
        ):
            fallback_matches.append(candidate)
    fallback_matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if fallback_matches:
        return fallback_matches[0]
    return None


def canonical_outputs_for_attempt(output_root: Path, attempt_path: Path) -> list[Path]:
    match = ATTEMPT_OUTPUT_RE.match(attempt_path.name)
    if not match:
        return []
    prefix = match.group("prefix")
    kind = match.group("kind")
    if kind == "dialogue":
        return [
            output_root / "dialogues" / f"{prefix}_dialogue.json",
            output_root / "accepted" / f"{prefix}_accepted_dialogue.json",
        ]
    if kind == "eval":
        return [output_root / "evals" / f"{prefix}_eval.json"]
    return []


def cleanup_redundant_attempt_outputs(output_root: Path) -> dict[str, int]:
    """Delete attempt files once a canonical output for the same run exists."""
    counts = {
        "dialogues_deleted": 0,
        "evals_deleted": 0,
        "kept_without_canonical": 0,
        "errors": 0,
    }
    for subdir in ("dialogues", "evals"):
        search_dir = output_root / subdir
        if not search_dir.exists():
            continue
        for attempt_path in sorted(search_dir.glob("*_attempt*_*.json")):
            match = ATTEMPT_OUTPUT_RE.match(attempt_path.name)
            if not match:
                continue
            canonical_paths = canonical_outputs_for_attempt(output_root, attempt_path)
            if not any(path.exists() for path in canonical_paths):
                counts["kept_without_canonical"] += 1
                continue
            try:
                attempt_path.unlink()
            except OSError as exc:
                counts["errors"] += 1
                print(f"[detective-v2] cleanup failed for {attempt_path}: {exc}")
                continue
            key = "dialogues_deleted" if match.group("kind") == "dialogue" else "evals_deleted"
            counts[key] += 1
            print(f"[detective-v2] cleanup removed redundant attempt file: {attempt_path}")
    return counts


def cleanup_redundant_attempt_outputs_for_run(output_root: Path, run_id: str) -> dict[str, int]:
    """Delete attempt files for one run once its successful output exists."""
    counts = {
        "dialogues_deleted": 0,
        "evals_deleted": 0,
        "kept_without_canonical": 0,
        "errors": 0,
    }
    for subdir in ("dialogues", "evals"):
        search_dir = output_root / subdir
        if not search_dir.exists():
            continue
        for attempt_path in sorted(search_dir.glob(f"*{run_id}_attempt*_*.json")):
            match = ATTEMPT_OUTPUT_RE.match(attempt_path.name)
            if not match or not match.group("prefix").endswith(run_id):
                continue
            canonical_paths = canonical_outputs_for_attempt(output_root, attempt_path)
            if not any(path.exists() for path in canonical_paths):
                counts["kept_without_canonical"] += 1
                continue
            try:
                attempt_path.unlink()
            except OSError as exc:
                counts["errors"] += 1
                print(f"[detective-v2] cleanup failed for {attempt_path}: {exc}")
                continue
            key = "dialogues_deleted" if match.group("kind") == "dialogue" else "evals_deleted"
            counts[key] += 1
            print(f"[detective-v2] cleanup removed redundant attempt file: {attempt_path}")
    return counts


def maybe_cleanup_redundant_attempt_outputs(output_root: Path, *, keep_attempt_files: bool) -> None:
    if keep_attempt_files:
        print("[detective-v2] keeping attempt files (--keep-attempt-files set)")
        return
    counts = cleanup_redundant_attempt_outputs(output_root)
    total_deleted = counts["dialogues_deleted"] + counts["evals_deleted"]
    print(
        "[detective-v2] cleanup summary: "
        f"deleted={total_deleted} "
        f"dialogues={counts['dialogues_deleted']} "
        f"evals={counts['evals_deleted']} "
        f"kept_without_canonical={counts['kept_without_canonical']} "
        f"errors={counts['errors']}"
    )


def extract_run_metadata(payload: dict) -> tuple[str | None, str | None, str | None, str | None, str, str | None, str | None, int]:
    return (
        infer_case_id_from_payload(payload),
        payload.get("trait_name"),
        payload.get("variant_name_a") or payload.get("variant_a"),
        payload.get("variant_name_b") or payload.get("variant_b"),
        infer_ground_truth_side_order_from_payload(payload),
        payload.get("content_plan_family"),
        payload.get("interpretation_set_id"),
        payload_rollout_index(payload),
    )


def build_reference_step_targets(
    reference_output_dir: Path,
    *,
    subdirs: tuple[str, ...],
    case_id: str | None,
    trait_name: str | None,
    variant_a: str | None,
    variant_b: str | None,
    ground_truth_side_order: str | None,
    interpretation_set: str | None = None,
) -> list[tuple[StepRun, Path]]:
    trait_name, variant_a, variant_b = canonicalize_requested_trait_variants(
        trait_name,
        variant_a,
        variant_b,
    )
    latest_by_key: dict[tuple[str, str, str, str, str, str, str, int], tuple[float, StepRun, Path]] = {}

    for subdir in subdirs:
        search_dir = reference_output_dir / subdir
        if not search_dir.exists():
            continue

        for path in search_dir.glob("*.json"):
            payload = load_json_if_possible(path)
            if not payload:
                continue

            payload_case, payload_trait, payload_variant_a, payload_variant_b, payload_ground_truth_side_order, payload_family, payload_interpretation, payload_rollout = extract_run_metadata(payload)
            if not all([payload_case, payload_trait, payload_variant_a, payload_variant_b, payload_ground_truth_side_order, payload_family, payload_interpretation]):
                continue
            if not is_supported_rollout_condition(payload_trait, payload_variant_a, payload_variant_b):
                continue
            payload_original_order = payload_ground_truth_side_order
            payload_canonical_order = canonical_ground_truth_side_order_for_variants(
                payload_trait,
                payload_variant_a,
                payload_variant_b,
                payload_ground_truth_side_order,
            )
            if payload_canonical_order != payload_original_order:
                continue
            if case_id is not None and payload_case != case_id:
                continue
            if trait_name is not None and payload_trait != trait_name:
                continue
            if variant_a is not None and payload_variant_a != variant_a:
                continue
            if variant_b is not None and payload_variant_b != variant_b:
                continue
            if ground_truth_side_order is not None and payload_ground_truth_side_order != ground_truth_side_order:
                continue
            if interpretation_set is not None and payload_interpretation != interpretation_set:
                continue
            payload_ground_truth_side_order = payload_canonical_order

            key = make_run_key(
                payload_case,
                payload_trait,
                payload_variant_a,
                payload_variant_b,
                payload_ground_truth_side_order,
                payload_family,
                payload_interpretation,
                payload_rollout,
            )
            run = (
                CASE_LABEL_BY_ID.get(payload_case, payload_case),
                payload_case,
                payload_trait,
                payload_variant_a,
                payload_variant_b,
                payload_ground_truth_side_order,
                payload_family,
                payload_interpretation,
                payload_rollout,
            )
            mtime = path.stat().st_mtime
            prev = latest_by_key.get(key)
            if prev is None or mtime > prev[0]:
                latest_by_key[key] = (mtime, run, path)

    return [
        (item[1], item[2])
        for item in sorted(latest_by_key.values(), key=lambda item: item[0], reverse=True)
    ]


def build_reference_step_runs(
    reference_output_dir: Path,
    *,
    subdirs: tuple[str, ...],
    case_id: str | None,
    trait_name: str | None,
    variant_a: str | None,
    variant_b: str | None,
    ground_truth_side_order: str | None,
    interpretation_set: str | None = None,
) -> list[StepRun]:
    return [
        run
        for run, _path in build_reference_step_targets(
            reference_output_dir,
            subdirs=subdirs,
            case_id=case_id,
            trait_name=trait_name,
            variant_a=variant_a,
            variant_b=variant_b,
            ground_truth_side_order=ground_truth_side_order,
            interpretation_set=interpretation_set,
        )
    ]


def find_reference_matches(
    reference_output_dir: Path,
    *,
    subdir: str,
    case_id: str,
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
) -> list[Path]:
    trait_name, variant_a, variant_b = canonicalize_requested_trait_variants(
        trait_name,
        variant_a,
        variant_b,
    )
    ground_truth_side_order = canonical_ground_truth_side_order_for_variants(
        trait_name,
        variant_a,
        variant_b,
        ground_truth_side_order,
    )
    search_dir = reference_output_dir / subdir
    if not search_dir.exists():
        return []

    matches: list[Path] = []
    for path in search_dir.glob("*.json"):
        payload = load_json_if_possible(path)
        if payload and metadata_matches(
            payload,
            case_id=case_id,
            trait_name=trait_name,
            variant_a=variant_a,
            variant_b=variant_b,
            ground_truth_side_order=ground_truth_side_order,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
            rollout_index=rollout_index,
            subdir=subdir,
        ):
            matches.append(path)
            continue

        stem = path.stem
        rollout_stem_matches = rollout_index == 1 or f"rollout-{rollout_index}" in stem
        if subdir == "plans":
            if case_id in stem and content_plan_family in stem and interpretation_set_id in stem and (
                ground_truth_side_order in stem
                or (
                    ground_truth_side_order == "gt_first"
                    and not any(order in stem for order in GROUND_TRUTH_SIDE_ORDER_CHOICES)
                )
            ):
                matches.append(path)
        elif subdir == "style_plans":
            style_trait = canonical_style_plan_trait_name(trait_name)
            if case_id in stem and content_plan_family in stem and interpretation_set_id in stem and style_trait in stem:
                matches.append(path)
        elif rollout_stem_matches and case_id in stem and content_plan_family in stem and interpretation_set_id in stem and trait_name in stem and variant_a in stem and variant_b in stem and (
            ground_truth_side_order in stem
            or (
                ground_truth_side_order == "gt_first"
                and not any(order in stem for order in GROUND_TRUTH_SIDE_ORDER_CHOICES)
            )
        ):
            matches.append(path)

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


def find_reference_file(
    reference_output_dir: Path,
    *,
    subdir: str,
    case_id: str,
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
) -> Path:
    matches = find_reference_matches(
        reference_output_dir,
        subdir=subdir,
        case_id=case_id,
        trait_name=trait_name,
        variant_a=variant_a,
        variant_b=variant_b,
        ground_truth_side_order=ground_truth_side_order,
        content_plan_family=content_plan_family,
        interpretation_set_id=interpretation_set_id,
        rollout_index=rollout_index,
    )
    if not matches:
        raise FileNotFoundError(
            f"No matching reference file found in {reference_output_dir / subdir} "
            f"for case={case_id} trait={trait_name} A={variant_a} B={variant_b} "
            f"order={ground_truth_side_order} family={content_plan_family} interp={interpretation_set_id}"
        )
    print(f"[detective-v2] selected reference file from {subdir}: {matches[0]}")
    return matches[0]


def find_content_plan_file(
    reference_output_dir: Path,
    *,
    case_id: str,
    content_plan_family: str,
    interpretation_set_id: str,
) -> Path:
    search_dir = reference_output_dir / "plans"
    canonical = search_dir / (
        f"{case_id}__{pipeline_detective.safe_artifact_part(content_plan_family)}__"
        f"{pipeline_detective.safe_artifact_part(interpretation_set_id)}.json"
    )
    if canonical.exists():
        return canonical
    matches: list[Path] = []
    for path in search_dir.glob("*.json") if search_dir.exists() else []:
        payload = load_json_if_possible(path)
        if payload and (
            (payload.get("case_id") == case_id or case_id in path.stem)
            and payload.get("content_plan_family") == content_plan_family
            and payload.get("interpretation_set_id") == interpretation_set_id
        ):
            matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No content plan found for case={case_id} family={content_plan_family} "
            f"interp={interpretation_set_id} in {search_dir}"
        )
    print(f"[detective-v2] selected content plan: {matches[0]}")
    return matches[0]


def find_style_plan_file(
    reference_output_dir: Path,
    *,
    case_id: str,
    trait_name: str,
    culprit_variant: str | None = None,
    rival_variant: str | None = None,
    content_plan_family: str = "standard",
    interpretation_set_id: str = "primary-1",
) -> Path:
    del culprit_variant, rival_variant
    search_dir = reference_output_dir / "style_plans"
    style_trait_name = canonical_style_plan_trait_name(trait_name)
    safe_trait = style_trait_name.replace("/", "_")
    suffix = f"family-{pipeline_detective.safe_artifact_part(content_plan_family)}__interp-{pipeline_detective.safe_artifact_part(interpretation_set_id)}"
    canonical = search_dir / f"{case_id}__{suffix}__{safe_trait}.json"
    if canonical.exists():
        return canonical
    matches: list[Path] = []
    for path in search_dir.glob("*.json") if search_dir.exists() else []:
        payload = load_json_if_possible(path)
        if payload and (
            payload.get("case_id") == case_id
            and payload.get("trait_name") == style_trait_name
            and payload.get("content_plan_family") == content_plan_family
            and payload.get("interpretation_set_id") == interpretation_set_id
        ):
            matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No style plan found for case={case_id} trait={style_trait_name} "
            f"family={content_plan_family} interp={interpretation_set_id} in {search_dir}"
        )
    print(f"[detective-v2] selected style plan: {matches[0]}")
    return matches[0]


def find_role_dialogue_file(
    reference_output_dir: Path,
    *,
    case_id: str,
    trait_name: str,
    culprit_variant: str,
    rival_variant: str,
    rollout_index: int = 1,
    seed: int | None = None,
    content_plan_family: str = "standard",
    interpretation_set_id: str = "primary-1",
) -> Path:
    trait_name, culprit_variant, rival_variant = canonicalize_requested_trait_variants(
        trait_name,
        culprit_variant,
        rival_variant,
    )
    if (
        trait_name == BASELINE_CONTROL_TRAIT_NAME
        and culprit_variant == "baseline"
        and rival_variant == "baseline"
    ):
        content_plan_family, interpretation_set_id, rollout_index = canonical_baseline_identity()
        seed = None
    search_dir = reference_output_dir / "role_dialogues"
    safe_trait = trait_name.replace("/", "_")
    seed_part = f"seed-{seed}" if seed is not None else "seed-none"
    suffix = f"family-{pipeline_detective.safe_artifact_part(content_plan_family)}__interp-{pipeline_detective.safe_artifact_part(interpretation_set_id)}"
    canonical = search_dir / (
        f"{case_id}__{suffix}__{safe_trait}__culprit-{culprit_variant}__rival-{rival_variant}__{seed_part}.json"
    )
    canonical_with_rollout = search_dir / (
        f"{case_id}__{suffix}__{safe_trait}__culprit-{culprit_variant}__rival-{rival_variant}__"
        f"rollout-{rollout_index}__{seed_part}.json"
    )
    if canonical_with_rollout.exists():
        return canonical_with_rollout
    if rollout_index == 1 and canonical.exists():
        payload = load_json_if_possible(canonical)
        if not payload or payload_rollout_index(payload) == 1:
            return canonical
    matches: list[Path] = []
    for path in search_dir.glob("*.json") if search_dir.exists() else []:
        payload = load_json_if_possible(path)
        if not payload:
            continue
        if payload.get("case_id") != case_id or payload.get("trait_name") != trait_name:
            continue
        if payload.get("content_plan_family") != content_plan_family or payload.get("interpretation_set_id") != interpretation_set_id:
            continue
        if payload.get("culprit_variant") != culprit_variant or payload.get("rival_variant") != rival_variant:
            continue
        if payload_rollout_index(payload) != rollout_index:
            continue
        if payload.get("seed") != seed:
            continue
        matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No role dialogue found for case={case_id} trait={trait_name} culprit={culprit_variant} "
            f"rival={rival_variant} seed={seed} in {search_dir}"
        )
    print(f"[detective-v2] selected role dialogue: {matches[0]}")
    return matches[0]


def find_reusable_case_openings(role_dialogues_dir: Path, case_id: str) -> Path | None:
    for path in sorted(
        role_dialogues_dir.glob("*.json") if role_dialogues_dir.exists() else [],
        key=lambda candidate: candidate.stat().st_mtime,
    ):
        payload = load_json_if_possible(path)
        if not payload or payload.get("case_id") != case_id:
            continue
        role_turns = payload.get("role_turns") or {}
        culprit_opening = role_turns.get("culprit_opening") or {}
        rival_opening = role_turns.get("rival_opening") or {}
        if culprit_opening.get("utterance") and rival_opening.get("utterance"):
            return path
    return None


def find_dialogue_file(
    reference_output_dir: Path,
    *,
    case_id: str,
    trait_name: str,
    culprit_variant: str,
    rival_variant: str,
    ground_truth_side_order: str,
    seed: int | None = None,
    content_plan_family: str = "standard",
    interpretation_set_id: str = "primary-1",
    rollout_index: int = 1,
) -> Path:
    trait_name, culprit_variant, rival_variant = canonicalize_requested_trait_variants(
        trait_name,
        culprit_variant,
        rival_variant,
    )
    ground_truth_side_order = canonical_ground_truth_side_order_for_variants(
        trait_name,
        culprit_variant,
        rival_variant,
        ground_truth_side_order,
    )
    search_dir = reference_output_dir / "dialogues"
    matches: list[Path] = []
    for path in search_dir.glob("*.json") if search_dir.exists() else []:
        payload = load_json_if_possible(path)
        if not payload:
            continue
        if payload.get("case_id") != case_id or payload.get("trait_name") != trait_name:
            continue
        if payload.get("content_plan_family") != content_plan_family or payload.get("interpretation_set_id") != interpretation_set_id:
            continue
        if payload.get("culprit_variant") != culprit_variant or payload.get("rival_variant") != rival_variant:
            continue
        if infer_ground_truth_side_order_from_payload(payload) != ground_truth_side_order:
            continue
        if payload_rollout_index(payload) != rollout_index:
            continue
        if seed is not None and payload.get("seed") != seed:
            continue
        matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No dialogue found for case={case_id} trait={trait_name} culprit={culprit_variant} "
            f"rival={rival_variant} order={ground_truth_side_order} seed={seed} in {search_dir}"
        )
    return matches[0]


def normalize_role_variants_from_args(args) -> tuple[str, str]:
    role_culprit = args.culprit_variant
    role_rival = args.rival_variant
    ab_a = args.variant_a
    ab_b = args.variant_b
    order = args.ground_truth_side_order
    if ab_a is not None or ab_b is not None:
        if order not in GROUND_TRUTH_SIDE_ORDER_CHOICES:
            raise ValueError("--gt-order is required when using --variant-a/--variant-b aliases.")
        a = ab_a or "active"
        b = ab_b or "baseline"
        if order == "gt_first":
            alias_culprit, alias_rival = a, b
        else:
            alias_culprit, alias_rival = b, a
        if role_culprit is not None and role_culprit != alias_culprit:
            raise ValueError("Contradictory --culprit-variant and --variant-a/--variant-b assignment.")
        if role_rival is not None and role_rival != alias_rival:
            raise ValueError("Contradictory --rival-variant and --variant-a/--variant-b assignment.")
        role_culprit = alias_culprit
        role_rival = alias_rival
    return role_culprit or "active", role_rival or "baseline"


def selected_role_variant_pairs(
    culprit_variant: str | None,
    rival_variant: str | None,
    *,
    role_variant_pairs: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    if role_variant_pairs is not None:
        return list(role_variant_pairs)
    if culprit_variant is None and rival_variant is None:
        return [ROLE_BASELINE_PAIR, *ROLE_BIAS_PAIRS]
    if culprit_variant is None:
        return [(culprit, rival_variant) for culprit, rival in [ROLE_BASELINE_PAIR, *ROLE_BIAS_PAIRS] if rival == rival_variant]
    if rival_variant is None:
        return [(culprit_variant, rival) for culprit, rival in [ROLE_BASELINE_PAIR, *ROLE_BIAS_PAIRS] if culprit == culprit_variant]
    return [(culprit_variant, rival_variant)]


def selected_style_variant_pairs(culprit_variant: str | None, rival_variant: str | None) -> list[tuple[str, str]]:
    if culprit_variant is None and rival_variant is None:
        return [ROLE_BASELINE_PAIR, ROLE_BIAS_PAIRS[0]]
    return selected_role_variant_pairs(culprit_variant, rival_variant)


def role_step_run_from_speaker_step_run(run: StepRun) -> StepRun:
    (
        case_label,
        case_id,
        trait_name,
        variant_a,
        variant_b,
        ground_truth_side_order,
        content_plan_family,
        interpretation_set_id,
        rollout_index,
    ) = run
    if ground_truth_side_order == "gt_first":
        culprit_variant, rival_variant = variant_a, variant_b
    elif ground_truth_side_order == "gt_second":
        culprit_variant, rival_variant = variant_b, variant_a
    else:
        raise ValueError(f"Unknown ground_truth_side_order {ground_truth_side_order!r}.")
    return (
        case_label,
        case_id,
        trait_name,
        culprit_variant,
        rival_variant,
        ground_truth_side_order,
        content_plan_family,
        interpretation_set_id,
        rollout_index,
    )


def build_role_stage_step_runs(
    expanded_traits: dict,
    *,
    step: str,
    case_id: str | None,
    trait_name: str | None,
    culprit_variant: str | None,
    rival_variant: str | None,
    ground_truth_side_order: str | None,
    role_variant_pairs: list[tuple[str, str]] | None = None,
    interpretation_set: str | None = None,
    rollouts_per_combo: int = 1,
) -> list[StepRun]:
    if culprit_variant == "active" and rival_variant == "active":
        raise ValueError("active/active detective rollouts are not part of the supported generation matrix.")
    case_runs = [
        (label, slug)
        for label, slug in CASE_SPECS
        if case_id is None or slug == case_id
    ]
    if case_id is not None and not case_runs:
        raise ValueError(f"Unknown case_id '{case_id}'. Expected one of: {', '.join(slug for _, slug in CASE_SPECS)}")

    if step in {"content-plan", "attention-question"}:
        if trait_name is not None and trait_name != BASELINE_CONTROL_TRAIT_NAME and trait_name not in expanded_traits:
            raise ValueError(
                f"Unknown trait_name '{trait_name}'. Expected one of: "
                f"{BASELINE_CONTROL_TRAIT_NAME}, {', '.join(expanded_traits)}"
            )
        variant_pairs = [("baseline", "baseline")]
        side_orders = ["gt_first"]
    else:
        variant_pairs = (
            selected_style_variant_pairs(culprit_variant, rival_variant)
            if step == "style-plan"
            else selected_role_variant_pairs(
                culprit_variant,
                rival_variant,
                role_variant_pairs=role_variant_pairs,
            )
        )
        side_orders = selected_ground_truth_side_orders(ground_truth_side_order) if step == "dialogue" else ["gt_first"]

    effective_rollouts_per_combo = rollouts_per_combo if step == "dialogue" else 1
    runs: list[StepRun] = []
    seen_keys: set[tuple[str, str, str, str, str, str, str, int]] = set()
    invalid_anchoring_requests: list[tuple[str, str, str]] = []
    for case_label, case_slug in case_runs:
        for role_culprit, role_rival in variant_pairs:
            for side_order in side_orders:
                if step == "dialogue" and not anchoring_role_variants_allowed(
                    trait_name or "",
                    role_culprit,
                    role_rival,
                    side_order,
                ):
                    invalid_anchoring_requests.append((role_culprit, role_rival, side_order))
                    continue
                validation_side_order = ground_truth_side_order or side_order
                speaker_variants = speaker_variants_from_role_variants(
                    role_culprit,
                    role_rival,
                    validation_side_order,
                )
                targets = (
                    [(BASELINE_CONTROL_TRAIT_NAME, "baseline", "baseline")]
                    if step in {"content-plan", "attention-question"}
                    else resolve_trait_ablation_targets(
                        expanded_traits,
                        trait_name=trait_name,
                        ablation_pairs=[(speaker_variants["A"], speaker_variants["B"])],
                    )
                )
                for trait, _variant_a, _variant_b in targets:
                    if step == "dialogue" and not anchoring_role_variants_allowed(
                        trait,
                        role_culprit,
                        role_rival,
                        side_order,
                    ):
                        invalid_anchoring_requests.append((role_culprit, role_rival, side_order))
                        continue
                    output_trait = trait
                    output_role_culprit = role_culprit
                    output_role_rival = role_rival
                    output_speaker_variants = dict(speaker_variants)
                    output_side_order = side_order
                    if step == "style-plan" and trait in BASELINE_STYLE_PLAN_TRAITS:
                        output_trait = BASELINE_CONTROL_TRAIT_NAME
                        output_role_culprit = "baseline"
                        output_role_rival = "baseline"
                        output_speaker_variants = {"A": "baseline", "B": "baseline"}
                        output_side_order = "gt_first"
                    if step in {"content-plan", "attention-question"} and trait_name is None and "verbosity_bias" in expanded_traits:
                        interpretation_targets = [
                            *selected_interpretation_targets(case_slug, BASELINE_CONTROL_TRAIT_NAME, interpretation_set, strict=False),
                            *selected_interpretation_targets(case_slug, "verbosity_bias", interpretation_set, strict=False),
                        ]
                    else:
                        interpretation_targets = selected_interpretation_targets(
                            case_slug,
                            trait,
                            interpretation_set,
                            strict=False,
                        )
                    if not interpretation_targets:
                        continue
                    collapse_to_canonical_baseline = (
                        step not in {"content-plan", "attention-question"}
                        and output_trait == BASELINE_CONTROL_TRAIT_NAME
                        and output_role_culprit == "baseline"
                        and output_role_rival == "baseline"
                        and not (step == "style-plan" and trait in BASELINE_STYLE_PLAN_TRAITS)
                    )
                    if collapse_to_canonical_baseline:
                        interpretation_targets = [(CANONICAL_BASELINE_CONTENT_PLAN_FAMILY, CANONICAL_BASELINE_INTERPRETATION_SET_ID)]
                        rollout_targets = [CANONICAL_BASELINE_ROLLOUT_INDEX]
                    else:
                        rollout_targets = list(rollout_indices(effective_rollouts_per_combo))
                    canonical_side_order = canonical_ground_truth_side_order_for_variants(
                        output_trait,
                        output_speaker_variants["A"],
                        output_speaker_variants["B"],
                        output_side_order,
                    )
                    for family, interpretation_id in interpretation_targets:
                        for rollout_index in rollout_targets:
                            run_key = make_run_key(
                                case_slug,
                                output_trait,
                                output_role_culprit,
                                output_role_rival,
                                canonical_side_order,
                                family,
                                interpretation_id,
                                rollout_index,
                            )
                            if run_key in seen_keys:
                                continue
                            seen_keys.add(run_key)
                            runs.append((
                                case_label,
                                case_slug,
                                output_trait,
                                output_role_culprit,
                                output_role_rival,
                                canonical_side_order,
                                family,
                                interpretation_id,
                                rollout_index,
                            ))
    if not runs and invalid_anchoring_requests and trait_name == "anchoring_bias":
        first_culprit, first_rival, _first_order = invalid_anchoring_requests[0]
        raise ValueError(
            anchoring_invalid_request_message(
                trait_name=trait_name,
                culprit_variant=first_culprit,
                rival_variant=first_rival,
                ground_truth_side_order=ground_truth_side_order,
            )
        )
    return runs


def list_reference_matches(
    reference_output_dir: Path,
    *,
    case_id: str,
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
) -> None:
    print(f"[detective-v2] reference output: {reference_output_dir}")
    for subdir in ("plans", "style_plans", "dialogues", "evals", "accepted", "rar_evals"):
        matches = find_reference_matches(
            reference_output_dir,
            subdir=subdir,
            case_id=case_id,
            trait_name=trait_name,
            variant_a=variant_a,
            variant_b=variant_b,
            ground_truth_side_order=ground_truth_side_order,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
            rollout_index=rollout_index,
        )
        print(f"[detective-v2] {subdir}: {len(matches)} match(es)")
        for path in matches:
            print(f"  - {path}")


def has_existing_step_output(
    output_root: Path,
    *,
    subdir: str,
    case_id: str,
    trait_name: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
) -> Path | None:
    matches = find_reference_matches(
        output_root,
        subdir=subdir,
        case_id=case_id,
        trait_name=trait_name,
        variant_a=variant_a,
        variant_b=variant_b,
        ground_truth_side_order=ground_truth_side_order,
        content_plan_family=content_plan_family,
        interpretation_set_id=interpretation_set_id,
        rollout_index=rollout_index,
    )
    return matches[0] if matches else None


def find_baseline_control_dialogue_path(
    search_roots: list[Path],
    *,
    case_id: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
    expanded_traits: dict,
) -> Path:
    baseline_control_trait = select_baseline_control_trait(expanded_traits)
    content_plan_family, interpretation_set_id, rollout_index = canonical_baseline_identity()
    checked_roots: set[Path] = set()

    for root in search_roots:
        if root in checked_roots:
            continue
        checked_roots.add(root)

        accepted_matches = find_reference_matches(
            root,
            subdir="accepted",
            case_id=case_id,
            trait_name=baseline_control_trait,
            variant_a="baseline",
            variant_b="baseline",
            ground_truth_side_order=ground_truth_side_order,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
            rollout_index=rollout_index,
        )
        for accepted_path in accepted_matches:
            eval_path = find_passing_eval_for_dialogue(root, accepted_path)
            rar_path = find_passing_rar_for_dialogue(root, accepted_path)
            if eval_path is not None and rar_path is not None:
                print(
                    "[detective-v2] selected baseline control dialogue from accepted "
                    f"with passing bias/RaR evals: {accepted_path} "
                    f"(eval: {eval_path}; rar: {rar_path})"
                )
                return accepted_path

        dialogue_matches = find_reference_matches(
            root,
            subdir="dialogues",
            case_id=case_id,
            trait_name=baseline_control_trait,
            variant_a="baseline",
            variant_b="baseline",
            ground_truth_side_order=ground_truth_side_order,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
            rollout_index=rollout_index,
        )
        skipped_without_required_evals = 0
        for dialogue_path in dialogue_matches:
            eval_path = find_passing_eval_for_dialogue(root, dialogue_path)
            rar_path = find_passing_rar_for_dialogue(root, dialogue_path)
            if eval_path is not None and rar_path is not None:
                print(
                    "[detective-v2] selected baseline control dialogue from dialogues "
                    f"with passing bias/RaR evals: {dialogue_path} "
                    f"(eval: {eval_path}; rar: {rar_path})"
                )
                return dialogue_path
            skipped_without_required_evals += 1
        if skipped_without_required_evals:
            print(
                "[detective-v2] skipped "
                f"{skipped_without_required_evals} baseline control dialogue candidate(s) "
                "without required passing bias/RaR evals"
            )

    searched = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        "No baseline control dialogue found for anchored reuse. "
        f"case={case_id} order={ground_truth_side_order} family={content_plan_family} interp={interpretation_set_id} "
        f"rollout={rollout_index} "
        f"trait={baseline_control_trait} variant_pair=baseline/baseline "
        f"searched roots: {searched}. "
        "Only completed dialogues with matching passing bias-eval and RaR-eval are reusable."
    )


def configure_baseline_dialogue_reuse(
    config: dict,
    *,
    output_root_abs: Path,
    reference_output: Path | None,
    case_id: str,
    variant_a: str,
    variant_b: str,
    ground_truth_side_order: str,
    content_plan_family: str,
    interpretation_set_id: str,
    rollout_index: int = 1,
    expanded_traits: dict,
) -> Path | None:
    config["run"].pop("reuse_baseline_dialogue_path", None)
    fixed_speakers = fixed_baseline_speakers_for_variants(variant_a, variant_b)
    if not fixed_speakers:
        return None

    search_roots = [output_root_abs]
    if reference_output is not None:
        search_roots.append(reference_output)

    baseline_family, baseline_interpretation, baseline_rollout = canonical_baseline_identity()
    try:
        baseline_dialogue_path = find_baseline_control_dialogue_path(
            search_roots,
            case_id=case_id,
            ground_truth_side_order=ground_truth_side_order,
            content_plan_family=baseline_family,
            interpretation_set_id=baseline_interpretation,
            rollout_index=baseline_rollout,
            expanded_traits=expanded_traits,
        )
    except FileNotFoundError as exc:
        print(f"[detective-v2] baseline control dialogue unavailable; generating without fixed baseline reuse. {exc}")
        return None
    config["run"]["reuse_baseline_dialogue_path"] = pipeline_detective.repo_display_path(baseline_dialogue_path)
    config["run"].update(pipeline_detective.DetectivePipelineRunner.canonical_baseline_source_metadata(
        pipeline_detective.repo_display_path(baseline_dialogue_path)
    ))
    return baseline_dialogue_path


def is_baseline_control_run(trait_name: str, variant_a: str, variant_b: str) -> bool:
    return (
        trait_name == BASELINE_CONTROL_TRAIT_NAME
        and variant_a == "baseline"
        and variant_b == "baseline"
    )


def add_baseline_retry_feedback_to_style_bundle(style_bundle: dict, eval_feedback: str | None) -> None:
    if not eval_feedback:
        return
    rule = (
        "Baseline retry feedback from the previous failed full-dialogue bias evaluation: "
        f"{eval_feedback}"
    )
    for plan in style_bundle.values():
        operations = getattr(plan, "allowed_inference_operations", None)
        if isinstance(operations, list) and rule not in operations:
            operations.append(rule)


def generate_dialogue_without_prevalidation(
    *,
    runner: pipeline_detective.DetectivePipelineRunner,
    run_id: str,
    topic_cfg: dict,
    content_plan,
    style_bundle: dict,
    trait_name: str,
    speaker_variants: dict[str, str],
    base_metadata: dict | None = None,
    fixed_dialogue=None,
    fixed_speakers: set[str] | None = None,
    fixed_turn_ids: set[int] | None = None,
    cached_opening_turns: dict | None = None,
    treatment_metadata: dict | None = None,
) -> tuple[object | None, Path | None, dict | None, None]:
    fixed_speakers = set(fixed_speakers or set())
    fixed_turn_ids = set(fixed_turn_ids or set())
    print("[detective-v2] generating dialogue without pre-bias validation gates")
    dialogue = runner.generate_dialogue(
        topic_cfg=topic_cfg,
        content_plan=content_plan,
        style_bundle=style_bundle,
        trait_name=trait_name,
        speaker_variants=speaker_variants,
        eval_feedback=None,
        fixed_dialogue=fixed_dialogue,
        fixed_speakers=fixed_speakers,
        fixed_turn_ids=fixed_turn_ids,
        cached_opening_turns=cached_opening_turns,
        treatment_metadata=treatment_metadata,
    )
    metadata = dict(base_metadata or {})
    metadata.update(runner.dialogue_runtime_metadata())
    dialogue_path = runner.save_dialogue(
        dialogue,
        run_id,
        extra_metadata=metadata,
    )
    print(f"[detective-v2] dialogue saved to: {dialogue_path}")
    return dialogue, dialogue_path, metadata, None


def generate_baseline_dialogue_with_immediate_bias_eval(
    *,
    runner: pipeline_detective.DetectivePipelineRunner,
    run_id: str,
    style_path: Path,
    topic_cfg: dict,
    content_plan,
    speaker_variants: dict[str, str],
    base_metadata: dict | None = None,
) -> tuple[Path | None, Path | None]:
    max_attempts = int(runner.config["run"].get("max_attempts", 1))
    trait_name = runner.config["run"]["trait_name"]
    eval_feedback = None
    accepted_dialogue_path = None
    accepted_eval_path = None

    for attempt in range(1, max_attempts + 1):
        print(
            "[detective-v2] baseline control dialogue "
            f"attempt {attempt}/{max_attempts}: generating with bias-eval feedback"
        )
        style_bundle = runner.load_style_bundle_from_path(style_path)
        add_baseline_retry_feedback_to_style_bundle(style_bundle, eval_feedback)
        dialogue = runner.generate_dialogue(
            topic_cfg=topic_cfg,
            content_plan=content_plan,
            style_bundle=style_bundle,
            trait_name=trait_name,
            speaker_variants=speaker_variants,
            eval_feedback=eval_feedback,
        )
        dialogue_metadata = dict(base_metadata or {})
        dialogue_metadata.update(runner.dialogue_runtime_metadata())
        attempt_dialogue_path = runner.save_dialogue(
            dialogue,
            run_id,
            attempt=attempt,
            extra_metadata=dialogue_metadata,
        )
        print(f"[detective-v2] baseline attempt dialogue saved to: {attempt_dialogue_path}")

        eval_result = runner.evaluate_bias(
            dialogue=dialogue,
            content_plan=content_plan,
            topic_cfg=topic_cfg,
            trait_name=trait_name,
            speaker_variants=speaker_variants,
            include_computational_verbosity=False,
        )
        attempt_eval_path = runner.save_bias_eval(eval_result, run_id, attempt=attempt)
        print(f"[detective-v2] baseline attempt bias eval saved to: {attempt_eval_path}")

        if eval_payload_passed(eval_result.model_dump()):
            accepted_dialogue_path = runner.save_dialogue(
                dialogue,
                run_id,
                extra_metadata=dialogue_metadata,
            )
            accepted_eval_path = runner.save_bias_eval(eval_result, run_id)
            cleanup_redundant_attempt_outputs_for_run(runner.output_root(), run_id)
            print(
                "[detective-v2] baseline control dialogue passed immediate bias eval "
                f"and was saved to: {accepted_dialogue_path}"
            )
            break

        eval_feedback = pipeline_detective.build_baseline_bias_retry_feedback(eval_result)
        print(
            "[detective-v2] baseline control dialogue failed immediate bias eval; "
            "regenerating with evaluator feedback"
        )

    if accepted_dialogue_path is None:
        raise RuntimeError(
            "No baseline control dialogue passed bias-eval after "
            f"{max_attempts} attempt(s): run_id={run_id} variants={speaker_variants}"
        )
    return accepted_dialogue_path, accepted_eval_path


def build_base_config(args) -> tuple[dict, str, Path]:
    base_config = yaml.safe_load(CONFIG_PATH.read_text())
    reference_output_root = resolve_output_root(args.reference_output)

    if args.step != "full" and reference_output_root is not None:
        try:
            output_root_rel = str(reference_output_root.relative_to(ROOT))
        except ValueError:
            output_root_rel = str(reference_output_root)
    elif args.resume_from:
        output_root_rel = f"outputs/{args.resume_from}"
        date_tag = args.resume_from.removeprefix("outputs/").replace("_detective_v2", "")
    else:
        date_tag = datetime.now().strftime("%m%d")
        output_root_rel = f"outputs/{date_tag}_detective_v2"
    output_root_abs = Path(output_root_rel)
    if not output_root_abs.is_absolute():
        output_root_abs = ROOT / output_root_rel
    base_config["run"]["output_root"] = output_root_rel
    base_config["run"].pop("reuse_plan_path", None)
    base_config["run"].pop("reuse_style_plan_path", None)
    base_config["run"].pop("reuse_baseline_dialogue_path", None)
    return base_config, output_root_rel, output_root_abs


def run_count_summary(runs: list[StepRun]) -> dict[str, int]:
    baseline = sum(1 for run in runs if run[2] == BASELINE_CONTROL_TRAIT_NAME)
    anchoring = sum(
        1
        for run in runs
        if run[2] == "anchoring_bias" and (run[3], run[4]) != ROLE_BASELINE_PAIR
    )
    non_anchoring_bias = sum(
        1
        for run in runs
        if run[2] not in {BASELINE_CONTROL_TRAIT_NAME, "anchoring_bias"}
    )
    return {
        "baseline": baseline,
        "non_anchoring_bias": non_anchoring_bias,
        "anchoring": anchoring,
        "total": len(runs),
    }


def print_run_count_summary(runs: list[StepRun]) -> None:
    summary = run_count_summary(runs)
    orders = ",".join(sorted({run[5] for run in runs}))
    print(
        "[detective-v2] dynamic run-count summary: "
        f"baseline={summary['baseline']} "
        f"non_anchoring_bias={summary['non_anchoring_bias']} "
        f"anchoring={summary['anchoring']} "
        f"total={summary['total']} "
        f"orders={orders}"
    )


def run_full_schedule(args, expanded_traits: dict, base_config: dict, output_root_rel: str, output_root_abs: Path) -> None:
    if args.smoke_test:
        runs = build_smoke_test_runs(
            expanded_traits=expanded_traits,
            case_id=args.case,
            trait_name=args.trait,
            variant_a=args.variant_a or "active",
            variant_b=args.variant_b or "baseline",
            ground_truth_side_order=args.ground_truth_side_order,
            interpretation_set=args.interpretation_set,
            rollouts_per_combo=args.rollouts_per_combo,
        )
        expected_total = len(runs)
        if not runs:
            print("[detective-v2] smoke test mode: scheduled 0 run(s)")
        else:
            print(
                f"[detective-v2] smoke test mode: scheduled {len(runs)} run(s) "
                f"(case={runs[0][1]} trait={runs[0][2]} A={runs[0][3]} B={runs[0][4]} "
                f"orders={','.join(sorted({run[5] for run in runs}))})"
            )
    else:
        runs = build_step_runs(
            expanded_traits,
            case_id=args.case,
            trait_name=args.trait,
            variant_a=args.variant_a,
            variant_b=args.variant_b,
            ground_truth_side_order=args.ground_truth_side_order,
            interpretation_set=args.interpretation_set,
            rollouts_per_combo=args.rollouts_per_combo,
        )
        side_orders = selected_ground_truth_side_orders(args.ground_truth_side_order)
        raw_total = len(runs)
        expected_total = len(runs)
        print(
            f"[detective-v2] scheduled {len(runs)} runs "
            f"({raw_total} valid matrix combinations after order-specific filters; "
            f"requested orders={','.join(side_orders)})"
        )

    print(f"[detective-v2] outputs -> {output_root_rel}")
    print(f"[detective-v2] expanded biases: {', '.join(expanded_traits)}")
    print_run_count_summary(runs)

    if len(runs) != expected_total:
        raise RuntimeError(f"Run count mismatch: expected {expected_total}, got {len(runs)}")

    completed_runs = load_completed_runs(output_root_abs)
    existing_plan_paths = load_existing_plan_paths(output_root_abs)
    existing_style_paths = load_existing_style_paths(output_root_abs)
    pending_runs = [
        run for run in runs
        if make_run_key(run[1], run[2], run[3], run[4], run[5], run[6], run[7], run[8]) not in completed_runs
    ]

    if args.resume_from:
        print(f"[detective-v2] resume mode from -> {output_root_rel}")
        print(f"[detective-v2] accepted runs already present -> {len(completed_runs)}")
        print(f"[detective-v2] runs remaining -> {len(pending_runs)}")

    if args.dry_run:
        for i, (case_label, case_id, trait_name, variant_a, variant_b, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index) in enumerate(pending_runs, 1):
            print(
                f"{i:03d}. case={case_id} ({case_label}) trait={trait_name} "
                f"A={variant_a} B={variant_b} order={ground_truth_side_order} "
                f"family={content_plan_family} interp={interpretation_set_id} rollout={rollout_index}"
            )
        return

    plan_paths: dict[tuple[str, str, str, str], str] = dict(existing_plan_paths)
    style_paths: dict[tuple[str, str, str, str], str] = dict(existing_style_paths)
    for i, (case_label, case_id, trait_name, variant_a, variant_b, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index) in enumerate(pending_runs, 1):
        print(
            f"[detective-v2] run {i}/{len(pending_runs)} | "
            f"case={case_id} ({case_label}) | trait={trait_name} | "
            f"A={variant_a} B={variant_b} | order={ground_truth_side_order} | "
            f"family={content_plan_family} interp={interpretation_set_id} rollout={rollout_index}"
        )
        config = copy.deepcopy(base_config)
        config["run"]["case_id"] = case_id
        config["run"]["trait_name"] = trait_name
        config["run"]["variant_name_a"] = variant_a
        config["run"]["variant_name_b"] = variant_b
        config["run"]["ground_truth_side_order"] = ground_truth_side_order
        config["run"]["content_plan_family"] = content_plan_family
        config["run"]["interpretation_set_id"] = interpretation_set_id
        config["run"]["rollout_index"] = rollout_index
        configure_baseline_dialogue_reuse(
            config,
            output_root_abs=output_root_abs,
            reference_output=None,
            case_id=case_id,
            variant_a=variant_a,
            variant_b=variant_b,
            ground_truth_side_order=ground_truth_side_order,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
            rollout_index=rollout_index,
            expanded_traits=expanded_traits,
        )
        plan_key = (case_id, ground_truth_side_order, content_plan_family, interpretation_set_id)
        if plan_key in plan_paths:
            config["run"]["reuse_plan_path"] = plan_paths[plan_key]
        style_key = (case_id, canonical_style_plan_trait_name(trait_name), content_plan_family, interpretation_set_id)
        if style_key in style_paths:
            config["run"]["reuse_style_plan_path"] = style_paths[style_key]
        result = pipeline_detective.DetectivePipelineRunner(config=config).run()
        if result["accepted"]:
            cleanup_redundant_attempt_outputs_for_run(output_root_abs, result["run_id"])
        if plan_key not in plan_paths:
            for order in GROUND_TRUTH_SIDE_ORDER_CHOICES:
                plan_paths[(case_id, order, content_plan_family, interpretation_set_id)] = result["plan_path"]
            print(
                f"[detective-v2] pairwise plan saved for {case_id} "
                f"family={content_plan_family} interp={interpretation_set_id} (all GT orders) "
                f"-> {result['plan_path']}"
            )
        if style_key not in style_paths and result.get("style_path"):
            style_paths[style_key] = result["style_path"]
        status = "accepted" if result["accepted"] else "REJECTED"
        print(f"[detective-v2] run {i}/{len(pending_runs)} {status}\n")

    print(f"[detective-v2] complete. outputs in {output_root_rel}")
    maybe_cleanup_redundant_attempt_outputs(
        output_root_abs,
        keep_attempt_files=args.keep_attempt_files,
    )


def run_single_step(args, expanded_traits: dict, base_config: dict, output_root_abs: Path) -> None:
    if args.step in {"content-plan", "attention-question", "style-plan", "role-dialogue", "dialogue"}:
        reference_output = resolve_output_root(args.reference_output)
        if args.step == "dialogue" and (args.variant_a is not None or args.variant_b is not None):
            speaker_step_runs = build_step_runs(
                expanded_traits,
                case_id=args.case,
                trait_name=args.trait,
                variant_a=args.variant_a,
                variant_b=args.variant_b,
                ground_truth_side_order=args.ground_truth_side_order,
                interpretation_set=args.interpretation_set,
                rollouts_per_combo=args.rollouts_per_combo,
            )
            step_runs = [role_step_run_from_speaker_step_run(run) for run in speaker_step_runs]
            if args.culprit_variant is not None or args.rival_variant is not None:
                before_filter = len(step_runs)
                step_runs = [
                    run
                    for run in step_runs
                    if (args.culprit_variant is None or run[3] == args.culprit_variant)
                    and (args.rival_variant is None or run[4] == args.rival_variant)
                ]
                if not step_runs and before_filter:
                    raise ValueError("Contradictory culprit/rival filters and --variant-a/--variant-b assignment.")
        elif args.variant_a is not None or args.variant_b is not None:
            normalized_culprit, normalized_rival = normalize_role_variants_from_args(args)
            role_culprit_filter: str | None = normalized_culprit
            role_rival_filter: str | None = normalized_rival
            step_runs = build_role_stage_step_runs(
                expanded_traits,
                step=args.step,
                case_id=args.case,
                trait_name=args.trait,
                culprit_variant=role_culprit_filter,
                rival_variant=role_rival_filter,
                ground_truth_side_order=args.ground_truth_side_order,
                interpretation_set=args.interpretation_set,
                rollouts_per_combo=args.rollouts_per_combo,
            )
        else:
            role_culprit_filter = args.culprit_variant
            role_rival_filter = args.rival_variant
            step_runs = build_role_stage_step_runs(
                expanded_traits,
                step=args.step,
                case_id=args.case,
                trait_name=args.trait,
                culprit_variant=role_culprit_filter,
                rival_variant=role_rival_filter,
                ground_truth_side_order=args.ground_truth_side_order,
                interpretation_set=args.interpretation_set,
                rollouts_per_combo=args.rollouts_per_combo,
            )
        if not step_runs:
            print(f"[detective-v2] no matching targets for step: {args.step}")
            return

        if args.dry_run:
            print(f"[detective-v2] running step: {args.step}")
            print(f"[detective-v2] step targets: {len(step_runs)}")
            for i, (_, case_id, trait_name, culprit_variant, rival_variant, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index) in enumerate(step_runs, 1):
                print(
                    f"{i:03d}. case={case_id} trait={trait_name} "
                    f"culprit={culprit_variant} rival={rival_variant} order={ground_truth_side_order} "
                    f"family={content_plan_family} interp={interpretation_set_id} rollout={rollout_index}"
                )
            return

        if args.list_reference_matches:
            if reference_output is None:
                raise ValueError("--reference-output is required with --list-reference-matches")
            for _, case_id, trait_name, culprit_variant, rival_variant, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index in step_runs:
                if args.step in {"content-plan", "attention-question"}:
                    print(find_content_plan_file(
                        reference_output,
                        case_id=case_id,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                    ))
                elif args.step == "style-plan":
                    print(find_style_plan_file(
                        reference_output,
                        case_id=case_id,
                        trait_name=trait_name,
                        culprit_variant=culprit_variant,
                        rival_variant=rival_variant,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                    ))
                elif args.step == "role-dialogue":
                    print(find_role_dialogue_file(
                        reference_output,
                        case_id=case_id,
                        trait_name=trait_name,
                        culprit_variant=culprit_variant,
                        rival_variant=rival_variant,
                        rollout_index=rollout_index,
                        seed=args.seed,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                    ))
                elif args.step == "dialogue":
                    print(find_dialogue_file(
                        reference_output,
                        case_id=case_id,
                        trait_name=trait_name,
                        culprit_variant=culprit_variant,
                        rival_variant=rival_variant,
                        ground_truth_side_order=ground_truth_side_order,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                        rollout_index=rollout_index,
                        seed=args.seed,
                    ))
            return

        print(f"[detective-v2] running step: {args.step}")
        print(f"[detective-v2] step targets: {len(step_runs)}")
        if reference_output is not None:
            print(f"[detective-v2] reference output: {reference_output}")

        dialogue_failed_count = 0
        for i, (case_label, case_id, trait_name, culprit_variant, rival_variant, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index) in enumerate(step_runs, 1):
            role_variants = {"culprit": culprit_variant, "rival": rival_variant}
            speaker_variants = pipeline_detective.DetectivePipelineRunner.speaker_variants_from_roles(
                role_variants,
                ground_truth_side_order,
            )
            print(
                f"[detective-v2] step {i}/{len(step_runs)} | "
                f"case={case_id} trait={trait_name} culprit={culprit_variant} rival={rival_variant} "
                f"order={ground_truth_side_order} family={content_plan_family} interp={interpretation_set_id} rollout={rollout_index}"
            )
            run_id = make_run_id()
            config = copy.deepcopy(base_config)
            config["run"]["case_id"] = case_id
            config["run"]["trait_name"] = trait_name
            config["run"]["culprit_variant"] = culprit_variant
            config["run"]["rival_variant"] = rival_variant
            config["run"]["variant_name_a"] = speaker_variants["A"]
            config["run"]["variant_name_b"] = speaker_variants["B"]
            config["run"]["ground_truth_side_order"] = ground_truth_side_order
            config["run"]["seed"] = args.seed
            config["run"]["content_plan_family"] = content_plan_family
            config["run"]["interpretation_set_id"] = interpretation_set_id
            config["run"]["rollout_index"] = rollout_index
            runner = pipeline_detective.DetectivePipelineRunner(config=config)
            if args.step == "dialogue":
                runner.enable_dialogue_api_logging()
            topic_cfg = runner.load_topic_cfg()
            if args.step == "dialogue":
                configure_baseline_dialogue_reuse(
                    config,
                    output_root_abs=output_root_abs,
                    reference_output=reference_output,
                    case_id=case_id,
                    variant_a=speaker_variants["A"],
                    variant_b=speaker_variants["B"],
                    ground_truth_side_order=ground_truth_side_order,
                    content_plan_family=content_plan_family,
                    interpretation_set_id=interpretation_set_id,
                    rollout_index=rollout_index,
                    expanded_traits=expanded_traits,
                )

            if args.step == "content-plan":
                canonical_path = runner.output_root() / "plans" / (
                    f"{case_id}__{pipeline_detective.safe_artifact_part(content_plan_family)}__"
                    f"{pipeline_detective.safe_artifact_part(interpretation_set_id)}.json"
                )
                if canonical_path.exists() and not args.force_content_plan:
                    existing_plan = runner.load_content_plan_from_path(canonical_path)
                    runner.validate_content_plan_pair(existing_plan, topic_cfg)
                    print(f"[detective-v2] content plan already exists, reusing: {canonical_path}")
                    continue
                _, plan_path = runner.generate_content_plan(
                    topic_cfg,
                    run_id,
                    rival_suspect=args.rival_suspect,
                )
                print(f"[detective-v2] content plan saved to: {plan_path}")
                continue

            if reference_output is None and args.step in {"style-plan"}:
                reference_output_for_plan = runner.output_root()
            elif reference_output is None:
                raise ValueError(f"--reference-output is required for step '{args.step}'")
            else:
                reference_output_for_plan = reference_output

            plan_path = find_content_plan_file(
                reference_output_for_plan,
                case_id=case_id,
                content_plan_family=content_plan_family,
                interpretation_set_id=interpretation_set_id,
            )
            print(f"[detective-v2] loading content plan from: {plan_path}")
            config["run"]["content_plan_source"] = pipeline_detective.repo_display_path(plan_path)
            runner.config["run"]["content_plan_source"] = config["run"]["content_plan_source"]
            content_plan = runner.load_content_plan_from_path(plan_path)
            runner.validate_content_plan_pair(content_plan, topic_cfg)
            if content_plan.content_plan_version != pipeline_detective.PAIRWISE_ROLE_CONTENT_PLAN_VERSION:
                raise ValueError(
                    f"Content plan {plan_path} has version {content_plan.content_plan_version!r}; "
                    f"expected {pipeline_detective.PAIRWISE_ROLE_CONTENT_PLAN_VERSION!r} for the role-based pairwise path."
                )

            if args.step == "attention-question":
                updated_path, did_update = runner.backfill_content_attention_question(
                    plan_path,
                    topic_cfg,
                    force=args.force_attention_question,
                )
                action = "updated" if did_update else "already present, skipped"
                print(f"[detective-v2] content attention question {action}: {updated_path}")
                continue

            if args.step == "style-plan":
                try:
                    existing_style = find_style_plan_file(
                        runner.output_root(),
                        case_id=case_id,
                        trait_name=trait_name,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                    )
                except FileNotFoundError:
                    existing_style = None
                if existing_style is not None:
                    try:
                        existing_bundle = runner.load_style_bundle_from_path(existing_style)
                        runner.validate_api_style_bundle(existing_bundle, content_plan=content_plan, trait_name=trait_name)
                        print(f"[detective-v2] validated style plan already exists, reusing: {existing_style}")
                        continue
                    except ValueError as exc:
                        print(f"[detective-v2] existing style plan is incompatible, regenerating: {exc}")
                style_bundle = runner.generate_role_style_bundle(
                    content_plan=content_plan,
                    trait_name=trait_name,
                )
                style_path = runner.save_style_bundle(style_bundle, run_id)
                print(f"[detective-v2] style plans saved to: {style_path}")
                continue

            if args.step in {"role-dialogue", "dialogue"}:
                try:
                    if args.step == "dialogue" and not args.force_dialogue:
                        try:
                            existing_dialogue = find_dialogue_file(
                                runner.output_root(),
                                case_id=case_id,
                                trait_name=trait_name,
                                culprit_variant=culprit_variant,
                                rival_variant=rival_variant,
                                ground_truth_side_order=ground_truth_side_order,
                                seed=args.seed,
                                content_plan_family=content_plan_family,
                                interpretation_set_id=interpretation_set_id,
                                rollout_index=rollout_index,
                            )
                        except FileNotFoundError:
                            existing_dialogue = None
                        if existing_dialogue is not None:
                            print(f"[detective-v2] dialogue already exists, skipping: {existing_dialogue}")
                            continue
                    role_dialogue_path = runner.role_dialogue_path(
                        trait_name=trait_name,
                        culprit_variant=culprit_variant,
                        rival_variant=rival_variant,
                        seed=args.seed,
                    )
                    if role_dialogue_path.exists() and not args.force_role_dialogue:
                        print(f"[detective-v2] loading canonical role dialogue: {role_dialogue_path}")
                        role_dialogue = runner.load_role_dialogue_from_path(role_dialogue_path)
                        runner.validate_role_dialogue_pair(role_dialogue, content_plan)
                    else:
                        action = "regenerating" if role_dialogue_path.exists() else "generating all role turns"
                        print(f"[detective-v2] canonical role dialogue {action}: {role_dialogue_path}")
                        style_path = find_style_plan_file(
                            reference_output,
                            case_id=case_id,
                            trait_name=trait_name,
                            content_plan_family=content_plan_family,
                            interpretation_set_id=interpretation_set_id,
                        )
                        print(f"[detective-v2] loading style plans from: {style_path}")
                        role_style_bundle = runner.load_style_bundle_from_path(style_path)
                        runner.validate_api_style_bundle(role_style_bundle, content_plan=content_plan, trait_name=trait_name)
                        role_style_bundle = runner.style_bundle_for_role_variants(role_style_bundle, role_variants)
                        role_dialogue, role_dialogue_path = runner.generate_role_dialogue(
                            topic_cfg=topic_cfg,
                            content_plan=content_plan,
                            role_style_bundle=role_style_bundle,
                            trait_name=trait_name,
                            run_id=run_id,
                            content_plan_source_path=runner.repo_display_path(plan_path),
                            style_plan_source_path=runner.repo_display_path(style_path),
                        )
                        print(f"[detective-v2] canonical role dialogue saved to: {role_dialogue_path}")
                    if args.step == "role-dialogue":
                        continue

                    print(f"[detective-v2] materializing GT order without generation: {ground_truth_side_order}")
                    materialized_plan, materialization_metadata = runner.materialize_content_plan_for_dialogue(
                        content_plan,
                        ground_truth_side_order=ground_truth_side_order,
                    )
                    dialogue = runner.materialize_role_dialogue(
                        role_dialogue,
                        ground_truth_side_order=ground_truth_side_order,
                    )
                    dialogue_extra_metadata = runner.content_plan_dialogue_metadata(content_plan)
                    dialogue_extra_metadata.update({
                        "culprit_name": materialized_plan.debate_setup["culprit_name"],
                        "rival_suspect_name": materialized_plan.debate_setup["rival_suspect_name"],
                        "speaker_role_map": materialization_metadata["speaker_role_map"],
                        "variant_name_a": speaker_variants["A"],
                        "variant_name_b": speaker_variants["B"],
                        "seed": args.seed,
                        "role_dialogue_source": runner.repo_display_path(role_dialogue_path),
                    })
                    baseline_source_role_dialogue = (
                        role_dialogue.get("baseline_source_role_dialogue")
                        if isinstance(role_dialogue, dict)
                        else getattr(role_dialogue, "baseline_source_role_dialogue", None)
                    )
                    if baseline_source_role_dialogue:
                        dialogue_extra_metadata["baseline_source_role_dialogue"] = baseline_source_role_dialogue
                        dialogue_extra_metadata.update(
                            pipeline_detective.DetectivePipelineRunner.canonical_baseline_source_metadata(
                                baseline_source_role_dialogue
                            )
                        )
                    elif is_baseline_control_run(trait_name, speaker_variants["A"], speaker_variants["B"]):
                        dialogue_extra_metadata.update(
                            pipeline_detective.DetectivePipelineRunner.canonical_baseline_source_metadata(
                                runner.repo_display_path(role_dialogue_path)
                            )
                        )
                    dialogue_path = runner.save_dialogue(
                        dialogue,
                        run_id,
                        extra_metadata=dialogue_extra_metadata,
                    )
                    print(f"[detective-v2] dialogue generated: {dialogue_path}")
                    continue
                except pipeline_detective.AdaptiveTurnGenerationError as exc:
                    if args.step != "dialogue":
                        raise
                    _failure_record, failure_log = record_dialogue_generation_failure(
                        output_root=output_root_abs,
                        case_label=case_label,
                        case_id=case_id,
                        trait_name=trait_name,
                        culprit_variant=culprit_variant,
                        rival_variant=rival_variant,
                        ground_truth_side_order=ground_truth_side_order,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                        rollout_index=rollout_index,
                        seed=args.seed,
                        step_index=i,
                        step_total=len(step_runs),
                        exc=exc,
                    )
                    dialogue_failed_count += 1
                    print("[detective-v2] dialogue target failed; recording and continuing:")
                    print(f"case={case_id}")
                    print(f"trait={trait_name}")
                    print(f"culprit={culprit_variant} rival={rival_variant} order={ground_truth_side_order} rollout={rollout_index} seed={args.seed}")
                    print(f"step={i}/{len(step_runs)}")
                    print(f"error={type(exc).__name__}: {exc}")
                    print(f"failure_log={failure_log}")
                    continue

        if args.step == "dialogue":
            failure_log = dialogue_generation_failure_log_path(output_root_abs)
            print("[detective-v2] dialogue sweep complete:")
            print(f"successful/skipped: {len(step_runs) - dialogue_failed_count}")
            print(f"failed: {dialogue_failed_count}")
            print(f"failure log: {failure_log}")
            maybe_cleanup_redundant_attempt_outputs(
                output_root_abs,
                keep_attempt_files=args.keep_attempt_files,
            )
        return

    validate_trait_variant_pair(args.trait, args.variant_a, args.variant_b)
    reference_output = resolve_output_root(args.reference_output)
    reference_paths_by_run: dict[StepRun, Path] = {}
    if args.step == "content-plan":
        step_runs = [
            (
                label,
                slug,
                BASELINE_CONTROL_TRAIT_NAME,
                "baseline",
                "baseline",
                "gt_first",
                family,
                interpretation_id,
                1,
            )
            for label, slug in CASE_SPECS
            if args.case is None or slug == args.case
            for family, interpretation_id in selected_interpretation_targets(
                slug,
                BASELINE_CONTROL_TRAIT_NAME,
                args.interpretation_set,
            )
        ]
        if args.case is not None and not step_runs:
            raise ValueError(f"Unknown case_id '{args.case}'. Expected one of: {', '.join(slug for _, slug in CASE_SPECS)}")
    elif args.step == "bias-eval":
        if reference_output is None:
            raise ValueError("--reference-output is required for step 'bias-eval'")
        step_targets = build_reference_step_targets(
            reference_output,
            subdirs=("dialogues",),
            case_id=args.case,
            trait_name=args.trait,
            variant_a=args.variant_a,
            variant_b=args.variant_b,
            ground_truth_side_order=args.ground_truth_side_order,
            interpretation_set=args.interpretation_set,
        )
        original_count = len(step_targets)
        step_targets = [
            (run, path)
            for run, path in step_targets
            if trait_enabled_for_variants(run[2], run[3], run[4], expanded_traits)
        ]
        skipped_disabled = original_count - len(step_targets)
        if skipped_disabled:
            print(
                f"[detective-v2] skipping {skipped_disabled} target(s) for disabled traits"
            )
        pre_skip_count = len(step_targets)
        step_targets = [
            (run, path)
            for run, path in step_targets
            if not pipeline_detective.skip_bias_eval_for_trait(run[2])
        ]
        skipped_no_bias_eval = pre_skip_count - len(step_targets)
        if skipped_no_bias_eval:
            print(
                "[detective-v2] skipping "
                f"{skipped_no_bias_eval} target(s) whose trait no longer uses bias-eval"
            )
        step_runs = [run for run, _path in step_targets]
        reference_paths_by_run = {run: path for run, path in step_targets}
    elif args.step == "rar-eval":
        if reference_output is None:
            raise ValueError("--reference-output is required for step 'rar-eval'")
        step_targets = build_reference_step_targets(
            reference_output,
            subdirs=("accepted", "dialogues"),
            case_id=args.case,
            trait_name=args.trait,
            variant_a=args.variant_a,
            variant_b=args.variant_b,
            ground_truth_side_order=args.ground_truth_side_order,
            interpretation_set=args.interpretation_set,
        )
        step_runs = [run for run, _path in step_targets]
        reference_paths_by_run = {run: path for run, path in step_targets}
    else:
        step_runs = build_step_runs(
            expanded_traits,
            case_id=args.case,
            trait_name=args.trait,
            variant_a=args.variant_a,
            variant_b=args.variant_b,
            ground_truth_side_order=args.ground_truth_side_order,
            interpretation_set=args.interpretation_set,
        )
        if args.step == "attention-question":
            deduped_step_runs = []
            seen_plan_keys = set()
            for run in step_runs:
                _case_label, case_id, _trait_name, _variant_a, _variant_b, ground_truth_side_order, content_plan_family, interpretation_set_id, _rollout_index = run
                plan_key = (case_id, ground_truth_side_order, content_plan_family, interpretation_set_id)
                if plan_key in seen_plan_keys:
                    continue
                seen_plan_keys.add(plan_key)
                deduped_step_runs.append(run)
            step_runs = deduped_step_runs

    if not step_runs:
        print(f"[detective-v2] no matching targets for step: {args.step}")
        return

    if args.step == "bias-eval":
        original_count = len(step_runs)
        if not args.force_bias_eval:
            step_runs = [
                run for run in step_runs
                if has_existing_step_output(
                    output_root_abs,
                    subdir="evals",
                    case_id=run[1],
                    trait_name=run[2],
                    variant_a=run[3],
                    variant_b=run[4],
                    ground_truth_side_order=run[5],
                    content_plan_family=run[6],
                    interpretation_set_id=run[7],
                    rollout_index=run[8],
                ) is None
            ]
            skipped_existing = original_count - len(step_runs)
            if skipped_existing:
                print(f"[detective-v2] skipping {skipped_existing} target(s) with existing bias evals")
        else:
            print(f"[detective-v2] force bias-eval enabled; not skipping existing evals")
        pre_skip_count = len(step_runs)
        step_runs = [
            run for run in step_runs
            if not pipeline_detective.skip_bias_eval_for_trait(run[2])
        ]
        skipped_no_bias_eval = pre_skip_count - len(step_runs)
        if skipped_no_bias_eval:
            print(
                "[detective-v2] skipping "
                f"{skipped_no_bias_eval} target(s) whose trait no longer uses bias-eval"
            )

    elif args.step == "rar-eval":
        original_count = len(step_runs)
        step_runs = [
            run for run in step_runs
            if has_existing_step_output(
                output_root_abs,
                subdir="rar_evals",
                case_id=run[1],
                trait_name=run[2],
                variant_a=run[3],
                variant_b=run[4],
                ground_truth_side_order=run[5],
                content_plan_family=run[6],
                interpretation_set_id=run[7],
                rollout_index=run[8],
            ) is None
        ]
        skipped = original_count - len(step_runs)
        if skipped:
            print(f"[detective-v2] skipping {skipped} target(s) with existing RaR evals")

    if not step_runs:
        print(f"[detective-v2] no remaining targets for step: {args.step}")
        return

    if args.dry_run:
        print(f"[detective-v2] running step: {args.step}")
        print(f"[detective-v2] step targets: {len(step_runs)}")
        for i, (_, case_id, trait_name, variant_a, variant_b, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index) in enumerate(step_runs, 1):
            print(
                f"{i:03d}. case={case_id} trait={trait_name} "
                f"A={variant_a} B={variant_b} order={ground_truth_side_order} "
                f"family={content_plan_family} interp={interpretation_set_id} rollout={rollout_index}"
            )
        return

    if args.list_reference_matches:
        if reference_output is None:
            raise ValueError("--reference-output is required with --list-reference-matches")
        for _, case_id, trait_name, variant_a, variant_b, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index in step_runs:
            list_reference_matches(
                reference_output,
                case_id=case_id,
                trait_name=trait_name,
                variant_a=variant_a,
                variant_b=variant_b,
                ground_truth_side_order=ground_truth_side_order,
                content_plan_family=content_plan_family,
                interpretation_set_id=interpretation_set_id,
                rollout_index=rollout_index,
            )
        return

    print(f"[detective-v2] running step: {args.step}")
    print(f"[detective-v2] step targets: {len(step_runs)}")
    if reference_output is not None:
        print(f"[detective-v2] reference output: {reference_output}")

    for i, (case_label, case_id, trait_name, variant_a, variant_b, ground_truth_side_order, content_plan_family, interpretation_set_id, rollout_index) in enumerate(step_runs, 1):
        print(
            f"[detective-v2] step {i}/{len(step_runs)} | "
            f"case={case_id} trait={trait_name} A={variant_a} B={variant_b} "
            f"order={ground_truth_side_order} family={content_plan_family} interp={interpretation_set_id} rollout={rollout_index}"
        )
        run_tuple: StepRun = (
            case_label,
            case_id,
            trait_name,
            variant_a,
            variant_b,
            ground_truth_side_order,
            content_plan_family,
            interpretation_set_id,
            rollout_index,
        )
        run_id = make_run_id()
        config = copy.deepcopy(base_config)
        config["run"]["case_id"] = case_id
        config["run"]["trait_name"] = trait_name
        config["run"]["variant_name_a"] = variant_a
        config["run"]["variant_name_b"] = variant_b
        config["run"]["ground_truth_side_order"] = ground_truth_side_order
        config["run"]["content_plan_family"] = content_plan_family
        config["run"]["interpretation_set_id"] = interpretation_set_id
        config["run"]["rollout_index"] = rollout_index
        runner = pipeline_detective.DetectivePipelineRunner(config=config)
        topic_cfg = runner.load_topic_cfg()
        speaker_variants = runner.speaker_variants()
        if args.step == "dialogue":
            configure_baseline_dialogue_reuse(
                config,
                output_root_abs=output_root_abs,
                reference_output=reference_output,
                case_id=case_id,
                variant_a=variant_a,
                variant_b=variant_b,
                ground_truth_side_order=ground_truth_side_order,
                content_plan_family=content_plan_family,
                interpretation_set_id=interpretation_set_id,
                rollout_index=rollout_index,
                expanded_traits=expanded_traits,
            )

        if args.step == "content-plan":
            _, plan_path = runner.generate_content_plan(
                topic_cfg,
                run_id,
                rival_suspect=args.rival_suspect,
            )
            print(f"[detective-v2] content plan saved to: {plan_path}")
            continue

        if reference_output is None:
            raise ValueError(f"--reference-output is required for step '{args.step}'")

        plan_path = find_reference_file(
            reference_output,
            subdir="plans",
            case_id=case_id,
            trait_name=trait_name,
            variant_a=variant_a,
            variant_b=variant_b,
            ground_truth_side_order=ground_truth_side_order,
            content_plan_family=content_plan_family,
            interpretation_set_id=interpretation_set_id,
            rollout_index=rollout_index,
        )
        print(f"[detective-v2] loading content plan from: {plan_path}")
        content_plan = runner.load_content_plan_from_path(plan_path)
        runner.validate_content_plan_pair(content_plan, topic_cfg)
        content_plan_metadata = runner.content_plan_dialogue_metadata(content_plan)

        if args.step == "attention-question":
            updated_path, did_update = runner.backfill_content_attention_question(
                plan_path,
                topic_cfg,
                force=args.force_attention_question,
            )
            action = "updated" if did_update else "already present, skipped"
            print(f"[detective-v2] content attention question {action}: {updated_path}")
            continue

        if args.step == "style-plan":
            style_bundle = runner.generate_role_style_bundle(
                content_plan=content_plan,
                trait_name=trait_name,
            )
            style_path = runner.save_style_bundle(style_bundle, run_id)
            print(f"[detective-v2] style plans saved to: {style_path}")
            continue

        if args.step == "dialogue":
            if is_baseline_control_run(trait_name, variant_a, variant_b):
                try:
                    existing_dialogue = find_baseline_control_dialogue_path(
                        [output_root_abs],
                        case_id=case_id,
                        ground_truth_side_order=ground_truth_side_order,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                        rollout_index=rollout_index,
                        expanded_traits=expanded_traits,
                    )
                except FileNotFoundError:
                    existing_dialogue = None
            else:
                existing_dialogue = has_existing_step_output(
                    output_root_abs,
                    subdir="dialogues",
                    case_id=case_id,
                    trait_name=trait_name,
                    variant_a=variant_a,
                    variant_b=variant_b,
                    ground_truth_side_order=ground_truth_side_order,
                    content_plan_family=content_plan_family,
                    interpretation_set_id=interpretation_set_id,
                    rollout_index=rollout_index,
                )
            if existing_dialogue is not None:
                print(f"[detective-v2] dialogue already exists, skipping: {existing_dialogue}")
                continue

            style_path = find_reference_file(
                reference_output,
                subdir="style_plans",
                case_id=case_id,
                trait_name=trait_name,
                variant_a=variant_a,
                variant_b=variant_b,
                ground_truth_side_order=ground_truth_side_order,
                content_plan_family=content_plan_family,
                interpretation_set_id=interpretation_set_id,
                rollout_index=rollout_index,
            )
            print(f"[detective-v2] loading style plans from: {style_path}")
            if is_baseline_control_run(trait_name, variant_a, variant_b):
                dialogue_path, eval_path = generate_baseline_dialogue_with_immediate_bias_eval(
                    runner=runner,
                    run_id=run_id,
                    style_path=style_path,
                    topic_cfg=topic_cfg,
                    content_plan=content_plan,
                    speaker_variants=speaker_variants,
                    base_metadata=content_plan_metadata,
                )
                if dialogue_path is not None:
                    print(f"[detective-v2] dialogue saved to: {dialogue_path}")
                    print(f"[detective-v2] immediate bias eval saved to: {eval_path}")
                continue

            role_style_bundle = runner.load_style_bundle_from_path(style_path)
            runner.validate_api_style_bundle(role_style_bundle, content_plan=content_plan, trait_name=trait_name)
            role_variants = runner.role_variants()
            role_style_bundle = runner.style_bundle_for_role_variants(role_style_bundle, role_variants)
            style_bundle = runner.role_style_bundle_to_speaker_bundle(
                role_style_bundle,
                ground_truth_side_order=ground_truth_side_order,
            )
            reuse_baseline_dialogue, fixed_speakers, fixed_turn_ids, baseline_dialogue_source = runner.load_reuse_baseline_dialogue()
            dialogue_extra_metadata = dict(content_plan_metadata)
            if baseline_dialogue_source is not None:
                baseline_source = runner.repo_display_path(baseline_dialogue_source)
                dialogue_extra_metadata["baseline_dialogue_source"] = baseline_source
                dialogue_extra_metadata.update(
                    pipeline_detective.DetectivePipelineRunner.canonical_baseline_source_metadata(baseline_source)
                )
            if fixed_speakers:
                dialogue_extra_metadata["fixed_speakers"] = sorted(fixed_speakers)
            if fixed_turn_ids:
                dialogue_extra_metadata["fixed_turn_ids"] = sorted(fixed_turn_ids)
            treatment_metadata = runner.adaptive_treatment_metadata(
                fixed_dialogue=reuse_baseline_dialogue,
                speaker_variants=speaker_variants,
                topic_cfg=topic_cfg,
            )
            dialogue_extra_metadata.update(treatment_metadata)
            _dialogue, dialogue_path, _metadata, _prevalidation_report = generate_dialogue_without_prevalidation(
                runner=runner,
                run_id=run_id,
                topic_cfg=topic_cfg,
                content_plan=content_plan,
                style_bundle=style_bundle,
                trait_name=trait_name,
                speaker_variants=speaker_variants,
                base_metadata=dialogue_extra_metadata,
                fixed_dialogue=reuse_baseline_dialogue,
                fixed_speakers=fixed_speakers,
                fixed_turn_ids=fixed_turn_ids,
                treatment_metadata=treatment_metadata,
            )
            if dialogue_path is not None:
                print(f"[detective-v2] dialogue generated: {dialogue_path}")
            continue

        if args.step == "bias-eval":
            existing_eval = has_existing_step_output(
                output_root_abs,
                subdir="evals",
                case_id=case_id,
                trait_name=trait_name,
                variant_a=variant_a,
                variant_b=variant_b,
                ground_truth_side_order=ground_truth_side_order,
                content_plan_family=content_plan_family,
                interpretation_set_id=interpretation_set_id,
                rollout_index=rollout_index,
            )
            if existing_eval is not None and not args.force_bias_eval:
                print(f"[detective-v2] bias eval already exists, skipping: {existing_eval}")
                continue
            dialogue_path = reference_paths_by_run.get(run_tuple)
            if dialogue_path is None:
                dialogue_path = find_reference_file(
                    reference_output,
                    subdir="dialogues",
                    case_id=case_id,
                    trait_name=trait_name,
                    variant_a=variant_a,
                    variant_b=variant_b,
                    ground_truth_side_order=ground_truth_side_order,
                    content_plan_family=content_plan_family,
                    interpretation_set_id=interpretation_set_id,
                    rollout_index=rollout_index,
                )
            print(f"[detective-v2] loading dialogue from: {dialogue_path}")
            dialogue = runner.load_dialogue_from_path(dialogue_path)
            eval_result = runner.evaluate_bias(
                dialogue=dialogue,
                content_plan=content_plan,
                topic_cfg=topic_cfg,
                trait_name=trait_name,
                speaker_variants=speaker_variants,
                include_computational_verbosity=False,
            )
            eval_path = runner.save_bias_eval(eval_result, run_id)
            cleanup_redundant_attempt_outputs_for_run(runner.output_root(), run_id)
            print(f"[detective-v2] bias eval saved to: {eval_path}")
            continue

        if args.step == "rar-eval":
            existing_rar = has_existing_step_output(
                output_root_abs,
                subdir="rar_evals",
                case_id=case_id,
                trait_name=trait_name,
                variant_a=variant_a,
                variant_b=variant_b,
                ground_truth_side_order=ground_truth_side_order,
                content_plan_family=content_plan_family,
                interpretation_set_id=interpretation_set_id,
                rollout_index=rollout_index,
            )
            if existing_rar is not None:
                print(f"[detective-v2] RaR eval already exists, skipping: {existing_rar}")
                continue
            dialogue_path = reference_paths_by_run.get(run_tuple)
            if dialogue_path is None:
                try:
                    dialogue_path = find_reference_file(
                        reference_output,
                        subdir="accepted",
                        case_id=case_id,
                        trait_name=trait_name,
                        variant_a=variant_a,
                        variant_b=variant_b,
                        ground_truth_side_order=ground_truth_side_order,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                        rollout_index=rollout_index,
                    )
                except FileNotFoundError:
                    dialogue_path = find_reference_file(
                        reference_output,
                        subdir="dialogues",
                        case_id=case_id,
                        trait_name=trait_name,
                        variant_a=variant_a,
                        variant_b=variant_b,
                        ground_truth_side_order=ground_truth_side_order,
                        content_plan_family=content_plan_family,
                        interpretation_set_id=interpretation_set_id,
                        rollout_index=rollout_index,
                    )
            print(f"[detective-v2] loading dialogue from: {dialogue_path}")
            dialogue = runner.load_dialogue_from_path(dialogue_path)
            rar_result = runner.evaluate_rar(dialogue)
            rar_path = runner.save_rar_eval(rar_result, run_id)
            print(f"[detective-v2] RaR eval saved to: {rar_path}")
            continue

    if args.step == "dialogue":
        maybe_cleanup_redundant_attempt_outputs(
            output_root_abs,
            keep_attempt_files=args.keep_attempt_files,
        )
    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the detective v2 sweep or individual pipeline steps using code/configs/traitsV2.py."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled runs without executing.")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run a single case on a single ablation.")
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--trait", "--bias", dest="trait", type=str, default=None)
    parser.add_argument("--variant-a", type=str, default=None, choices=("baseline", "active"))
    parser.add_argument("--variant-b", type=str, default=None, choices=("baseline", "active"))
    parser.add_argument("--culprit-variant", type=str, default=None, choices=("baseline", "active"))
    parser.add_argument("--rival-variant", type=str, default=None, choices=("baseline", "active"))
    parser.add_argument(
        "--gt-order",
        dest="ground_truth_side_order",
        type=str,
        default=None,
        choices=GROUND_TRUTH_SIDE_ORDER_CHOICES,
    )
    parser.add_argument("--step", type=str, default="full", choices=STEP_CHOICES)
    parser.add_argument("--reference-output", type=str, default=None)
    parser.add_argument(
        "--interpretation-set",
        type=str,
        default=None,
        help="Run only one interpretation set, e.g. primary-2 or primary-1__additional-2.",
    )
    parser.add_argument("--rival-suspect", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--rollouts-per-combo",
        type=positive_int,
        default=1,
        help="Generate each final dialogue combination this many independent times.",
    )
    parser.add_argument("--list-reference-matches", action="store_true")
    parser.add_argument(
        "--force-attention-question",
        action="store_true",
        help="Regenerate content_attention_question even when the referenced plan already has one.",
    )
    parser.add_argument("--force-content-plan", action="store_true")
    parser.add_argument(
        "--force-role-dialogue",
        action="store_true",
        help="Regenerate canonical role dialogue even when the role_dialogues file exists.",
    )
    parser.add_argument(
        "--force-dialogue",
        action="store_true",
        help="Generate dialogue even when a matching dialogue file already exists.",
    )
    parser.add_argument(
        "--force-bias-eval",
        action="store_true",
        help="Run bias-eval even when matching eval files already exist.",
    )
    parser.add_argument(
        "--keep-attempt-files",
        action="store_true",
        help="Keep redundant *_attemptN_*.json files after dialogue/full generation.",
    )
    args = parser.parse_args()

    expanded_traits = build_expanded_trait_library()
    pipeline_detective.TRAIT_LIBRARY = expanded_traits
    core_evaluator.TRAIT_LIBRARY = expanded_traits

    base_config, output_root_rel, output_root_abs = build_base_config(args)
    print(f"[detective-v2] outputs -> {output_root_rel}")

    if args.step == "full" and not args.list_reference_matches:
        run_full_schedule(args, expanded_traits, base_config, output_root_rel, output_root_abs)
        return

    run_single_step(args, expanded_traits, base_config, output_root_abs)


if __name__ == "__main__":
    main()
