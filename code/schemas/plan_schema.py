from pydantic import BaseModel
from typing import Dict, List, Literal, Optional


class InferenceEdge(BaseModel):
    source_fact_units: List[str]
    interpretation_unit: str
    supports_conclusion: str


class TurnPlan(BaseModel):
    turn_id: int
    speaker: str
    # Legacy debate fields. Detective Turn 1/2 openings should leave these
    # argumentative fields null and use evidence_ids/fact_units below.
    # Detective post-opening turns use the full semantic contract below; rollout
    # derives these text fields when prompting speakers.
    turn_goal: Optional[str] = None
    claim_to_defend: Optional[str] = None
    case_theory: Optional[str] = None
    narrative_beats: Optional[List[str]] = None
    opponent_point_to_attack: Optional[str] = None
    attack_type: Optional[str] = None  # e.g. "rebuttal", "counter-example", "burden-shift", "expose-gap"
    required_move: Optional[str] = None
    allowed_concession: Optional[str] = None
    end_state: Optional[str] = None
    # Detective openings use evidence_ids/fact_units as ordered factual
    # reconstructions. Detective post-opening turns use these plus the rest of
    # the locked semantic contract fields.
    turn_type: Optional[str] = None
    evidence_ids: Optional[List[int]] = None
    fact_units: Optional[List[str]] = None
    interpretation_units: Optional[List[str]] = None
    opponent_claim_target: Optional[str] = None
    required_concession: Optional[str] = None
    core_conclusion: Optional[str] = None
    inference_edges: Optional[List[InferenceEdge]] = None
    certainty_level: Optional[str] = None
    # Detective pairwise_reason_slots_v1 fields. These describe fixed reasons
    # and transformable reasoning slots without prewriting inference edges.
    subject_suspect: Optional[str] = None
    accusation_clause: Optional[str] = None
    reason_refs: Optional[List[str]] = None
    sentence_jobs: Optional[List[dict]] = None


class ContentAttentionQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    source_evidence_ids: List[str]


class ContentPlan(BaseModel):
    content_plan_version: Optional[str] = None
    content_plan_family: Optional[Literal["standard", "verbosity"]] = None
    interpretation_set_id: Optional[str] = None
    selected_interpretations: Optional[dict] = None
    topic: Optional[str] = None
    debate_question: Optional[str] = None
    agent_a_stance: Optional[str] = None
    agent_b_stance: Optional[str] = None
    debate_setup: Optional[dict] = None
    claim_realization_bundle: Optional[dict] = None
    reasoning_finding_source_path: Optional[str] = None
    selected_reasons: Optional[dict] = None
    role_turn_templates: Optional[dict] = None
    turn_dependencies: Optional[dict] = None
    ground_truth_side_order: Optional[str] = None
    private_truth_used: bool = False
    official_solution_public_evidence_map: Optional[List[dict]] = None
    outcome_only_revelations_to_exclude: Optional[List[str]] = None
    suspect_evidence_map: Optional[List[dict]] = None
    reason_units: Optional[List[dict]] = None
    evidence_utility_classification: Optional[List[dict]] = None
    content_attention_question: Optional[ContentAttentionQuestion] = None
    turns: List[TurnPlan]


class TransformationPlan(BaseModel):
    trait_name: str
    variant: str
    anchor_unit_id: Optional[str] = None
    # DISABLED: confirmation_bias removed from the current experiment.
    # Retained as an optional compatibility field for older saved style plans.
    provisional_side: Optional[str] = None
    fallacy_subtype: Optional[str] = None
    hook_unit_ids: Optional[List[str]] = None
    branch_metadata: Optional[dict] = None
    ordered_unit_ids: List[str]
    foreground_unit_ids: List[str]
    background_unit_ids: List[str]
    repeat_unit_ids: List[str]
    allowed_surface_operations: List[str]
    allowed_inference_operations: List[str]


class StyleGuidelineEntry(BaseModel):
    baseline: str
    biased: str


class MinimalRoleStylePlan(BaseModel):
    style_plan_version: str
    style_plan_generation: Literal["deterministic", "api"]
    case_id: str
    trait_name: str
    guidelines: Dict[str, StyleGuidelineEntry]
