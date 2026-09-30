from typing import Any, Optional

from pydantic import BaseModel, Field


class RoundTraitScore(BaseModel):
    round_idx: int
    score: int
    reason: str


class TurnBiasScore(BaseModel):
    turn_id: int
    speaker: str
    variant_name: str
    turn_type: Optional[str] = None
    required_for_pass: bool = True
    applicability_reason: Optional[str] = None
    expected_guidance_satisfied: Optional[bool] = None
    expected_guidance_type: Optional[str] = None
    question_1_guidance_satisfied: Optional[bool] = None
    question_1_markers_present: Optional[bool] = None
    question_1_all_guidance_satisfied: Optional[bool] = None
    question_1_clear_guidance_violation: Optional[bool] = None
    question_1_predicted_guidance_satisfied: Optional[bool] = None
    expected_has_bias: bool
    expected_bias_type: Optional[str] = None
    question_1_predicted_has_bias: Optional[bool] = None
    question_1_correct: bool
    question_1_reason: Optional[str] = None
    question_2_asked: bool = False
    question_2_predicted_bias_type: Optional[str] = None
    question_2_correct: Optional[bool] = None
    question_2_reason: Optional[str] = None
    passed: bool
    failure_modes: list[str] = Field(default_factory=list)


class BiasTurnAggregate(BaseModel):
    turn_id: int
    speaker: str
    variant_name: str
    turn_type: Optional[str] = None
    required_for_pass: bool = True
    applicability_reason: Optional[str] = None
    expected_guidance_satisfied: Optional[bool] = None
    expected_guidance_type: Optional[str] = None
    expected_has_bias: bool
    expected_bias_type: Optional[str] = None
    models_passing: int
    models_total: int
    passed: bool


class BiasSideSummary(BaseModel):
    speaker: str
    variant_name: str
    passed_turns: int
    total_turns: int
    non_applicable_turns: int = 0
    passed: bool


class EvalResult(BaseModel):
    target_trait_score: int
    target_trait_round_scores: list[RoundTraitScore] = Field(default_factory=list)
    content_alignment_score: int
    naturalness_score: int
    adversarial_engagement_score: int
    disagreement_persistence_score: int
    rebuttal_coverage_score: int
    strategic_pressure_score: int
    evidence_grounding_score: Optional[int] = None
    passed: bool
    reason: str
    failure_modes: list[str] = Field(default_factory=list)


class DetectiveBiasEvalResult(BaseModel):
    model: str
    turn_scores: list[TurnBiasScore] = Field(default_factory=list)
    expected_biased_speaker: Optional[str] = None
    predicted_biased_speaker: Optional[str] = None
    side_selection_correct: Optional[bool] = None
    bias_present: Optional[bool] = None
    expected_bias_type: Optional[str] = None
    predicted_bias_type: Optional[str] = None
    bias_type_correct: Optional[bool] = None
    confidence: Optional[str] = None
    correct_question_1_turns: int
    correct_question_2_turns: int
    passed_turns: int
    total_turns: int
    passed_turns_a: int
    total_turns_a: int
    passed_turns_b: int
    total_turns_b: int
    non_applicable_turns: int = 0
    passed: bool
    reason: str
    failure_modes: list[str]
    raw_payload: Optional[dict[str, Any]] = None


class CrossModelDetectiveEvalResult(BaseModel):
    evaluation_type: str = "llm_semantic"
    deterministic_details: Optional[dict[str, Any]] = None
    computational_verbosity_passed: Optional[bool] = None
    verbosity_length_difference: Optional[float] = None
    verbosity_abs_length_difference: Optional[float] = None
    verbosity_threshold: Optional[float] = None
    judge_match_count: Optional[int] = None
    judge_total_count: Optional[int] = None
    judge_status: Optional[str] = None
    llm_judge_passed: Optional[bool] = None
    final_bias_eval_status: Optional[str] = None
    final_bias_eval_passed: Optional[bool] = None
    expected_biased_speaker: Optional[str] = None
    predicted_biased_speaker: Optional[str] = None
    side_selection_correct: Optional[bool] = None
    side_selection_model_votes: dict[str, int] = Field(default_factory=dict)
    bias_present: Optional[bool] = None
    expected_bias_type: Optional[str] = None
    predicted_bias_type: Optional[str] = None
    bias_type_correct: Optional[bool] = None
    bias_type_model_votes: dict[str, int] = Field(default_factory=dict)
    per_model_results: list[DetectiveBiasEvalResult]
    turn_outcomes: list[BiasTurnAggregate]
    side_summaries: list[BiasSideSummary]
    passed_turns: int
    total_turns: int
    non_applicable_turns: int = 0
    models_with_all_turns_passing: int
    passed: bool
    reason: str
    failure_modes: list[str]
