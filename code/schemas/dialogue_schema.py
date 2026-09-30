from typing import List, Optional
from pydantic import BaseModel


class DialogueTurn(BaseModel):
    turn_id: int
    speaker: str
    utterance: str
    opponent_claim_targeted: Optional[str] = None
    attack_move_used: Optional[str] = None
    content_fidelity_note: Optional[str] = None
    # Evidence-grounded mode only: indices from the shared evidence bank that
    # this speaker explicitly cited or relied on in their utterance.
    evidence_citations: Optional[List[int]] = None
    content_preservation_note: Optional[str] = None
    treatment_realization_note: Optional[str] = None
    generated_from_role_opening: Optional[str] = None
    speaker_role: Optional[str] = None
    role_turn_id: Optional[str] = None


class Dialogue(BaseModel):
    topic: str
    case_id: Optional[str] = None
    content_plan_family: Optional[str] = None
    interpretation_set_id: Optional[str] = None
    trait_name: str
    variant_name_a: str
    variant_name_b: str
    ground_truth_side_order: Optional[str] = None
    agent_a_stance: Optional[str] = None
    agent_b_stance: Optional[str] = None
    metadata: Optional[dict] = None
    turns: List[DialogueTurn]


class RoleDialogueTurn(BaseModel):
    speaker_role: str
    role_turn_id: str
    utterance: str
    opponent_claim_targeted: Optional[str] = None
    attack_move_used: Optional[str] = None
    content_fidelity_note: Optional[str] = None
    evidence_citations: Optional[List[int]] = None
    content_preservation_note: Optional[str] = None
    treatment_realization_note: Optional[str] = None
    generated_from_role_opening: Optional[str] = None
    replies_to_role_turn_id: Optional[str] = None


class RoleDialogue(BaseModel):
    role_dialogue_version: str
    case_opening_version: Optional[str] = None
    case_id: str
    content_plan_family: Optional[str] = None
    interpretation_set_id: Optional[str] = None
    trait_name: str
    culprit_variant: str
    rival_variant: str
    rollout_index: Optional[int] = None
    seed: Optional[int] = None
    topic: Optional[str] = None
    culprit_name: Optional[str] = None
    rival_suspect_name: Optional[str] = None
    culprit_stance: Optional[str] = None
    rival_stance: Optional[str] = None
    content_plan_source: Optional[str] = None
    style_plan_source: Optional[str] = None
    opening_source_role_dialogue: Optional[str] = None
    baseline_source_role_dialogue: Optional[str] = None
    baseline_canonical_source: Optional[dict] = None
    role_turns: dict[str, RoleDialogueTurn]
