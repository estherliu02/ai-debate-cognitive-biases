"""Content planner prompt for the pairwise detective debate pipeline."""

from __future__ import annotations


def _format_evidence_bank(evidence_bank: list[dict]) -> str:
    lines = []
    for item in evidence_bank:
        lines.append(f"[E{item['index']}] {item['text']}")
    return "\n".join(lines)


def build_detective_planner_prompt(
    motion: str,
    question: str,
    stance_a: str,
    stance_b: str,
    turns_per_agent: int,
    evidence_bank: list[dict],
    correct_answer: str | None = None,
    outcome_reference: str | None = None,
    mystery_text: str | None = None,
    all_suspects: list[str] | None = None,
    suspect_options: list[dict] | None = None,
    other_suspects: list[str] | None = None,
    ground_truth_supporting_speaker: str | None = None,
) -> str:
    if turns_per_agent != 3:
        raise ValueError(
            f"Detective planner requires turns_per_agent=3 for pairwise_role_slots_v2, got {turns_per_agent}."
        )
    evidence_text = _format_evidence_bank(evidence_bank)
    suspect_names = all_suspects or [
        str(item.get("name") or "").strip()
        for item in (suspect_options or [])
        if str(item.get("name") or "").strip()
    ]
    if not suspect_names:
        suspect_names = [correct_answer] if correct_answer else []
        suspect_names.extend(other_suspects or [])
    suspect_roster_text = ", ".join(dict.fromkeys(name for name in suspect_names if name)) or "None listed"
    other_suspects_text = ", ".join(other_suspects or []) or "None listed"
    gt_speaker = ground_truth_supporting_speaker or "UNKNOWN"
    non_gt_speaker = "B" if gt_speaker == "A" else "A" if gt_speaker == "B" else "UNKNOWN"

    return f"""
You are creating a detective debate content plan for the pairwise role-slot pipeline.

The current runtime builds this plan deterministically from the converted case
JSON's static reasoning object. Do not generate new suspect reasons or
bias-specific reasoning.

Fixed format:
- content_plan_version must be "pairwise_role_slots_v2".
- Exactly 6 turns: 1=A, 2=B, 3=A, 4=B, 5=A, 6=B.
- Turns 1-2 are factual reconstructions only.
- Turns 3-4 are arguments.
- Turns 5-6 are rebuttals.
- No Turn 7, Turn 8, summary turn, or final_focus turn exists.

Debate question:
TOPIC: {motion}
QUESTION: {question}
Agent A stance: {stance_a}
Agent B stance: {stance_b}

Private planner context:
canonical_culprit_from_answer: {correct_answer or "UNKNOWN"}
suspect_roster_from_answer_options: {suspect_roster_text}
other_suspects: {other_suspects_text}
ground_truth_supporting_speaker: {gt_speaker}
non_ground_truth_speaker: {non_gt_speaker}
outcome_reference:
{outcome_reference or "UNKNOWN"}
mystery_text:
{mystery_text or "See the sentence-split evidence bank below."}

Shared evidence bank:
{evidence_text}

Strict requirements:
- Do not generate, rewrite, or reinterpret the four reason units.
- Do not output neutral_reasoning, interpretation_units, inference_edges, or certainty_level.
- Every fact_unit in selected reasons, role turn templates, and opening packages must be an observable public fact only.
- Do not put interpretations, motives, causal explanations, likelihood judgments, guilt/innocence conclusions, or phrases such as "suggests", "indicates", "may be interpreted as", "unlikely", or "to gain attention" inside fact_units.
- Do not merge separate evidence items into one event, object, or causal relation in fact_units unless the story explicitly states that relation.
- Any connection not explicitly stated in the story belongs only in later reasoning, phrased as a possibility rather than a definite fact.
- The plan must only select a culprit/rival pair, role-based opening evidence packages, fixed reason references, sentence_jobs, reasoning slot IDs, and core conclusions.
- Later style may operate only inside named reasoning slots.
- Normalize character references around the two debated suspects. Do not use narrator-centered identity labels such as "the narrator", "the narrator's brother", or "the narrator's sister" in turn templates, sentence jobs, core conclusions, or reason references.
- Convert first-person source narration into explicit third-person names or supported relationships. Prefer the shortest clear relationship between named characters, especially the relationship between the two suspects, and do not invent relationships not supported by the story or structured case data.
- Return JSON only.
""".strip()
