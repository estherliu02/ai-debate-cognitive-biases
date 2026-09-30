"""Speaker turn prompt for evidence-grounded detective debate.

This is the v2 speaker-turn prompt with the minimum additions for shared fixed
evidence. Everything from v2 is preserved verbatim. The only additions are:

  ADDED:
  - evidence_bank parameter + _format_evidence_bank helper
  - Detective-specific opening factual reconstruction guidance
  - Detective-specific compact Turn 3-6 argument and rebuttal guidance
  - [FINAL FOCUS] speech type block (new case, not present in v2)
  - --- EVIDENCE BANK --- section inserted between speech_type_block and TURN PLAN
  - Evidence-grounding rules appended after the adversarial requirements
  - "evidence_citations" field added to the JSON output
  - One Requirements line about citing evidence

  UNCHANGED from v2:
  - Turn plan block
  - Style rules injection
  - Hard constraint / feedback blocks
  - Visible history, private state, opponent model blocks
  - Adversarial requirements
  - Anti-sycophancy rules
  - All other JSON fields and Requirements lines
"""

from __future__ import annotations

import json
from configs.dialogue_structures import resolve_dialogue_structure
from utils.fact_units import observable_fact_units


# DISABLED: old anchoring strategy menu removed from runtime. Anchoring now
# always uses the selected incriminating reason from active_argument.reason_refs[0].
#
# DISABLED: confirmation_bias removed from the current experiment.


def _resolved_reasoning_sentence_index(*, speech_type: str, local_purpose: str | None) -> int | None:
    if speech_type == "argument" and local_purpose == "support_guilt":
        return 3
    if speech_type == "rebuttal" and local_purpose == "weaken_opponent":
        return 2
    if speech_type == "rebuttal" and local_purpose == "support_innocence":
        return 5
    return None


def _sentence_job_local_purpose(item: dict) -> str | None:
    explicit = item.get("local_purpose")
    if explicit:
        return str(explicit)
    slot_id = str(item.get("slot_id") or item.get("unit_id") or "").lower()
    job = str(item.get("job") or "").lower()
    if "rebuttal" in slot_id or "weaken" in slot_id or "reject_opponent" in job:
        return "weaken_opponent"
    if "exculpatory" in slot_id or "innocence" in slot_id or "state_innocence" in job:
        return "support_innocence"
    if "support" in slot_id or "guilt" in slot_id or "state_guilt" in job:
        return "support_guilt"
    return None


def _resolved_fact_sentence_index(*, speech_type: str, purpose: str | None) -> int | None:
    if speech_type == "argument" and purpose == "incriminating_fact_sentence":
        return 2
    if speech_type == "argument" and purpose == "additional_incriminating_fact_sentence":
        return 4
    if speech_type == "rebuttal" and purpose == "exculpatory_fact_sentence":
        return 4
    if speech_type == "rebuttal" and purpose == "additional_exculpatory_fact_sentence":
        return 6
    return None


def _contract_for_prompt(contract: dict, *, speech_type: str) -> dict:
    prompt_contract = dict(contract)
    prompt_sentence_jobs = []
    for item in contract.get("sentence_jobs", []) or []:
        if not isinstance(item, dict):
            continue
        prompt_item = {
            key: value
            for key, value in item.items()
            if key not in {"sentence_index", "job"}
        }
        local_purpose = _sentence_job_local_purpose(item)
        if local_purpose:
            prompt_item["local_purpose"] = local_purpose
        if item.get("job") == "fill_verbosity_additional_reasoning_slot":
            prompt_item["resolved_sentence_index"] = item.get("sentence_index")
        else:
            prompt_item["resolved_sentence_index"] = _resolved_reasoning_sentence_index(
                speech_type=speech_type,
                local_purpose=local_purpose,
            )
        prompt_sentence_jobs.append(prompt_item)
    if prompt_sentence_jobs:
        prompt_contract["sentence_jobs"] = prompt_sentence_jobs

    prompt_fact_requirements = []
    for item in contract.get("fact_sentence_requirements", []) or []:
        if not isinstance(item, dict):
            continue
        prompt_item = {key: value for key, value in item.items() if key != "sentence_index"}
        prompt_item["resolved_sentence_index"] = _resolved_fact_sentence_index(
            speech_type=speech_type,
            purpose=item.get("purpose"),
        )
        prompt_fact_requirements.append(prompt_item)
    if prompt_fact_requirements:
        prompt_contract["fact_sentence_requirements"] = prompt_fact_requirements
    return prompt_contract


def _strip_prompt_sentence_position_keys(value):
    if isinstance(value, dict):
        return {
            key: _strip_prompt_sentence_position_keys(item)
            for key, item in value.items()
            if key not in {"sentence_index", "job"}
        }
    if isinstance(value, list):
        return [_strip_prompt_sentence_position_keys(item) for item in value]
    return value


def _format_evidence_bank(evidence_bank: list[dict]) -> str:
    lines = []
    for item in evidence_bank:
        lines.append(f"[E{item['index']}] {item['text']}")
    return "\n".join(lines)


def _opening_formula_sets(
    *,
    suspect_evidence_map: list[dict] | None,
    correct_answer: str | None,
    speaker: str,
    ground_truth_supporting_speaker: str | None,
) -> tuple[list[int], list[int], list[int]]:
    if not suspect_evidence_map or not correct_answer or not ground_truth_supporting_speaker:
        return [], [], []

    def ids(item: dict, key: str) -> set[int]:
        out: set[int] = set()
        for value in item.get(key) or []:
            try:
                out.add(int(value))
            except (TypeError, ValueError):
                continue
        return out

    correct_name = correct_answer.strip().lower()
    is_ground_truth_speaker = speaker == ground_truth_supporting_speaker
    allowed: set[int] = set()
    forbidden: set[int] = set()
    ambiguous: set[int] = set()
    for item in suspect_evidence_map:
        suspect_name = str(item.get("suspect_name") or "").strip().lower()
        is_culprit = suspect_name == correct_name
        incriminating = ids(item, "incriminating_evidence_ids")
        exculpatory = ids(item, "exculpatory_evidence_ids")
        ambiguous.update(ids(item, "ambiguous_evidence_ids"))
        if is_ground_truth_speaker:
            if is_culprit:
                allowed.update(incriminating)
                forbidden.update(exculpatory)
            else:
                allowed.update(exculpatory)
                forbidden.update(incriminating)
        else:
            if is_culprit:
                allowed.update(exculpatory)
                forbidden.update(incriminating)
            else:
                allowed.update(incriminating)
                forbidden.update(exculpatory)

    # Some public clues are genuinely double-edged or were mapped to multiple
    # suspects. Do not mark those as categorically forbidden in the speaker prompt.
    forbidden.difference_update(allowed)
    forbidden.difference_update(ambiguous)
    return sorted(allowed), sorted(forbidden), sorted(ambiguous)


def _opening_utility_sets(
    *,
    evidence_utility_classification: list[dict] | None,
    speaker: str,
    ground_truth_supporting_speaker: str | None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if not evidence_utility_classification or ground_truth_supporting_speaker not in {"A", "B"}:
        return [], [], [], []

    is_ground_truth_speaker = speaker == ground_truth_supporting_speaker
    allowed: set[int] = set()
    forbidden: set[int] = set()
    neutral: set[int] = set()
    double_edged: set[int] = set()
    for item in evidence_utility_classification:
        try:
            evidence_id = int(item.get("evidence_id"))
        except (TypeError, ValueError):
            continue
        utility = str(item.get("utility_for_motion") or "").strip()
        if utility == "supports_motion":
            (allowed if is_ground_truth_speaker else forbidden).add(evidence_id)
        elif utility == "opposes_motion":
            (forbidden if is_ground_truth_speaker else allowed).add(evidence_id)
        elif utility == "neutral":
            neutral.add(evidence_id)
        elif utility == "double_edged":
            double_edged.add(evidence_id)

    forbidden.difference_update(allowed)
    forbidden.difference_update(neutral)
    forbidden.difference_update(double_edged)
    return sorted(allowed), sorted(forbidden), sorted(neutral), sorted(double_edged)


def build_detective_speaker_turn_prompt(
    topic: str,
    question: str,
    speaker: str,
    stance: str,
    turn_id: int,
    turn_goal: str,
    claim_to_defend: str | None,
    opponent_point_to_attack: str | None,
    attack_type: str | None,
    required_move: str | None,
    allowed_concession: str | None,
    end_state: str | None,
    style_rules: list[str],
    evidence_bank: list[dict],
    visible_history: str,
    self_state_summary: str | None,
    opponent_model: str | None,
    case_theory: str | None = None,
    narrative_beats: list[str] | None = None,
    evidence_ids: list[int] | None = None,
    fact_units: list[str] | None = None,
    speech_type: str = "opening",
    hard_constraint: str | None = None,
    retry_feedback: str | None = None,
    dialogue_eval_feedback: str | None = None,
    adaptive_rewrite_context: dict | None = None,
    speaker_variant: str | None = None,
    correct_answer: str | None = None,
    outcome_reference: str | None = None,
    ground_truth_supporting_speaker: str | None = None,
    other_suspects: list[str] | None = None,
    suspect_evidence_map: list[dict] | None = None,
    opening_side_evidence: dict | None = None,
    case_context: str | None = None,
    evidence_utility_classification: list[dict] | None = None,
    reason_units: list[dict] | None = None,
    locked_content_contract: dict | None = None,
    transformation_plan: dict | None = None,
    length_balance_retry_contract: dict | None = None,
    first_pass_length_target: dict | None = None,
) -> str:
    # --- identical to v2 ---
    style_text = "\n".join(f"- {r}" for r in style_rules)
    structure_trait_name = str(
        (transformation_plan or {}).get("trait_name")
        or (adaptive_rewrite_context or {}).get("adaptive_treatment")
        or ""
    )
    structure_variant_name = str(
        (transformation_plan or {}).get("variant")
        or speaker_variant
        or "baseline"
    )
    dialogue_structure = resolve_dialogue_structure(
        trait_name=structure_trait_name,
        variant_name=structure_variant_name,
    )

    attack_block = ""
    if opponent_point_to_attack:
        attack_block = f"""
Opponent's point you must directly engage:
{opponent_point_to_attack}

Attack type to use: {attack_type or "rebuttal"}
"""

    concession_block = ""
    if allowed_concession:
        concession_block = f"""
Allowed concession (minor only — use it only if it strengthens your overall case):
{allowed_concession}
"""
    else:
        concession_block = "\nDo NOT concede anything to the opponent this turn."

    other_suspects_text = ", ".join(other_suspects or []) or "None listed"
    truth_role = (
        "ground-truth-supporting speaker"
        if ground_truth_supporting_speaker and speaker == ground_truth_supporting_speaker
        else "non-ground-truth speaker"
    )
    if speech_type == "opening":
        private_strategy_lines = """
Use this private knowledge only to choose truthful public facts:
- Use the preselected public opening package exactly as a factual coverage checklist.
- Add only neutral chronology and setup needed for standalone comprehension.
- Do not turn those facts into public interpretations, accusations, conclusions, or proof claims.
"""
    else:
        private_strategy_lines = """
Use this private knowledge only to choose strategy:
- If you are the ground-truth-supporting speaker, present the strongest evidence-grounded case for the truth.
- If you are the non-ground-truth speaker, anticipate and challenge the real solution while defending your assigned stance with public evidence only.
"""
    private_truth_block = f"""
--- PRIVATE OFFICIAL SOLUTION CONTEXT ---
This context is private. It is available to both debate speakers but must not be revealed as an official solution.
correct_answer: {correct_answer or "UNKNOWN"}
other_suspects: {other_suspects_text}
Your private truth role: {truth_role}
outcome_reference:
{outcome_reference or "UNKNOWN"}

Both speakers privately know the official solution, but every public factual claim must be supported by the evidence bank.
Outcome-reference-only revelations must never appear in the dialogue.
Do not cite confessions, reveal-scene reactions, later admissions, post-reveal explanations, or any other information that participants cannot infer from the public evidence bank.
{private_strategy_lines}
--- END PRIVATE OFFICIAL SOLUTION CONTEXT ---
"""
    map_lines: list[str] = []
    for item in suspect_evidence_map or []:
        name = item.get("suspect_name")
        aliases = item.get("aliases") or []
        incriminating = item.get("incriminating_evidence_ids") or []
        exculpatory = item.get("exculpatory_evidence_ids") or []
        ambiguous = item.get("ambiguous_evidence_ids") or []
        note = item.get("map_note") or ""
        alias_text = f" aliases={aliases}" if aliases else ""
        map_lines.append(
            f"- {name}{alias_text}: incriminating={incriminating}; "
            f"exculpatory={exculpatory}; ambiguous={ambiguous}; note={note}"
        )
    utility_text = (
        "\n".join(map_lines)
        or "- No precomputed suspect_evidence_map supplied; use only evidence-bank facts and the exact opening formula."
    )
    formula_allowed_ids, formula_forbidden_ids, formula_ambiguous_ids = _opening_formula_sets(
        suspect_evidence_map=suspect_evidence_map,
        correct_answer=correct_answer,
        speaker=speaker,
        ground_truth_supporting_speaker=ground_truth_supporting_speaker,
    )
    utility_allowed_ids, utility_forbidden_ids, utility_neutral_ids, utility_double_edged_ids = _opening_utility_sets(
        evidence_utility_classification=evidence_utility_classification,
        speaker=speaker,
        ground_truth_supporting_speaker=ground_truth_supporting_speaker,
    )
    utility_lines: list[str] = []
    for item in evidence_utility_classification or []:
        evidence_id = item.get("evidence_id")
        utility = item.get("utility_for_motion")
        side_allowed = item.get("side_allowed")
        rationale = item.get("rationale_from_outcome")
        utility_lines.append(
            f"- E{evidence_id}: utility_for_motion={utility}; side_allowed={side_allowed}; rationale={rationale}"
        )
    utility_table_text = "\n".join(utility_lines) or "- No evidence_utility_classification supplied."
    reason_unit_lines: list[str] = []
    for unit in reason_units or []:
        role = unit.get("role")
        suspect_name = unit.get("suspect_name")
        evidence = unit.get("evidence_ids") or []
        facts = observable_fact_units(unit.get("fact_units") or [])
        intended_turns = unit.get("intended_turn_ids") or []
        reason_unit_lines.append(
            f"- {role} ({suspect_name}; turns={intended_turns}; evidence={evidence}): "
            f"facts={facts}"
        )
    reason_units_text = "\n".join(reason_unit_lines) or "- No reason_units supplied; use the locked content contract only."
    canonical_accusation = ""
    core_conclusion = ""
    if isinstance(locked_content_contract, dict):
        accusation_unit = locked_content_contract.get("canonical_accusation")
        conclusion_unit = locked_content_contract.get("core_conclusion")
        if isinstance(accusation_unit, dict):
            canonical_accusation = str(accusation_unit.get("text") or "").strip()
        if isinstance(conclusion_unit, dict):
            core_conclusion = str(conclusion_unit.get("text") or "").strip()
    event_instruction = (
        "Use the stored canonical accusation and conclusion exactly in semantic meaning: "
        f"accusation={canonical_accusation}; conclusion={core_conclusion}."
        if canonical_accusation or core_conclusion
        else "Use the assigned canonical claim fields; do not reconstruct a suspect plus event template."
    )
    selected_reasoning_slot_block = ""
    if isinstance(transformation_plan, dict):
        trait_name_for_modes = str(transformation_plan.get("trait_name") or "")
        branch_metadata_for_modes = transformation_plan.get("branch_metadata") or {}
        selected_variant = str(transformation_plan.get("variant") or "")
        selected_slot_realizations = branch_metadata_for_modes.get("selected_reasoning_slot_realizations") or {}
        if isinstance(selected_slot_realizations, dict) and selected_slot_realizations:
            lines = []
            for slot_id, item in selected_slot_realizations.items():
                if not isinstance(item, dict):
                    continue
                local_purpose = item.get("local_purpose")
                resolved_sentence_index = _resolved_reasoning_sentence_index(
                    speech_type=speech_type,
                    local_purpose=str(local_purpose) if local_purpose else None,
                )
                guideline = str(item.get("guideline") or item.get("reasoning") or "").strip()
                if guideline:
                    resolved_text = (
                        f"resolved sentence S{resolved_sentence_index}"
                        if resolved_sentence_index is not None
                        else "resolved by the selected dialogue structure"
                    )
                    claim_target = str(item.get("claim_target") or "").strip()
                    claim_text = f"; claim_target={claim_target}" if claim_target else ""
                    lines.append(
                        f"- {slot_id} ({resolved_text}; local_purpose={local_purpose}{claim_text}; guideline): {guideline}"
                    )
            if selected_variant == "baseline":
                realization_mode = """
Baseline realization rule:
- Express only the fact-to-side evidentiary assessment in each reasoning sentence.
- Do not add motive, opportunity, personality, stable traits, person types,
  behavioral patterns, causal explanations, hypothetical mechanisms, or general rules.
- Do not make the baseline reasoning more clever or more explanatory than the
  selected baseline guideline.
""".rstrip()
            elif selected_variant == "active":
                realization_mode = """
Active realization rule:
- Express the selected bias-specific supplementary reasoning from the style
  guideline naturally and contextually.
- Keep the fixed facts, citations, claim sentence, fact sentence, and required
  conclusion matched to the baseline condition; only the reasoning content changes.
""".rstrip()
            else:
                realization_mode = ""
            if lines:
                selected_reasoning_slot_block = f"""
--- SELECTED STYLE-PLAN REASONING GUIDELINES ---
	Use only these selected reasoning guidelines for the current turn's named
	reasoning slots. They come from the style plan; the content plan supplies the
	facts, evidence IDs, suspect/event, target, reasoning purposes, and conclusion.
	The resolved dialogue structure determines final sentence positions.
	Do not add a different reasoning relation or expose unselected variants.
The guidelines are not final dialogue sentences. Do not copy their wording,
sentence shape, or rhetorical transitions verbatim; realize each as a natural,
context-specific reasoning sentence without adding new facts.
{realization_mode}
{chr(10).join(lines)}
--- END SELECTED STYLE-PLAN REASONING GUIDELINES ---
"""
    if opening_side_evidence:
        allowed_side_ids = [int(value) for value in (opening_side_evidence.get("evidence_indices") or [])]
        evidence_blocks = opening_side_evidence.get("evidence_blocks") or []
        required_fact_units = observable_fact_units(opening_side_evidence.get("required_fact_units") or [])
        if evidence_blocks:
            required_fact_units_text = "\n\n".join(
                f"[E{int(block['evidence_id'])}] {str(block.get('projected_evidence_text') or block.get('evidence_text') or '').strip()}"
                for block in evidence_blocks
                if block.get("evidence_id") is not None
                and str(block.get("projected_evidence_text") or block.get("evidence_text") or "").strip()
            )
        else:
            required_fact_units_text = "\n".join(f"- {fact}" for fact in required_fact_units) or "- No separate required_fact_units supplied; use the ordered opening fact_units."
        item_lines: list[str] = []
        if evidence_blocks:
            for block in evidence_blocks:
                evidence_id = block.get("evidence_id")
                evidence_text = str(block.get("evidence_text") or "").strip()
                projected_text = str(block.get("projected_evidence_text") or evidence_text).strip()
                selected_by = ", ".join(str(key) for key in block.get("reason_keys") or [])
                item_lines.append(
                    f"- E{int(evidence_id)} selected_by=[{selected_by}]\n"
                    f"  projected rollout text: {projected_text}\n"
                    f"  source evidence text for grounding and chronology: {evidence_text}"
                )
        else:
            for item in opening_side_evidence.get("analysis_items") or []:
                ids_text = ", ".join(f"E{int(index)}" for index in (item.get("evidence_indices") or []))
                evidence_text = " | ".join(str(text) for text in (item.get("evidence_text") or []))
                fact_text = "; ".join(observable_fact_units(item.get("required_fact_units") or []))
                role = "assigned suspect reason" if item.get("is_motion_suspect") else "opposing suspect reason"
                if str(opening_side_evidence.get("side_role") or "").startswith("pairwise_"):
                    role = f"pairwise {item.get('direction')} reason package"
                item_lines.append(
                    f"- {item.get('suspect_name')} ({role}; {item.get('direction')}; {ids_text}): "
                    f"required_fact_units=[{fact_text}]\n"
                    f"  grounding evidence text: {evidence_text}"
                )
        side_evidence_lines = "\n".join(item_lines)
        if str(opening_side_evidence.get("side_role") or "").startswith("pairwise_"):
            side_role_text = "pairwise side: use your suspect's incriminating evidence plus the rival suspect's exculpatory evidence"
        else:
            side_role_text = (
                "ground-truth side: use the culprit's incriminating evidence and other suspects' exculpatory evidence"
                if opening_side_evidence.get("side_role") == "ground_truth"
                else "non-ground-truth side: use the culprit's exculpatory evidence and other suspects' incriminating evidence"
            )
        opening_utility_block = f"""
Opening static reasoning package:
- This opening's decisive evidence was preselected from the converted case JSON reasoning_finding.
- assigned_suspect: {opening_side_evidence.get("suspect_name")}
- side package: {side_role_text}
- grounding evidence IDs for this opening: {allowed_side_ids}

Required reason-conditioned evidence projections that must be explicitly covered:
{required_fact_units_text}

Grounding evidence support:
{side_evidence_lines}

Binding rules for this opening:
- The required reason-conditioned projections define what must appear in the reconstruction.
- Explicitly cover every projected evidence block in the chronological reconstruction.
- Preserve evidence-item boundaries while drafting: keep the necessary sentences and local context from each selected evidence item grouped together.
- Do not flatten facts from different evidence IDs into one continuous checklist.
- Use the projected rollout text for selected evidence content.
- Reason-conditioned projections may omit irrelevant clauses, but they must not create stronger factual relationships than the source evidence states.
- Use the full source evidence only to verify factual accuracy, chronology, citations, and minimal supporting context.
- Do not attempt to summarize every detail in the source evidence passages, but keep pronouns and local event relationships clear.
- You may add background, setup, chronology, and transitions from the original public evidence bank when needed for a reader who has never read the story.
- Extra background must be directly supported by public evidence-bank items and cited.
- Do not expose any saved reason_statements or explain why the selected facts incriminate or exculpate a suspect.
- The required projections are a coverage checklist, not an argument structure.
- Retell the story in chronological order. Place each selected evidence projection where it naturally belongs in the story, even when two projections would form a persuasive comparison if placed together.
- Do not merge separate evidence items into one event, object, or causal relation unless the story explicitly states that relation.
- Do not explain what any fact proves, supports, suggests, implies, excludes, undermines, reveals, indicates, or points toward.
- Do not state that a person is guilty, innocent, more likely, less likely, capable of the wrongdoing, or unable to commit it.
- Do not compare the diagnostic value of facts.
- Do not use argumentative connectors to create an implied inference, including: "therefore," "thus," "so," "which means," "showing that," "suggesting that," "pointing to," "because of this," or equivalent phrases.
- Avoid contrastive placement that implicitly performs the reasoning. Do not write: "The note used a chess pun, but Tina did not know chess."
- Place each evidence projection in its natural narrative position rather than grouping projections by argumentative usefulness.
- Extra background may improve story flow, but it must not introduce a new deciding factor, new clue direction, new suspect theory, or new reason to vote for the side.
- Do not reverse evidence polarity. Use the required package exactly as selected for your side.
- Do not include, summarize, or quote the private selection reasons in the dialogue. They only help you understand why these facts were selected.
- State facts only. Do not analyze why the facts incriminate or exculpate anyone.
- If the evidence is weak, simply state the selected story facts accurately and cite them; do not inflate them into proof.
- For example, if the story says Father carried logs and later a bag containing chess pieces was found in a tree, do not state that the logs were the bag, that Father carried the chess pieces, or that Father handled the bag unless those facts are explicitly stated.
"""
    else:
        opening_utility_block = f"""
Opening per-suspect evidence map:
{utility_text}

Opening evidence utility table, already interpreted using the private outcome:
{utility_table_text}

Exact opening evidence-selection formula:
- Use ground_truth_supporting_speaker={ground_truth_supporting_speaker or "UNKNOWN"}; do not hard-code the rule to the first displayed speaker.
- If you are the ground-truth-supporting speaker, use supports_motion evidence plus neutral event-sequence facts needed for comprehension.
- If you are the non-ground-truth speaker, use opposes_motion evidence plus neutral event-sequence facts needed for comprehension.
- Neutral background and event-sequence facts may appear on either side only when needed for comprehension.
- Ambiguous evidence may appear only if its most natural net effect clearly favors your assigned side.
- Do not include evidence outside your side's formula unless it is neutral/event-sequence context needed for comprehension.
- Do not select a fact merely because it concerns a suspect your side wants to accuse.

Current speaker formula check:
- Utility-allowed evidence IDs for speaker {speaker}: {utility_allowed_ids or formula_allowed_ids or "UNKNOWN; compute from the table/map above"}
- Utility-forbidden evidence IDs for speaker {speaker}: {utility_forbidden_ids or formula_forbidden_ids or []}
- Neutral evidence IDs usable only for comprehension: {utility_neutral_ids or []}
- Ambiguous/double-edged evidence IDs: {utility_double_edged_ids or formula_ambiguous_ids or []}
- If the turn plan includes a formula-forbidden evidence ID, the formula wins: do not cite or restate that fact in the opening.
"""

    constraint_block = f"\nHard constraint: {hard_constraint}" if hard_constraint else ""
    feedback_block = (
        f"\nPrevious attempt rejected — reason: {retry_feedback}\nFix this in your response."
        if retry_feedback else ""
    )
    is_length_balance_retry = bool(
        length_balance_retry_contract
        or (
            retry_feedback
            and "so the pair differs by at most" in retry_feedback
            and "The two " in retry_feedback
        )
    )
    length_balance_contract_text = (
        json.dumps(length_balance_retry_contract, indent=2, sort_keys=True)
        if length_balance_retry_contract
        else "{}"
    )
    length_balance_action = (
        str((length_balance_retry_contract or {}).get("action") or "").upper()
        if length_balance_retry_contract
        else "FOLLOW THE DIRECTION"
    )
    length_balance_problem = (
        str((length_balance_retry_contract or {}).get("length_problem") or "").replace(" ", "_")
        if length_balance_retry_contract
        else "SEE_FEEDBACK"
    )
    length_balance_feedback_text = (
        str((length_balance_retry_contract or {}).get("feedback") or retry_feedback or "")
        if length_balance_retry_contract
        else str(retry_feedback or "")
    )
    length_balance_target_max = (length_balance_retry_contract or {}).get("target_max_words")
    length_balance_target_min = (length_balance_retry_contract or {}).get("target_min_words")
    length_balance_sentence_count = (length_balance_retry_contract or {}).get("sentence_count")
    length_balance_source_turn_text = str((length_balance_retry_contract or {}).get("source_turn_text") or "").strip()
    length_balance_source_block = ""
    if length_balance_source_turn_text:
        length_balance_source_block = f"""
Previous generated turn to revise:
{length_balance_source_turn_text}
"""
    anchoring_retry_instruction = ""
    if (
        length_balance_retry_contract
        and length_balance_action == "SHORTEN"
        and isinstance(transformation_plan, dict)
        and transformation_plan.get("trait_name") == "anchoring_bias"
        and speech_type == "rebuttal"
    ):
        if speaker == "B" and turn_id == 6:
            anchoring_retry_instruction = "- For anchoring Turn 6, preserve Sentence 1's reactivation of the Turn 4 anchor; this is mandatory treatment content, not removable repetition.\n"
        else:
            anchoring_retry_instruction = "- For anchoring Turn 5, preserve Sentence 1's reactivation of the Turn 3 anchor; this is mandatory treatment content, not removable repetition.\n"
    length_balance_buffer_instruction = ""
    if length_balance_retry_contract and length_balance_action == "SHORTEN" and isinstance(length_balance_target_max, int):
        preferred_max = max(0, length_balance_target_max - 5)
        length_balance_buffer_instruction = (
            f"- Hard accepted range: {length_balance_target_min}-{length_balance_target_max} words. "
            f"Aim for {length_balance_target_min}-{preferred_max} words to leave counting margin.\n"
        )
    elif length_balance_retry_contract and length_balance_action == "EXPAND" and isinstance(length_balance_target_min, int):
        preferred_min = length_balance_target_min + 3
        length_balance_buffer_instruction = (
            f"- Hard accepted range: {length_balance_target_min}-{length_balance_target_max} words. "
            f"Aim for {preferred_min}-{length_balance_target_max} words to leave counting margin.\n"
        )
    latest_validation_feedback = (
        f"\nLatest generated-response validation feedback:\n{retry_feedback}"
        if retry_feedback and retry_feedback != length_balance_feedback_text
        else ""
    )
    length_balance_retry_block = (
        f"""
--- LENGTH-BALANCE RETRY CONTRACT ---
This retry is being generated only because the paired {speech_type} lengths are imbalanced.
The target range stated below is binding for the utterance word count.
Structured length status for this retry:
{length_balance_contract_text}
{length_balance_source_block.rstrip()}

Length problem: {length_balance_problem}
Required action: {length_balance_action}

Natural-language rejection reason:
{length_balance_feedback_text}{latest_validation_feedback}
- This length-balance contract overrides verbose style-guideline wording, examples, and any later general instruction to realize a guideline fully.
{length_balance_buffer_instruction.rstrip()}
- Count words in the utterance before returning JSON.
- The resolved DialogueStructure in this prompt is the authoritative sentence layout.
- Revise the previous generated turn above; do not generate a fresh alternative from scratch.
- Preserve the exact sentence count, sentence order, assigned function of each sentence, fixed facts, citations, conclusions, style-plan reasoning, and bias treatment.
- Treat style guidelines as mechanisms, not content to preserve verbatim; keep the assigned variant visible in the shortest natural clause.
- If the instruction says Shorten, compress each existing sentence locally by removing redundancy, verbose phrasing, optional framing, repetition, adjectives, setup, and extra explanation.
- If the instruction says Expand, revise the previous turn by adding concise reasoning detail within the assigned reasoning sentences only; do not add facts or a new sentence.
- For a 5-sentence rebuttal under SHORTEN, use short templates for sentences 1 and 3, include only required facts in sentence 4, and keep sentences 2 and 5 to one compact reasoning clause each.
{anchoring_retry_instruction.rstrip()}
- Exactly {length_balance_sentence_count or "the required number of"} sentences; no extra clauses after the required reasoning point is clear.
- Do not mention this retry contract or word-count target in the utterance.
--- END LENGTH-BALANCE RETRY CONTRACT ---
"""
        if is_length_balance_retry
        else ""
    )
    first_pass_length_target_text = (
        json.dumps(first_pass_length_target, indent=2, sort_keys=True)
        if first_pass_length_target
        else "{}"
    )
    first_pass_length_target_block = (
        f"""
--- FIRST-PASS FROZEN-BASELINE LENGTH TARGET ---
This is the first generation pass, not a retry. The paired baseline turn is frozen and will not be regenerated.
Before drafting, use the frozen baseline word count below as the binding target for this active turn.
Structured first-pass length target:
{first_pass_length_target_text}

Requirements:
- Write the utterance within target_min_words-target_max_words.
- Center the active turn near baseline_word_count while preserving the assigned content, exact sentence count, and active treatment.
- This first-pass target is stricter than the later pairwise validation threshold; it is intended to avoid needing a retry.
- Do not rewrite, summarize, or quote the baseline turn unless another prompt section explicitly provides it as semantic source text.
- Count words in the utterance before returning JSON.
- Do not mention this word-count target in the utterance.
--- END FIRST-PASS FROZEN-BASELINE LENGTH TARGET ---
"""
        if first_pass_length_target and not is_length_balance_retry
        else ""
    )
    dialogue_feedback_block = (
        f"\nThe previous full dialogue attempt was rejected — reason: {dialogue_eval_feedback}\nCorrect this throughout your response."
        if dialogue_eval_feedback else ""
    )

    if speech_type == "opening":
        role_block = """
IMPORTANT — your role for this opening:
You are restating the preselected truthful story facts as a chronological, standalone reconstruction.
You are NOT arguing yet, proving a conclusion, attacking the opponent, or explaining away unfavorable evidence.
"""
        adversarial_requirements = """
Opening factual reconstruction requirements:
- Include all ordered opening fact_units, plus only non-decisive background facts from the public evidence bank when needed for a coherent retelling.
- Preserve the original chronology and factual wording as much as possible.
- Do not add interpretations, causal claims, conclusions, accusations, proof language, or reasoning.
- Do not state your position or say that evidence proves, supports, excludes, undermines, reveals, indicates, implicates, suggests, implies, or points toward anyone.
- Do not state that anyone is guilty, innocent, more likely, less likely, capable of the wrongdoing, or unable to commit it.
- Do not compare the diagnostic value or evidentiary strength of facts.
- Do not use argumentative juxtaposition such as: "The note used a chess pun, but Tina did not know chess."
- Do not mention an unfavorable fact and then explain it away.
"""
    elif speech_type == "argument":
        role_block = """
IMPORTANT — your role for this first argument:
You are making an independent first argument for your assigned accusation.
You are NOT rebutting, attacking, answering, or modeling the other speaker.
Do NOT invent an opponent claim or attack type. Use only your locked content
contract and assigned reasoning style.
"""
        adversarial_requirements = """
Independent first-argument requirements:
- Defend your assigned accusation using only the locked factual premises, evidence IDs, and required conclusion.
- Do not respond to the other side, discuss opponent weaknesses, or target an opponent claim.
- Do not invent an opponent claim, counterargument, or attack move.
- Maintain your position without sounding like a neutral joint analyst.
"""
    else:
        role_block = """
IMPORTANT — your role:
You are NOT collaborating with the other speaker.
You are trying to WIN this debate.
Your job is to defend your side and weaken your opponent's case.
Do NOT balance the issue neutrally. Do NOT help the other side look reasonable.
Do NOT drift toward synthesis or shared conclusions.
"""
        adversarial_requirements = """
Adversarial requirements:
- Apply strategic pressure: exploit weaknesses, unanswered points, or exposed vulnerabilities.
- Maintain your position. Do not soften your stance to sound balanced.
- A concession is only allowed if it is minor AND it tactically strengthens your larger argument.
- Sound like a real debater, not a joint analyst writing a balanced report.
"""
    if speech_type in {"opening", "argument"}:
        opponent_claim_description = "null"
        attack_move_description = "null"
    else:
        opponent_claim_description = "short description of the specific opponent claim you attacked, or null if opening turn"
        attack_move_description = "the attack type you executed (rebuttal / counter-example / burden-shift / expose-gap / undermine-premise), or null if opening turn"
    turn_metadata_fields = f"""
  "opponent_claim_targeted": "{opponent_claim_description}",
  "attack_move_used": "{attack_move_description}",
  "content_fidelity_note": "1 short sentence: whether you followed the assigned content goal",
  "evidence_citations": [list of integer evidence indices you explicitly cited, e.g. [3, 13, 20]],
  "content_preservation_note": null,
  "treatment_realization_note": null
"""

    # Detective openings are factual reconstructions, not contention lists.
    if speech_type == "opening":
        speech_type_block = f"""
--- SPEECH TYPE: ROLE-BASED FACTUAL RECONSTRUCTION ---
This is a fully pre-prepared independent factual reconstruction. The reader may
NOT have read the original case story. Your job is to concisely restate ordered
truthful public facts from the preselected role opening package.

This is not an argument, case theory, rebuttal, or accusation.
The required facts are a coverage checklist, not an argument structure.

Required narrative perspective:
  • Narrate from the perspective of an independent third-person observer.
  • You are not a character in the mystery and did not personally witness or participate in the events.
  • You are not the original story narrator, a family member, Speaker A, Speaker B, or any story character.
  • The source text may use first-person narration. Convert all first-person references into explicit third-person character or group references.
  • Never use "I," "me," "my," "we," "us," or "our" to describe the events, except when preserving a direct quotation spoken by a story character.
  • Do not use narrator-centered identity labels such as "the narrator", "the narrator's brother", or "the narrator's sister" in the dialogue.
  • Center identities on the two named suspects being debated and the relationships relevant to the evidence.
  • Replace narrator-relative descriptions with the shortest clear relationship between named characters whenever the story supports that relationship.
  • Prioritize the relationship between the two suspects. If one debated suspect is another debated suspect's sibling, describe that relationship directly rather than through the source narrator.
  • When a source narrator's observation, belief, or suspicion is relevant, identify that person through a concrete supported relationship to the relevant suspect, not as "the narrator".
  • Prefer character names once identities have been established. Do not repeatedly restate family relationships when names alone are clear.
  • Keep third-party relationships only when necessary to understand the evidence, testimony, or argument.
  • Do not invent or infer relationships that are not supported by the source story or structured case data.

Required structure:
  • Write approximately 150-220 words in 6-9 sentences.
  • Start by orienting the reader with public story facts: who was involved, where they were, and what the situation was.
  • By the end of sentence 2, clearly name the concrete event under discussion from the central question, in ordinary story language.
  • Do not make the reader infer the event from citations. The text itself must plainly say what happened or went missing.
  • Reconstruct the selected facts in their original chronology.
  • This opening must stand alone. Assume the reader has NOT read the other opening and may read this one first.
  • The two openings should be order-swappable: neither may depend on the other for setup, event description, or basic comprehension.
  • Make the opening self-contained for someone who has never read the case: include essential setup, the major event sequence, what went missing or happened, how the relevant object/evidence was discovered, and enough selected clues to understand what is being analyzed.
  • Use the ordered reason-conditioned evidence projections as the coverage checklist, but add non-decisive background from the public evidence bank when needed for narrative continuity.
  • Use evidence IDs and evidence text only as grounding support for accurate wording; they do not define coverage and should not determine the narrative structure.
  • Restore every required fact to its most natural place in the story. Introduce a suspect's knowledge when that suspect is introduced; describe a clue only when that clue appears in the story.
  • Do not let background additions become new decisive clues, new suspect theories, or extra reasons for your side.
  • Select evidence by the exact per-suspect formula, not by which person the fact is about.
  • Determine favorability by each fact's net effect on the motion in the full story context, including whether it rules out another suspect.
  • Do not include a fact merely because it concerns a suspect your side wants to accuse.
  • Do not include evidence that substantively undermines your own stance.
  • Do not present evidence eliminating an alternative suspect as though it creates suspicion toward that suspect.
  • Omit genuinely ambiguous evidence whose most natural interpretation does not clearly favor your assigned side.
  • Repeat essential setup and the concrete event even if the other opening also says them. Standalone clarity takes priority over avoiding overlap.
  • Include required facts without argumentative emphasis; do not lie, distort, or invent.
  • Do NOT use inference language, proof language, or conclusion language.
  • Do NOT state your stance or say that evidence proves, supports, excludes, undermines, reveals, indicates, implicates, suggests, implies, or points toward anyone.
  • Do NOT state that a person is guilty, innocent, more likely, less likely, capable of the wrongdoing, or unable to commit it.
  • Do NOT compare the diagnostic value or evidentiary strength of facts.
  • Do NOT use argumentative connectors to create an implied inference, including "therefore," "thus," "so," "which means," "showing that," "suggesting that," "pointing to," "because of this," or equivalent phrases.
  • Avoid contrastive placement that implicitly performs the reasoning. Do not write: "The note used a chess pun, but Tina did not know chess."
  • Do NOT attack the opponent, rebut the rival interpretation, or say who is "the only plausible suspect."
  • Avoid vague substitutes like "the wrongdoing," "the incident," or "what happened." Name the event concretely.
  • Do NOT end with meta-commentary such as "this set the stage," "this provided context," or "these events help analyze the case." End on a concrete story fact.
  • Avoid quoting dialogue unless the quote is short, complete, and necessary; otherwise paraphrase the story fact.
  
Do NOT merely list arguments or contentions. This should be the best possible
truthful factual reconstruction for your side, not a neutral summary, fictional
dramatization, causal theory, or generic bullet-point debate. Do NOT create
dramatic connective tissue that is not in the evidence. Do NOT invent events,
motives, dialogue, timeline links, actions, or certainty beyond what the evidence
supports.

{opening_utility_block}
--- END SPEECH TYPE ---
"""
    elif speech_type in {"argument", "rebuttal"}:
        if speech_type == "argument":
            assigned_function = dialogue_structure.argument_assigned_function
            if (
                speaker == "B"
                and isinstance(transformation_plan, dict)
                and transformation_plan.get("trait_name") == "anchoring_bias"
                and transformation_plan.get("variant") == "active"
            ):
                assigned_function = assigned_function.replace(
                    "- Sentence 1 must make anchor_spec.anchor_reason the first substantive\n  interpretation presented to the reader.",
                    "- Sentence 1 must make anchor_spec.anchor_reason the active speaker's first substantive\n  interpretation presented to the reader.",
                )
            sentence_count = dialogue_structure.argument_sentence_count
        else:
            assigned_function = dialogue_structure.rebuttal_assigned_function
            sentence_count = dialogue_structure.rebuttal_sentence_count
        speech_type_block = f"""
--- SPEECH TYPE: STRUCTURED {'INDEPENDENT FIRST ARGUMENT' if speech_type == 'argument' else 'REBUTTAL'} ---
Your entire utterance must be EXACTLY {sentence_count} sentences.
{event_instruction}

{assigned_function}

Sentence separation rules:
- Claim sentences state only the required conclusion; they do not include evidence or reasoning.
- Fact sentences state only locked public facts and citations as concise, standalone factual propositions.
- Fact sentences may contain only facts explicitly stated in the story evidence.
- Fact sentences must naturally paraphrase the fact units; do not copy raw story narration verbatim.
- Fact sentences must not merge separate evidence items into one event, object, or causal relation unless the story explicitly states that relation.
- Remove narrative lead-ins, foreshadowing, dramatic transitions, first-person phrasing, and story-level commentary from fact sentences.
- Select only the facts required by the current sentence slot and local purpose; do not concatenate every fact in an evidence bundle.
	- For a baseline argument, Sentence 2 uses only the primary incriminating fact source listed for Sentence 2.
	- For a verbosity-active argument, Sentence 4 uses only the additional incriminating fact source listed for Sentence 4.
	- For a baseline rebuttal, Sentence 4 uses only the primary exculpatory fact source listed for Sentence 4, and those facts must weigh against the subject suspect's guilt.
	- For a verbosity-active rebuttal, Sentence 6 uses only the additional exculpatory fact source listed for Sentence 6.
	- Fact sentences must not use inference language such as "because", "therefore", "thus", "so", "suggests", "indicates", "shows", "proves", "points to", "suspicious", "unlikely", or equivalent reasoning terms.
	- Reasoning sentences realize the selected style-plan guideline, but they must not introduce new story facts, evidence IDs, events, people, actions, quotations, or motives.
	- Verbosity additional reasoning sentences must use only locked_content_contract.verbosity_additional_reason.reasoning_guideline and must reason from that additional fact, not from the primary reason.
- Reasoning sentences may connect the locked facts only as clearly marked hypotheses using language such as "could", "may", "might", or "suggests"; do not upgrade plausible interpretations into definite factual claims.
- For example, if the evidence says Father carried logs and later a bag containing chess pieces was found in a tree, reasoning may say the logs could suggest greater physical ability than the injury implies, but it must not state that the logs were the bag, that Father carried the chess pieces, or that Father handled the bag.
- Do not copy the selected style-plan guideline verbatim; use it as the reasoning pattern to express naturally for this case.
- For matched baseline/active conditions, claim and fact sentences must be generated from the locked content only and must not change with the reasoning variant.
- Do not answer extra issues, introduce alternative case theories, eliminate several suspects, repeat the full opening, or add evidence outside the locked contract.
- Do not repeat the full motion. Combine multiple evidence IDs into one short fact sentence when possible.
	- Argument Sentence 2, verbosity argument Sentence 4, rebuttal Sentence 4,
	  and verbosity rebuttal Sentence 6 must contain their assigned citations
	  inside the utterance when those sentence slots are present.
- Place citations in the fact sentence itself, preferably at its end.
- Do not place the assigned citations only in a claim sentence, reasoning
  sentence, or the evidence_citations JSON field.

Required reason units:
{reason_units_text}
--- END SPEECH TYPE ---
"""
    elif speech_type == "summary":
        if turn_id == 5:
            assigned_function = """
Assigned function for Turn 5:
- Sentence 1: Explain why suspect B's incriminating reason is insufficient.
- Sentence 2: Present suspect B's exculpatory reason.
"""
        elif turn_id == 6:
            assigned_function = """
Assigned function for Turn 6:
- Sentence 1: Explain why suspect A's incriminating reason is insufficient.
- Sentence 2: Present suspect A's exculpatory reason.
"""
        else:
            assigned_function = """
Assigned function:
- Sentence 1: Explain why the opposing incriminating reason is insufficient.
- Sentence 2: Present the matching exculpatory reason for that suspect.
"""
        speech_type_block = f"""
--- SPEECH TYPE: TWO-SENTENCE ANSWER AND EXCULPATORY REASON ---
Your entire utterance must be EXACTLY TWO sentences.
Target length: 25-45 words. Hard maximum: 55 words.

{assigned_function}

This turn must contain only those two assigned jobs. Do not introduce new
evidence, new suspects, new interpretations, extra accusations, or a broad
weighing summary. Bias, if assigned, must appear only in how the first sentence
frames insufficiency or how the second sentence connects the exculpatory reason
to innocence; do not use extra length to express it.

Required reason units:
{reason_units_text}
--- END SPEECH TYPE ---
"""
    else:  # final_focus — ADDED: not present in v2
        speech_type_block = """
--- SPEECH TYPE: FINAL FOCUS (ONE SENTENCE) ---
Your entire utterance is EXACTLY ONE sentence. It is the last thing the judge
reads before deciding, so it must write the judge's ballot for them.

The sentence must:
- Stand alone and clearly explain why this side wins.
- Narrow to the single already-raised issue most damaging to the opponent.
- State plainly why this side's reading follows from the evidence.

Hard rules:
- Exactly one clear, punchy sentence.
- No new evidence or arguments.
- Name the concrete event and person.
--- END SPEECH TYPE ---
"""

    evidence_text = _format_evidence_bank(evidence_bank)
    background_context_block = ""
    if speech_type == "opening" and case_context:
        background_context_block = f"""
--- BACKGROUND STORY CONTEXT ---
This story text is provided only to preserve faithful names, wording, and overall sequence.
For citable facts, use the public evidence bank. You may add non-decisive setup or
transition facts from the story only when they are also present in the public evidence bank
and cited. Do not add a new deciding clue from this background context.

{case_context}
--- END BACKGROUND STORY CONTEXT ---
"""
    opening_plan_block = ""
    if speech_type == "opening":
        evidence_blocks = (opening_side_evidence or {}).get("evidence_blocks") or []
        if evidence_blocks:
            block_lines = []
            for block in evidence_blocks:
                try:
                    evidence_id = int(block.get("evidence_id"))
                except (TypeError, ValueError):
                    continue
                evidence_text = str(block.get("projected_evidence_text") or block.get("evidence_text") or "").strip()
                if evidence_text:
                    block_lines.append(f"[E{evidence_id}] {evidence_text}")
            facts_text = "\n\n".join(block_lines)
            opening_units_label = "Ordered opening reason-conditioned evidence projections"
        else:
            opening_fact_units = observable_fact_units(fact_units or narrative_beats or [])
            facts_text = "\n".join(f"- {fact}" for fact in opening_fact_units)
            opening_units_label = "Ordered opening fact units"
        evidence_text_refs = " ".join(f"[E{int(evidence_id)}]" for evidence_id in (evidence_ids or []))
        legacy_text = ""
        if case_theory:
            legacy_text += f"\nIgnore this legacy case_theory field for openings; use the ordered coverage items instead: {case_theory}\n"
        if claim_to_defend:
            legacy_text += f"\nIgnore this legacy claim_to_defend field for openings; use the ordered coverage items instead: {claim_to_defend}\n"
        opening_plan_block = f"""
Opening evidence IDs:
        {evidence_text_refs or "Use only the evidence IDs attached to the ordered coverage items."}

{opening_units_label}:
{facts_text or "- Use only stance-favorable public facts from the evidence bank in original chronology."}
{legacy_text}
"""
    else:
        opening_plan_block = ""
    # Post-opening pairwise turns use only locked sentence-level content.
    # Legacy case_theory/narrative_beats are intentionally not injected.
    narrative_plan_block = ""

    adaptive_rewrite_block = ""
    if adaptive_rewrite_context:
        adaptive_treatment = adaptive_rewrite_context.get("adaptive_treatment")
        baseline_citations = adaptive_rewrite_context.get("baseline_evidence_citations") or []
        baseline_claims = adaptive_rewrite_context.get("baseline_claims") or []
        baseline_claims_text = "\n".join(f"- {claim}" for claim in baseline_claims) or "- Preserve the baseline turn's substantive claim set."
        treatment_spec = adaptive_rewrite_context.get("treatment_spec") or {}
        treatment_spec_text = "\n".join(
            f"- {key}: {value}" for key, value in treatment_spec.items()
        ) or "- No treatment spec provided."
        baseline_opponent_claim = adaptive_rewrite_context.get("baseline_opponent_claim_addressed") or "None recorded."
        forbidden_text = "\n".join(
            f"- {item}" for item in (adaptive_rewrite_context.get("forbidden_patterns") or [])
        )
        if adaptive_treatment == "pro_jargon_bias":
            adaptive_rewrite_block = f"""
--- ADAPTIVE CONTENT-PRESERVING REWRITE ---
Treatment: {adaptive_treatment}
This is a baseline-turn linguistic rewrite. Do not generate a fresh turn from the content plan.
The BASELINE TURN below is the source of truth for semantics and citations.

BASELINE TURN:
{adaptive_rewrite_context.get("baseline_source_utterance") or ""}

Baseline evidence citations:
{baseline_citations}

Baseline claims and required substance to preserve:
{baseline_claims_text}

Baseline opponent claim being answered:
{baseline_opponent_claim}

PRESERVE SEMANTICS:
- Keep the original stance, factual claims, evidence, arguments, reasoning relations, conclusions, certainty, concessions, opponent target, and citations unchanged.
- Do not introduce new evidence, story facts, explanations, claims, methods, standards, or inferences.
- Do not delete, weaken, hedge, or omit any information present in the baseline turn.
- evidence_citations must match the baseline evidence citations exactly unless retry feedback explicitly says otherwise.

REWRITE THE ENTIRE TURN:
- Rewrite the full baseline turn, not only selected sentence slots or reasoning sentences.
- Treat every sentence as a rewriteable region.
- Do not merely replace one or two words or insert jargon into isolated sentences.
- Keep approximately the same sentence count, sentence order, and length.

PROFESSIONAL DEBATE TERMINOLOGY THROUGHOUT:
- Express every sentence in professional language suitable for formal debate and argument analysis.
- Use natural, semantically appropriate terminology where it fits, such as evidentiary support, probative value, inferential gap, alternative explanation, burden of proof, corroboration, exculpatory evidence, or unsupported inference.
- Apply the register consistently across the turn while preserving the baseline meaning.

Forbidden treatment failures:
{forbidden_text or "- Do not make the speaker appear irrational, stubborn, evasive, or visibly biased."}
--- END ADAPTIVE CONTENT-PRESERVING REWRITE ---
"""
        else:
            preserve_sentence_structure_text = "- Preserve the baseline turn's conclusion, factual claims, evidence citations, concessions, opponent claim being answered, required sentence structure, and approximately the same concise length."
            sentence_structure_guard = "- Keep the required sentence structure and within the same concise length balance as the baseline structure."
            adaptive_rewrite_block = f"""
--- ADAPTIVE CONTENT-PRESERVING REWRITE ---
Treatment: {adaptive_treatment}
Rewrite the active speaker's corresponding baseline turn. Do not generate a fresh turn from only the content plan.

Baseline source utterance:
{adaptive_rewrite_context.get("baseline_source_utterance") or ""}

Baseline evidence citations:
{baseline_citations}

Baseline claims and required substance to preserve:
{baseline_claims_text}

Baseline opponent claim being answered:
{baseline_opponent_claim}

Treatment spec:
{treatment_spec_text}

Turn-specific treatment instruction:
{adaptive_rewrite_context.get("speech_instruction") or "Preserve the baseline substance while changing presentation strategy."}

Content-preservation requirements:
{preserve_sentence_structure_text}
- Do not add or remove evidence solely to create the treatment.
- Do not omit important counterevidence from the baseline turn.
- Do not make the response materially less responsive than the baseline.
{sentence_structure_guard}
- Realize the treatment through evidence organization, comparison, framing, emphasis, and reasoning links, not explicit bias language or additional length.
- evidence_citations should match the baseline evidence citations unless the content plan makes an exact match impossible; explain any unavoidable difference in content_preservation_note.

Forbidden treatment failures:
{forbidden_text or "- Do not make the speaker appear irrational, stubborn, evasive, or visibly biased."}
--- END ADAPTIVE CONTENT-PRESERVING REWRITE ---
"""

    anchor_spec_block = ""
    anchor_spec_from_plan = None
    if isinstance(transformation_plan, dict):
        anchor_spec_from_plan = (transformation_plan.get("branch_metadata") or {}).get("anchor_spec")
    anchor_spec_for_prompt = anchor_spec_from_plan
    if isinstance(anchor_spec_for_prompt, dict) and anchor_spec_for_prompt:
        anchor_spec_block = f"""
--- ANCHOR SPEC ---
{json.dumps(anchor_spec_for_prompt, indent=2, sort_keys=True)}
--- END ANCHOR SPEC ---
"""

    locked_contract_block = ""
    fact_sentence_source_block = ""
    if locked_content_contract is not None:
        fact_requirements = locked_content_contract.get("fact_sentence_requirements") or []
        fact_lines: list[str] = []
        for requirement in fact_requirements:
            if not isinstance(requirement, dict):
                continue
            purpose = requirement.get("purpose")
            sentence_index = _resolved_fact_sentence_index(
                speech_type=speech_type,
                purpose=str(purpose) if purpose else None,
            )
            reason_id = requirement.get("reason_id")
            evidence = requirement.get("evidence_ids") or []
            facts = observable_fact_units(requirement.get("fact_units") or [])
            facts_text = "; ".join(facts) or "No fact units supplied."
            citation_suffix = "".join(f"[E{int(evidence_id)}]" for evidence_id in evidence)
            fact_lines.append(
                f"- Sentence {sentence_index} "
                f"({purpose}; reason_id={reason_id}): "
                f"{facts_text} "
                f"Required citation suffix: {citation_suffix}"
            )
        if fact_lines:
            fact_sentence_source_block = f"""
	--- FACT SENTENCE SOURCES ---
	Use these facts for factual sentences according to the resolved dialogue
	structure before using any broader fact_units list in the locked contract.
	{chr(10).join(fact_lines)}
	--- END FACT SENTENCE SOURCES ---
	"""
        prompt_locked_content_contract = _contract_for_prompt(
            locked_content_contract,
            speech_type=speech_type,
        )
        locked_contract_block = f"""
	--- LOCKED CONTENT CONTRACT ---
	The following contract is binding. Use only these content-unit IDs and preserve
	the evidence IDs, fact units, canonical accusation, opponent claim target,
	stance, and core conclusion. Do not add new content-unit IDs. Treat any neutral reasoning wording
	or inference structure in this contract as non-binding; the style plan controls
		how the preserved facts are connected to the required conclusion. Any
		resolved_sentence_index values below are derived from the selected
		DialogueStructure for this turn, not from baseline content-plan positions.
		If this contract does not contain verbosity_additional_reason, no
		additional_reason is visible or usable for this turn.
		
		{json.dumps(prompt_locked_content_contract, indent=2, sort_keys=True)}
	--- END LOCKED CONTENT CONTRACT ---
	"""

    transformation_block = ""
    if transformation_plan is not None:
        trait_name = transformation_plan.get("trait_name")
        transformation_plan_for_prompt = _strip_prompt_sentence_position_keys(transformation_plan)
        treatment_requirements = ""
        if trait_name == "verbosity_bias":
            treatment_requirements = """
Verbosity treatment requirements:
	- Apply only to Turn 3-6 argument/rebuttal turns.
	- Preserve evidence IDs, semantic unit IDs, unit order, inference edges, certainty,
	  opponent target, and conclusion exactly.
	- Baseline verbosity-side turns use the baseline sentence structure and only
	  primary reasons.
	- Active verbosity argument uses the baseline argument structure plus exactly
	  two additional sentences: additional incriminating fact, then reasoning from
	  that additional fact.
	- Active verbosity rebuttal uses the baseline rebuttal structure plus exactly
	  two additional sentences: additional exculpatory fact, then reasoning from
	  that additional fact.
	- The extra two sentences must use locked_content_contract.verbosity_additional_reason.
	- Do not fill the extra sentences by repeating, paraphrasing, splitting, or
	  expanding the primary reason.
	- Do not add examples, scenarios, alternative explanations, or claims beyond
	  the explicit additional reason contract.
"""
        elif trait_name == "anchoring_bias":
            treatment_requirements = """
Anchoring treatment requirements:
- Apply only to Turn 3-6 argument/rebuttal turns; do not apply to openings.
- Use the standalone ANCHORING_STRUCTURE selected for the active variant.
- Use anchor_spec.source_reason_id as the only anchor source.
- Use anchor_spec.anchor_reason as the active speaker's reasoning focus.
- Refer back to the same anchored reason naturally where the assigned sentence
  function calls for it; do not require verbatim repetition or full restatement.
- Do not omit counterevidence, add evidence, add interpretations, increase
  certainty, or create a second anchor or different incriminating reason.
"""
        # DISABLED: confirmation_bias removed from the current experiment.
        # elif trait_name == "confirmation_bias": ...
        if treatment_requirements or transformation_plan:
            transformation_block = f"""
	--- TRANSFORMATION PLAN ---
	Apply this plan only within the explicit treatment requirements below.
It may not change or invent evidence IDs, fact units, opponent claim target,
stance, core conclusion, sentence functions, sentence order, citations, or
certainty. If the treatment is pro-jargon, the style plan is the baseline plan:
preserve the baseline reasoning relation and change only lexical register.
{treatment_requirements}

	{json.dumps(transformation_plan_for_prompt, indent=2, sort_keys=True)}
	--- END TRANSFORMATION PLAN ---
	"""

    if transformation_plan is not None:
        style_instruction_block = ""
    else:
        style_instruction_block = f"""
Your style notes (do not mention them explicitly):
{style_text}
""" if style_text.strip() else ""

    is_active_fallacy_turn = (
        isinstance(transformation_plan, dict)
        and str(transformation_plan.get("trait_name") or "").startswith("fallacy_trait__")
        and transformation_plan.get("variant") == "active"
        and speech_type in {"argument", "rebuttal"}
    )
    reasoning_grounding_block = (
        """
- Factual grounding: every premise must come from the locked public facts and cited public evidence.
- Reasoning relation: follow only the selected style-plan guideline for the current reasoning slot, even when it creates a less valid or less proportionate connection; do not invent a separate reasoning mechanism.
"""
        if is_active_fallacy_turn
        else """
- Do not overclaim certainty. Present interpretation as inference, not as directly observed fact.
- Keep conclusions proportionate to the evidence and use a valid reasoning relation.
"""
    )
  
    # Build the turn plan separately to keep opening prompts free of argumentative fields.
    if speech_type == "opening":
        turn_plan_block = f"""
--- TURN PLAN ---
{opening_plan_block}
--- END TURN PLAN ---
"""
    else:
        turn_plan_block = f"""
--- TURN PLAN ---
Turn goal: {turn_goal}
Claim / semantic conclusion: {claim_to_defend}
{narrative_plan_block}
{attack_block}
Required move: {required_move}
{concession_block}
Where you should be at the end of this turn: {end_state}
--- END TURN PLAN ---
"""
    if speech_type == "opening":
        prompt_intro = "You are generating one opening factual reconstruction in a detective debate."
        anti_sycophancy_block = """
Opening boundary rules:
- Begin immediately with the selected public facts.
- Do not acknowledge, praise, attack, or answer the opponent.
- Do not frame the reconstruction as an argument, proof, case theory, or position statement.
"""
        requirements_block = """
Requirements:
- Keep content aligned with the ordered opening fact_units and evidence_ids.
- Write approximately 150-220 words and 6-9 sentences.
- This opening must be fully standalone; do not rely on the other opening for setup or event description.
- The concrete event from the central question must appear in the utterance text itself by sentence 2.
- Narrate as an independent third-person observer, not as a witness, family member, story narrator, Speaker A, or Speaker B.
- Convert first-person source narration into explicit third-person references centered on the named suspects and supported relationships.
- Do not use narrator-centered identity labels such as "the narrator", "the narrator's brother", or "the narrator's sister".
- When the source narrator's observation or suspicion matters, identify that person through a concrete supported relationship to the relevant suspect.
- Do not use "I," "me," "my," "we," "us," or "our" for event narration except inside an exact direct quotation spoken by a story character.
- Do not use meta-storytelling phrases like "set the stage", "provided context", "for analysis", or "what is being analyzed".
- utterance must cite at least one [E#] item explicitly; evidence_citations must list every index cited.
- opponent_claim_targeted and attack_move_used must be null.
- Set treatment_realization_note to null; do not self-evaluate treatment realization.
- Do not quote from or reveal the official solution / outcome explanation.
- Return JSON only with no extra text.
"""
    else:
        prompt_intro = "You are generating one turn in an adversarial debate."
        if speech_type == "argument":
            length_requirement = dialogue_structure.argument_length_requirement
        elif speech_type == "rebuttal":
            length_requirement = dialogue_structure.rebuttal_length_requirement
        elif speech_type == "summary":
            length_requirement = "- The utterance must contain exactly two sentences, 25-45 words total; never exceed 55 words."
        else:
            length_requirement = "- Keep the utterance concise and aligned with the speech type."
        anti_sycophancy_block = """
Anti-sycophancy rules (strictly enforced — violations will be rejected):
- NEVER open with positive acknowledgment of your opponent's arguments.
  BANNED openers include (but are not limited to): "Good point", "That's a fair
  point", "My opponent rightly noted", "I agree that", "That's an interesting
  argument", "While my opponent makes a compelling case", "While it is true",
  "Certainly", "You raise a fair point", "I concede that", "To be fair".
- Begin IMMEDIATELY with your own substantive argument or counter-argument.
- NEVER say "great question", "that's fair", or validate questions or statements
  from your opponent in any way. Answer directly or redirect — never affirm.
"""
        if speech_type == "argument":
            turn_function_requirements = dialogue_structure.argument_turn_function_requirements
        elif speech_type == "rebuttal":
            turn_function_requirements = dialogue_structure.rebuttal_turn_function_requirements
        else:
            turn_function_requirements = """
- Perform only the assigned function for the current turn.
"""
        requirements_block = f"""
Requirements:
- Keep content aligned with the assigned turn goal and claim.
- Do not change the core argument trajectory.
{length_requirement}
{turn_function_requirements.rstrip()}
- No new evidence or unrelated argument may be introduced.
- Express style only through the selected reasoning/style instruction — do not mention style instructions.
- utterance must cite at least one [E#] item explicitly; evidence_citations must list every index cited.
- Set treatment_realization_note to null; do not self-evaluate treatment realization.
- Do not quote from or reveal the official solution / outcome explanation.
- Return JSON only with no extra text.
"""

    return f"""
{prompt_intro}
{role_block}

Topic: {topic}
Question: {question}
You are speaker {speaker}.
Your stance: {stance}
Current turn id: {turn_id}

{private_truth_block}

{speech_type_block}
{adaptive_rewrite_block}
{anchor_spec_block}
{locked_contract_block}
{fact_sentence_source_block}
{transformation_block}
--- PUBLIC EVIDENCE BANK ---
Both speakers share access to ONLY the following evidence items.
Cite items by their index, e.g. [E3], [E12], in your utterance.
You may NOT invent facts, names, times, or events not present in these items.
You MAY interpret the same item differently from your opponent in later argumentative turns.

{evidence_text}
--- END PUBLIC EVIDENCE BANK ---
{background_context_block}

{turn_plan_block}

{style_instruction_block}{constraint_block}{feedback_block}{dialogue_feedback_block}

--- DIALOGUE HISTORY ---
{visible_history}
--- END DIALOGUE HISTORY ---

{adversarial_requirements}

Evidence-grounding requirements:
- Ground every factual claim in a specific evidence item — cite it as [E#] in your utterance.
- Internal evidence identifiers may appear only as citation markers like [E35]
  in utterance text and as integers in evidence_citations; never verbalize them
  as ordinary dialogue prose such as "E35 shows" or "Evidence 35 indicates".
- You may NOT introduce facts, names, times, or events not present in the evidence bank.
- Keep observed facts and speculative argumentative inferences separate.
- Do not merge separate evidence items into one definite event, object, or causal relation unless the story explicitly states that relation.
- Any connection not explicitly stated in the story must appear only in reasoning and must be expressed as a possibility, not as a definite factual claim.
- For post-opening turns with a locked content contract, do not introduce content-unit IDs not present in locked_content_contract.
- For post-opening turns with a transformation plan, preserve factual premises, evidence IDs, suspect/event, stance, opponent target, and required conclusion.
- Realize the assigned canonical claim without changing its semantic meaning; do not rebuild it as "[suspect] likely [short event]".
- Private truth is not public evidence. You may use it only to reason about strategy, not as a source for public factual claims.
- Outcome-reference-only revelations must never appear in the public utterance.
- You may NOT invent dialogue, motives, or actions not supported by evidence.
- You may NOT publicly mention answer-option labels, the private answer field, or that any suspect came from answer_options.
- Do not publicly mention a suspect name unless that name or alias is grounded in the evidence bank or appears in the public question.
{reasoning_grounding_block.rstrip()}
- Treat the central question's wrongdoing event as fixed. Do not rename it into something vaguer or into a different offense.
- In openings, state only selected public facts in chronology. Do not state why your side's reading supports or undermines an accusation.

{anti_sycophancy_block}

{selected_reasoning_slot_block}

Return valid JSON with this exact structure:
{{
  "turn_id": {turn_id},
  "speaker": "{speaker}",
  "utterance": "...",
{turn_metadata_fields}
}}

{requirements_block}
{first_pass_length_target_block}
{length_balance_retry_block}
""".strip()
