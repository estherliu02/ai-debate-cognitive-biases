import json


def _prompt_bias_definition(entry: dict) -> dict:
    return {
        key: value
        for key, value in (entry or {}).items()
        if key != "examples" or value is not None
    }


def build_detective_style_reasoning_prompt(
    *,
    content_plan,
    role: str,
    trait_name: str,
    contexts: dict[str, dict],
    bias_keys: list[str],
    trait_library: dict,
) -> str:
    setup = content_plan.debate_setup or {}
    fallacy_subtype = trait_name.removeprefix("fallacy_trait__") if trait_name.startswith("fallacy_trait__") else None
    if fallacy_subtype:
        selected_bias_definition = trait_library["fallacy_trait"]["subtraits"][fallacy_subtype]
        baseline_bias_definition = trait_library["fallacy_trait"]["baseline"]
    else:
        trait_cfg = trait_library.get(trait_name, {})
        selected_bias_definition = trait_cfg.get("active", {})
        baseline_bias_definition = trait_cfg.get("baseline", {})
    slot_payload = []
    for slot_id, context in contexts.items():
        local_purpose = context.get("local_purpose") or context.get("slot_local_purpose")
        slot_entry = {
            "slot_id": slot_id,
            "local_purpose": local_purpose,
            "speaker_role": context.get("speaker_role"),
            "role_turn_id": context.get("role_turn_id"),
            "turn_type": context.get("turn_type"),
            "sentence_index": context.get("sentence_index"),
            "job": context.get("job"),
            "subject_suspect": context.get("subject_suspect"),
            "opponent_source_role": context.get("opponent_source_role"),
            "accusation_being_weakened": context.get("accusation_being_weakened"),
            "claim_target": context.get("claim_target"),
            "required_conclusion": context.get("required_conclusion") or context.get("core_conclusion"),
            "evidence_ids": context.get("evidence_ids") or [],
            "fact_units": context.get("fact_units") or [],
        }
        slot_payload.append(slot_entry)
    selected_prompt_definition = _prompt_bias_definition(selected_bias_definition)
    baseline_prompt_definition = _prompt_bias_definition(baseline_bias_definition)
    semantic_fields = sorted(
        set(selected_prompt_definition)
        | set(baseline_prompt_definition)
    )
    semantic_fields_text = ", ".join(semantic_fields)
    return f"""
Generate context-specific reasoning guidelines for a pairwise detective debate
style plan.

The content plan is fixed and separately supplies the facts, evidence IDs,
suspects, roles, turn IDs, sentence indices, claim targets, and conclusions.
Use those details only to understand the task. Do not repeat them in the
style-plan output.

Each slot has exactly one local_purpose:
- support_guilt: apply the selected bias mechanism to connect the fixed
  incriminating facts to the assigned guilt conclusion.
- weaken_opponent: apply the selected bias mechanism to weaken the opponent's
  specific inference from their fixed incriminating facts to the suspect's
  guilt. Always target the opponent's specific accusation and the reasoning
  that connects the fixed facts to that accusation.
- support_innocence: apply the selected bias mechanism to connect the fixed
  exculpatory facts to the assigned innocence conclusion.

Do not look for predefined application modes or local-purpose instructions in
the trait library. Derive how the selected bias operates for each local_purpose
from the available selected-bias semantic fields ({semantic_fields_text}) and
the current case-specific slot context.

Generate paired baseline and active guidelines with this separation:
- Baseline: fixed facts -> side-level evidentiary assessment -> conclusion.
- Active: same fixed facts -> bias-specific supplementary reasoning -> same
  conclusion.

Baseline guidelines must stay at the fact/side evidentiary-assessment level.
They may only say how the fixed facts count for or against the relevant side:
- support_guilt: the fixed incriminating facts support the suspect's guilt.
- weaken_opponent: the opponent's cited facts do not directly establish the
  suspect's guilt.
- support_innocence: the fixed exculpatory facts weigh against the suspect's
  guilt.

Baseline guidelines must not introduce motive, opportunity, personality,
stable traits, person types, behavioral patterns, causal explanations,
hypothetical mechanisms, or general rules. Do not make baseline more clever
than a direct assessment of what the fixed facts do for the side.

Active guidelines must preserve the same fixed facts and same required
conclusion, but add the supplementary reasoning mechanism specified by the
selected active bias definition and its contrast_with_baseline. They should
guide dialogue rollout toward that requested bias without supplying
dialogue-ready wording.

For active weaken_opponent guidelines, do not merely say that the opponent's
cited facts are insufficient. Start from one weakness, ambiguity, or gap in the
opponent's specific inference, generalize it into a category-level claim that
this kind of accusation, evidence pattern, or reasoning approach is generally
unreliable, and then apply that generalization back to the current accusation.

For every bias type, active guidelines should describe a naturally persuasive
shortcut a competent courtroom advocate could use. Prefer subtle, plausible
overreach over explicit absolutes or cartoonishly invalid claims. Baseline and
active guidelines should differ by reasoning mechanism, not polish or
competence.

Reasoning direction must match local_purpose. For support_innocence, do not
guide the rollout toward guilt-supporting reasoning.

Guidelines must be specific to this case, the supplied facts, the suspect, and
the local purpose, but the output must not repeat facts, evidence IDs, suspect
names, conclusions, roles, turn IDs, sentence indices, citations, rhetorical
transitions, or final polished claims.

Keep each guideline concise: 8-22 words. It should name the reasoning mechanism
only, not a full sentence that could be pasted into the debate. If a bias
mechanism is complex, compress it to the shortest case-specific instruction.

Do not write final dialogue sentences. Do not include citation markers. Do not
include a sentence that can be copied directly into the debate.

Case setup:
{json.dumps(setup, indent=2, sort_keys=True)}

Trait: {trait_name}
Bias keys to generate: {bias_keys}
Selected active bias definition:
{json.dumps(selected_prompt_definition, indent=2, sort_keys=True)}

Baseline bias definition:
{json.dumps(baseline_prompt_definition, indent=2, sort_keys=True)}

Reasoning slots:
{json.dumps(slot_payload, indent=2, sort_keys=True)}

Return JSON only:
{{
  "guidelines": {{
    "slot_id": {{
      "baseline": "Context-specific valid reasoning guideline, not final dialogue.",
      "biased": "Context-specific biased reasoning guideline, not final dialogue."
    }}
  }}
}}
""".strip()


def build_style_prompt(agent_name: str, trait_name: str, variant_name: str, trait_rules: list[str], eval_feedback: str | None = None) -> str:
    bullets = "\n".join(f"- {r}" for r in trait_rules)
    treatment_constraints = ""
    if trait_name == "verbosity_bias":
        treatment_constraints = """
For verbosity_bias, baseline turns may use only primary locked reasons. Active turns may
use additional reasons only when the rollout locked_content_contract explicitly includes
verbosity_additional_reason for the current argument or rebuttal turn.
It must not add examples, scenarios, alternative explanations, or claims outside that contract.
It must not apply normal verbosity word ranges to final_focus.
"""
    elif trait_name == "pro_jargon_bias":
        treatment_constraints = """
For pro_jargon_bias, do not create sentence-specific jargon targets or change reasoning
structure in the style plan. Use the same baseline structure and baseline reasoning plan;
the rollout rewrite stage applies the treatment later.
"""
    elif trait_name == "anchoring_bias":
        treatment_constraints = """
For anchoring_bias, the plan must select one existing favorable interpretation unit as
anchor_unit_id, put that unit first in the active speaker's first eligible presentation,
frame it as the starting point/central comparison/most diagnostic issue, and return to
the same anchor in later active turns. It must preserve evidence IDs, fact units,
interpretation units, inference edges, certainty, concessions, and conclusion exactly.
Do not apply this treatment to final_focus.
"""
    # DISABLED: confirmation_bias removed from the current experiment.
    # elif trait_name == "confirmation_bias": ...
    elif trait_name.startswith("fallacy_trait__"):
        treatment_constraints = """
For fallacy_trait subtypes, traitsV2.py provides only semantic definitions.
Derive operational constraints from the fixed pipeline contract: preserve public
evidence IDs, factual premises, argumentative topic, opponent claim target,
stance, and core conclusion. Active may change only the reasoning relation in
the designated reasoning sentences. It must not fabricate facts or introduce a
second dominant fallacy.
Do not apply fallacy transformations to openings or final_focus.
"""
    feedback_block = (
        f"\nA previous dialogue using this style plan was rejected for the following reason:\n"
        f"{eval_feedback}\n"
        f"Generate stronger, more concrete behavioral rules that directly fix this failure."
    ) if eval_feedback else ""
    return f"""
You are writing a transformation plan for one debate speaker.

Agent: {agent_name}
Trait: {trait_name}
Variant: {variant_name}

Legacy trait notes, for context only. Do not turn these into generic "be more biased" instructions:
{bullets}
{feedback_block}
{treatment_constraints}
Return valid JSON with this structure:
{{
  "trait_name": "{trait_name}",
  "variant": "{variant_name}",
  "anchor_unit_id": null,
  "fallacy_subtype": null,
  "hook_unit_ids": [],
  "branch_metadata": null,
  "ordered_unit_ids": [],
  "foreground_unit_ids": [],
  "background_unit_ids": [],
  "repeat_unit_ids": [],
  "allowed_surface_operations": ["..."],
  "allowed_inference_operations": ["..."]
}}

The plan may decide ordering, foregrounding/backgrounding, repetition, compression or expansion,
lexical register, discourse framing, and whether a required unit is stated directly or conceded briefly.
It may not change or invent evidence IDs, fact units, substantive interpretation units, opponent claim
target, stance, or core conclusion.
""".strip()
