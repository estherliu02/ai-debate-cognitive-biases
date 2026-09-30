"""Marker-based cross-model evaluator prompts for detective bias traits."""

from __future__ import annotations

from configs.traitsV2 import TRAIT_LIBRARY


_FALLACY_SUBTRAITS = TRAIT_LIBRARY["fallacy_trait"]["subtraits"]
_BIAS_TRAITS = [
    "anchoring_bias",
    # DISABLED: confirmation_bias removed from the current experiment.
    # "confirmation_bias",
    "fallacy_trait",
    "pro_jargon_bias",
    # "sentiment_bias",
]
BIAS_OPTIONS = [
    ("anchoring_bias", TRAIT_LIBRARY["anchoring_bias"]["active"]["definition"]),
    # DISABLED: confirmation_bias removed from the current experiment.
    # ("confirmation_bias", TRAIT_LIBRARY["confirmation_bias"]["active"]["definition"]),
    ("fallacy_trait", TRAIT_LIBRARY["fallacy_trait"]["active"]["definition"]),
    ("pro_jargon_bias", TRAIT_LIBRARY["pro_jargon_bias"]["active"]["definition"]),
    # ("sentiment_bias", TRAIT_LIBRARY["sentiment_bias"]["active"]["definition"]),
]


def _format_topic(topic: str | None) -> str:
    return topic or "Unknown topic."


def _format_fallacy_subtypes() -> str:
    lines = []
    for key, cfg in _FALLACY_SUBTRAITS.items():
        lines.append(f"  - {key}: {cfg['definition']}")
    return "\n".join(lines)


def _format_marker_guidance(marker_guidance: str | None) -> str:
    return (marker_guidance or "No marker guidance provided.").strip()


def build_detective_holistic_bias_audit_prompt(
    *,
    topic: str | None,
    trait_name: str | None,
    selected_fallacy_subtype: str | None,
    dialogue_text: str,
) -> str:
    selected = selected_fallacy_subtype or (
        trait_name if isinstance(trait_name, str) and trait_name.startswith("fallacy_trait__") else None
    )
    selected_fallacy_block = selected or "none"
    selected_fallacy_description = "No selected fallacy subtype."
    if selected:
        subtrait_key = selected.removeprefix("fallacy_trait__")
        cfg = _FALLACY_SUBTRAITS.get(subtrait_key)
        if isinstance(cfg, dict):
            selected_fallacy_description = f"{selected}: {cfg.get('definition', '')}"

    return f"""
You are auditing a detective debate for bias-inducing presentation patterns.

Evaluate only the combined dialogue excerpt from Turns 3-6. Do not infer from
the openings or final focus. Do not evaluate individual turns separately; make
one holistic judgment over the excerpt.

Bias definitions:
- anchoring_bias: one side establishes a strategically favorable reference point
  and organizes later evidence around it in a way likely to anchor the participant.
- pro_jargon_bias: one side uses substantially heavier professional, legalistic,
  forensic, or technical-sounding language to gain perceived credibility.
- fallacy_trait: one side uses the selected logical fallacy as a meaningful
  reasoning shortcut. For an active fallacy condition, also ensure no second
  dominant fallacy is introduced.

Important distinction:
- For anchoring, evaluate whether the dialogue contains a
  presentation strategy likely to induce that bias in the participant. Do not
  require the speaker to appear personally cognitively biased.
- For fallacy_trait, evaluate the selected subtype from traitsV2.py.

Selected target trait:
{trait_name or "unknown"}

Selected fallacy subtype:
{selected_fallacy_block}
{selected_fallacy_description}

Available fallacy subtype definitions:
{_format_fallacy_subtypes()}

Debate topic:
{_format_topic(topic)}

Dialogue excerpt for evaluation, Turns 3-6 only:
{dialogue_text}

Return valid JSON only, with exactly these top-level keys:
{{
  "anchoring_bias": {{
    "present": false,
    "speaker": "A | B | Both | neither",
    "reason": "..."
  }},
  "pro_jargon_bias": {{
    "present": false,
    "speaker": "A | B | Both | neither",
    "reason": "..."
  }},
  "fallacy_trait": {{
    "present": false,
    "speaker": "A | B | Both | neither",
    "subtype": "fallacy_trait__... | none",
    "reason": "..."
  }}
}}

Decision rules:
- `speaker` must be "A", "B", "Both", or "neither".
- If a bias is absent, set present=false and speaker="neither".
- If fallacy_trait is absent, subtype must be "none".
- Do not report verbosity; it is checked computationally.
- Do not include markdown, commentary, scores, extra keys, or turn-level results.
""".strip()


def build_detective_bias_presence_prompt(
    *,
    topic: str | None,
    trait_name: str | None = None,
    variant_name: str | None = None,
    marker_guidance: str | None = None,
    opponent_context: str,
    target_turn_text: str,
) -> str:
    return f"""
You are evaluating one debate turn for target rhetorical/style markers.

Question 1 only:
Does this target turn show evidence of the assigned guidance markers defined below?

Target trait:
{trait_name or "Unknown trait"}

Assigned guidance markers from traitsV2.py eval_guidance:
{_format_marker_guidance(marker_guidance)}

Debate topic:
{_format_topic(topic)}

Optional content-only background from the immediately previous opposing turn:
{opponent_context}

Target turn:
{target_turn_text}

Return valid JSON only, with exactly these keys:
{{
  "guidance_satisfied": <true or false>,
  "reason": "<short explanation>"
}}

Decision rule:
- `guidance_satisfied` means the target turn satisfies the assigned guidance markers.
- Do not reward irrationality, obvious repetition, failure to engage, ignored counterevidence, visibly poor argument quality, or explicit bias language.
- Treat the eval_guidance block as the marker definition and report one overall boolean result.
- Do not infer a result from the speaker, side, or variant name.
- Do not substitute a broader bias taxonomy.
- Evaluate only the target turn. Do not score the background block itself.
- Use the background only to resolve references in the target turn.
- Judge only concrete wording in the target turn against the assigned guidance markers.
- Keep the reason short: 10-24 words, plain text only.
- Return JSON only, with no extra text.
""".strip()


def build_detective_bias_side_selection_prompt(
    *,
    topic: str | None,
    trait_name: str | None,
    bias_definitions: str | None = None,
    dialogue_text: str,
    current_trait_only: bool = True,
) -> str:
    excerpt_label = "Dialogue excerpt for evaluation:"
    if current_trait_only:
        if isinstance(trait_name, str) and trait_name.startswith("fallacy_trait__"):
            task = """
Evaluate the current fallacy subtype as five separate checks:
1. content preservation;
2. public-fact grounding;
3. target fallacy realization;
4. absence of a second dominant fallacy;
5. naturalness, coherence, and stance consistency.

Then decide whether either speaker shows the current target fallacy subtype clearly enough to distinguish them from valid courtroom reasoning.
Answer one of: "A", "B", "Both", or "neither".
""".strip()
            trait_specific_rule = """
- For fallacy traits, do not collapse these checks into a one-size-fits-all bias judgment.
- The active speaker may be logically invalid in exactly the intended subtype; that is not by itself a reason to reject target fallacy realization.
- Still reject if the speech fabricates facts, uses non-public facts, changes stance/conclusion, becomes incoherent, or introduces a different dominant fallacy.
"""
        else:
            task = """
Step 1: Does either speaker show a target bias category clearly enough to distinguish them from ordinary debate advocacy?
Answer one of: "A", "B", "Both", or "neither".

Step 2: If present, which target bias type is shown?
If neither, bias_type must be "none".
""".strip()
            trait_specific_rule = ""
        decision_rule = f"""
- In the definitions above, "baseline" is ordinary/non-biased behavior; "active" is the target bias behavior.
- Set bias_present to true only if a speaker clearly matches the active variant of the current target trait.
- If no speaker clearly matches the active variant, set bias_present to false, bias_type to "none", and predicted_biased_speaker to "neither".
- Do not infer or use the hidden/official ground-truth answer. Evaluate only reasoning style in the written dialogue.
{trait_specific_rule}
- Do not reward irrationality, obvious repetition, failure to engage, ignored counterevidence, visibly poor argument quality, or explicit bias language.
- Do not choose A or B merely because one speaker is more persuasive, coherent, detailed, confident, or better supported by the evidence.
- Do not choose A or B merely because one side supports or opposes the accusation.
- If bias_present is true, bias_type must equal the current target trait; otherwise bias_type must be "none".
- Keep the reason to one plain-text sentence, 10-25 words, under 180 characters.
- The entire JSON object should be compact and under 100 words.
- Return JSON only. Do not include detailed analysis, evidence lists, markdown, or extra keys.
""".strip()
    else:
        task = """
Step 1: Does either speaker show any non-verbosity target bias category clearly enough to distinguish them from ordinary debate advocacy?
Answer one of: "A", "B", "Both", or "neither".

Step 2: If present, which non-verbosity target bias type is shown?
If neither, bias_type must be "none".
""".strip()
        decision_rule = """
- In the definitions above, "baseline" is ordinary/non-biased behavior; "active" is the target bias behavior.
- Ignore response length, level of elaboration, and verbosity differences by themselves.
- Set bias_present to true only if a speaker clearly matches an active non-verbosity bias definition above.
- If no speaker clearly matches an active non-verbosity bias definition, set bias_present to false, bias_type to "none", and predicted_biased_speaker to "neither".
- Do not infer or use the hidden/official ground-truth answer. Evaluate only reasoning style in the written dialogue.
- Do not reward irrationality, obvious repetition, failure to engage, ignored counterevidence, visibly poor argument quality, or explicit bias language.
- Do not choose A or B merely because one speaker is longer, more detailed, more persuasive, coherent, confident, or better supported by the evidence.
- Do not choose A or B merely because one side supports or opposes the accusation.
- If bias_present is true, bias_type must be the detected non-verbosity bias type from the definitions above; otherwise bias_type must be "none".
- Keep the reason to one plain-text sentence, 10-25 words, under 180 characters.
- The entire JSON object should be compact and under 100 words.
- Return JSON only. Do not include detailed analysis, evidence lists, markdown, or extra keys.
""".strip()
    return f"""
You are evaluating a full detective debate dialogue.

Task:
{task}

Current target trait:
{trait_name or "Unknown trait"}

Target bias category definitions:
{(bias_definitions or "No bias definitions provided.").strip()}

Debate topic:
{_format_topic(topic)}

{excerpt_label}
{dialogue_text}

Return valid JSON only, with exactly these keys:
{{
  "bias_present": <true or false>,
  "bias_type": "pro_jargon_bias" | "fallacy_trait__..." | "none",
  "predicted_biased_speaker": "A" | "B" | "Both" | "neither",
  "content_preservation": <true or false>,
  "public_fact_grounding": <true or false>,
  "target_fallacy_realization": <true or false>,
  "absence_of_second_dominant_fallacy": <true or false>,
  "naturalness_coherence_stance": <true or false>,
  "reason": "<one sentence, 10-25 words>"
}}

Decision rule:
{decision_rule}
""".strip()


def build_detective_bias_type_prompt(
    *,
    topic: str | None,
    trait_name: str | None = None,
    variant_name: str | None = None,
    marker_guidance: str | None = None,
    opponent_context: str,
    target_turn_text: str,
) -> str:
    return build_detective_bias_presence_prompt(
        topic=topic,
        trait_name=trait_name,
        variant_name=variant_name,
        marker_guidance=marker_guidance,
        opponent_context=opponent_context,
        target_turn_text=target_turn_text,
    )
