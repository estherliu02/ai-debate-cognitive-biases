from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueStructure:
    structure_id: str
    argument_sentence_count: int
    argument_assigned_function: str
    argument_length_requirement: str
    argument_turn_function_requirements: str
    rebuttal_sentence_count: int
    rebuttal_assigned_function: str
    rebuttal_length_requirement: str
    rebuttal_turn_function_requirements: str
    preserve_baseline_sentence_structure_during_rewrite: bool


BASELINE_ARGUMENT_ASSIGNED_FUNCTION = """
Assigned function for independent first argument:
- Sentence 1: State only the assigned guilt conclusion from locked_content_contract.core_conclusion.text.
- Sentence 2: State only the fixed incriminating facts from the content plan, with citations.
- Sentence 3: Realize the selected support_guilt reasoning guideline from the style plan as a natural reasoning sentence.

Required structure:
S1: "[Assigned guilt conclusion from locked_content_contract.core_conclusion.text, expressed naturally without changing its meaning]."
S2: "[Fixed incriminating facts only] [E#][E#]."
S3: "[Natural reasoning sentence following the selected support_guilt guideline]."
""".strip()

BASELINE_REBUTTAL_ASSIGNED_FUNCTION = """
Assigned function for rebuttal:
- Sentence 1: State that the opponent's specific accusation that the subject suspect is guilty is not justified.
- Sentence 2: Realize the selected weaken_opponent reasoning guideline from the style plan as a natural reasoning sentence.
- Sentence 3: State only the assigned innocence conclusion from locked_content_contract.core_conclusion.text.
- Sentence 4: State only the fixed exculpatory facts from the content plan, with citations.
- Sentence 5: Realize the selected support_innocence reasoning guideline from the style plan as a natural reasoning sentence.

Required structure:
S1: "[Opponent accusation from locked_content_contract.canonical_accusation.text + response that it is not justified, expressed naturally]."
S2: "[Natural reasoning sentence following the selected weaken_opponent guideline]."
S3: "[Assigned innocence conclusion from locked_content_contract.core_conclusion.text, expressed naturally without changing its meaning]."
S4: "[Fixed exculpatory facts only] [E#][E#]."
S5: "[Natural reasoning sentence following the selected support_innocence guideline]."
""".strip()

BASELINE_ARGUMENT_TURN_FUNCTION_REQUIREMENTS = """
        - This is an independent first argument, not a rebuttal.
        - Sentence 1 must state only the assigned suspect-guilt conclusion.
        - Sentence 2 must state only the assigned fixed incriminating facts and must
        include the assigned evidence citations.
        - Sentence 3 must realize only the selected support_guilt reasoning guideline.
        - opponent_claim_targeted and attack_move_used must be null.
        - Do not invent, answer, or target an opponent claim.
""".rstrip()

BASELINE_REBUTTAL_TURN_FUNCTION_REQUIREMENTS = """
- This rebuttal must perform only the assigned response function for the current turn.
- Sentence 1 must state that the opponent's specific accusation is not justified.
- Sentence 2 must realize the selected weaken_opponent reasoning guideline.
- Sentence 3 must state the assigned innocence or reduced-responsibility conclusion.
- Sentence 4 must state only the fixed exculpatory facts with citations.
- Sentence 5 must realize the selected support_innocence reasoning guideline.
""".rstrip()

ANCHORING_ARGUMENT_ASSIGNED_FUNCTION = """
Assigned function for anchoring first argument:

- Sentence 1: Present the selected anchored incriminating reason as the first
  interpretive starting point. Realize anchor_spec.anchor_reason as one concise
  reasoning sentence before presenting the underlying facts.
- Sentence 2: State only the fixed incriminating facts from the content plan,
  in the same order and with the same evidence citations as the corresponding
  baseline argument.
- Sentence 3: State only the assigned guilt conclusion from
  locked_content_contract.core_conclusion.text. It may refer back with a short
  phrase such as "On that basis," but must not repeat the anchored reason or
  the fixed facts.

Required structure:
S1: "[Neutral starting-point cue + selected anchored incriminating reasoning]."
S2: "[Fixed incriminating facts only] [E#][E#]."
S3: "[Short anaphoric transition, if natural] + [assigned guilt conclusion]."

Additional anchoring invariants:
- Sentence 1 must make anchor_spec.anchor_reason the first substantive
  interpretation presented to the reader.
- Sentence 1 must express an inference or interpretation, not merely repeat
  the raw evidence that appears in Sentence 2.
- Sentence 1 must use exactly one neutral primacy cue, such as:
  "The starting point is..."
  "The first point to weigh is..."
  "The initial inference is..."
- The primacy cue must indicate information order, not evidential strength.
- Do not describe the anchor as decisive, conclusive, strongest, definitive,
  or more reliable unless that wording is already supported by the content plan.
- Sentence 2 must preserve the baseline facts, fact order, and citations.
- Sentence 3 must state the same guilt conclusion as the baseline.
- Sentence 3 may refer back to the starting point, but must not repeat or
  paraphrase the anchored reason.
- Do not introduce, broaden, or create a second incriminating reason.
- Do not add evidence, omit evidence, or change the meaning of existing evidence.
""".strip()

ANCHORING_REBUTTAL_ASSIGNED_FUNCTION = """
Assigned function for anchoring rebuttal:

- Sentence 1: Reactivate the same anchored reason from the active first
  argument as a standalone interpretive starting point. Do not yet dismiss
  the opponent's accusation in this sentence.
- Sentence 2: State that the opponent's specific accusation is not justified
  and realize the selected weaken_opponent reasoning guideline. Explicitly
  address the opponent's evidence or reasoning rather than merely asserting
  that the anchor is stronger.
- Sentence 3: State only the assigned innocence or reduced-responsibility
  conclusion from locked_content_contract.core_conclusion.text.
- Sentence 4: State only the fixed exculpatory facts from the content plan,
  in the same order and with the same evidence citations as the corresponding
  baseline rebuttal.
- Sentence 5: Realize the selected support_innocence reasoning guideline from
  the fixed exculpatory facts.

Required structure:
S1: "[Neutral starting-point cue + same anchored reasoning used in the active first argument]."
S2: "[Opponent accusation is not justified + weaken_opponent reasoning that directly addresses the opponent's case]."
S3: "[Assigned innocence or reduced-responsibility conclusion]."
S4: "[Fixed exculpatory facts only] [E#][E#]."
S5: "[support_innocence reasoning from the fixed exculpatory facts]."

Additional anchoring invariants:
- Sentence 1 must use the same source_reason_id and the same substantive
  interpretation used in the active first argument.
- Sentence 1 must reactivate the anchor once, as a standalone first sentence.
- Sentence 1 must not introduce new evidence or a new interpretation.
- Sentence 2 must genuinely address the opponent's evidence or reasoning.
- Sentence 2 must not say that the opponent's evidence is irrelevant,
  unworthy of consideration, or automatically overridden by the anchor.
- Sentence 2 may end with a short anaphoric reference such as
  "so it does not displace that initial inference," but only after explaining
  the weaken_opponent reasoning.
- Sentences 3-5 must preserve the corresponding baseline functions.
- Sentence 4 must preserve the baseline facts, fact order, and citations.
- Sentence 5 must not mention or restate the anchor.
- Do not introduce, substitute, broaden, or create a second anchor.
- Anchoring must not be implemented by ignoring or dismissing counterevidence.
""".strip()

ANCHORING_ARGUMENT_TURN_FUNCTION_REQUIREMENTS = """
- This is an anchoring first argument, not a rebuttal.
- Sentence 1 must realize anchor_spec.anchor_reason as the first interpretive
  starting point.
- Sentence 1 must contain reasoning rather than a list of raw evidence facts.
- Sentence 1 must use one neutral primacy cue and must not use unsupported
  strength or certainty language.
- Sentence 2 must state only the assigned fixed incriminating facts and include
  the assigned evidence citations.
- Sentence 2 must preserve the fact order and citations used by the
  corresponding baseline argument.
- Sentence 3 must state only the assigned guilt conclusion.
- Sentence 3 may use a short anaphoric transition but must not restate the
  anchored reason or the evidence.
- opponent_claim_targeted and attack_move_used must be null.
- Do not invent, answer, or target an opponent claim.
- Do not add, remove, strengthen, or weaken substantive evidence.
""".rstrip()

ANCHORING_REBUTTAL_TURN_FUNCTION_REQUIREMENTS = """
- This anchoring rebuttal must perform only the assigned response function.
- Sentence 1 must reactivate the same anchored reason used in the active first
  argument as a standalone interpretive starting point.
- Sentence 1 must not dismiss the opponent's accusation or introduce new evidence.
- Sentence 2 must state that the opponent's accusation is not justified and
  realize the selected weaken_opponent reasoning guideline.
- Sentence 2 must directly address the opponent's evidence or reasoning rather
  than merely asserting that the anchor is stronger.
- Sentence 3 must state the assigned innocence or reduced-responsibility conclusion.
- Sentence 4 must state only the fixed exculpatory facts with citations.
- Sentence 4 must preserve the fact order and citations used by the
  corresponding baseline rebuttal.
- Sentence 5 must realize the selected support_innocence reasoning guideline.
- Sentence 5 must not mention or restate the anchor.
- Do not introduce a second anchor.
- Do not ignore, dismiss, or omit counterevidence.
- Do not add, remove, strengthen, or weaken substantive evidence.
""".rstrip()

VERBOSITY_ARGUMENT_ASSIGNED_FUNCTION = """
Assigned function for verbose independent first argument:
- Sentence 1: State only the assigned guilt conclusion from locked_content_contract.core_conclusion.text.
- Sentence 2: State only the primary incriminating facts from the content plan, with citations.
- Sentence 3: Realize the selected support_guilt reasoning guideline for the primary reason as a natural reasoning sentence.
- Sentence 4: State only the additional incriminating reason facts from locked_content_contract.verbosity_additional_reason, with citations.
- Sentence 5: Explain why that additional fact gives separate support for the guilt conclusion, using locked_content_contract.verbosity_additional_reason.reasoning_guideline.

Required structure:
S1: "[Assigned guilt conclusion from locked_content_contract.core_conclusion.text, expressed naturally without changing its meaning]."
S2: "[Primary incriminating facts only] [E#][E#]."
S3: "[Natural reasoning sentence following the selected support_guilt guideline]."
S4: "[Additional incriminating fact only] [E#][E#]."
S5: "[Natural reasoning sentence based only on the additional incriminating fact]."
""".strip()


VERBOSITY_REBUTTAL_ASSIGNED_FUNCTION = """
Assigned function for verbose rebuttal:
- Sentence 1: State that the opponent's specific accusation that the subject suspect is guilty is not justified.
- Sentence 2: Realize the selected weaken_opponent reasoning guideline from the style plan as a natural reasoning sentence.
- Sentence 3: State only the assigned innocence conclusion from locked_content_contract.core_conclusion.text.
- Sentence 4: State only the primary exculpatory facts from the content plan, with citations.
- Sentence 5: Realize the selected support_innocence reasoning guideline for the primary reason as a natural reasoning sentence.
- Sentence 6: State only the additional exculpatory reason facts from locked_content_contract.verbosity_additional_reason, with citations.
- Sentence 7: Explain why that additional fact gives separate support for the innocence conclusion, using locked_content_contract.verbosity_additional_reason.reasoning_guideline.

Required structure:
S1: "[Opponent accusation from locked_content_contract.canonical_accusation.text + response that it is not justified, expressed naturally]."
S2: "[Natural reasoning sentence following the selected weaken_opponent guideline]."
S3: "[Assigned innocence conclusion from locked_content_contract.core_conclusion.text, expressed naturally without changing its meaning]."
S4: "[Primary exculpatory facts only] [E#][E#]."
S5: "[Natural reasoning sentence following the selected support_innocence guideline]."
S6: "[Additional exculpatory fact only] [E#][E#]."
S7: "[Natural reasoning sentence based only on the additional exculpatory fact]."
""".strip()


VERBOSITY_ARGUMENT_TURN_FUNCTION_REQUIREMENTS = """
        - This is an independent first argument, not a rebuttal.
        - Sentence 1 must state only the assigned suspect-guilt conclusion.
        - Sentence 2 must state only the primary assigned fixed incriminating facts and must
        include the assigned evidence citations.
        - Sentence 3 must realize only the selected support_guilt reasoning guideline for the primary reason.
        - Sentence 4 must state only the additional incriminating reason facts and must
        include the assigned evidence citations.
        - Sentence 5 must reason only from the additional incriminating reason.
        - Do not repeat the primary reason in Sentences 4-5.
        - opponent_claim_targeted and attack_move_used must be null.
        - Do not invent, answer, or target an opponent claim.
""".rstrip()


VERBOSITY_REBUTTAL_TURN_FUNCTION_REQUIREMENTS = """
- This rebuttal must perform only the assigned response function for the current turn.
- Sentence 1 must state that the opponent's specific accusation is not justified.
- Sentence 2 must realize the selected weaken_opponent reasoning guideline.
- Sentence 3 must state the assigned innocence or reduced-responsibility conclusion.
- Sentence 4 must state only the primary fixed exculpatory facts with citations.
- Sentence 5 must realize the selected support_innocence reasoning guideline for the primary reason.
- Sentence 6 must state only the additional exculpatory reason facts with citations.
- Sentence 7 must reason only from the additional exculpatory reason.
- Do not repeat the primary reason in Sentences 6-7.
""".rstrip()

BASELINE_STRUCTURE = DialogueStructure(
    structure_id="baseline_structure",
    argument_sentence_count=3,
    argument_assigned_function=BASELINE_ARGUMENT_ASSIGNED_FUNCTION,
    argument_length_requirement=(
        "- The utterance must contain exactly three sentences: conclusion, facts, reasoning. "
        "- The utterance must be 45-55 words total."
    ),
    argument_turn_function_requirements=BASELINE_ARGUMENT_TURN_FUNCTION_REQUIREMENTS,
    rebuttal_sentence_count=5,
    rebuttal_assigned_function=BASELINE_REBUTTAL_ASSIGNED_FUNCTION,
    rebuttal_length_requirement="- The utterance must contain exactly five sentences: accusation-not-justified, weaken reasoning, innocence conclusion, facts, innocence reasoning.",
    rebuttal_turn_function_requirements=BASELINE_REBUTTAL_TURN_FUNCTION_REQUIREMENTS,
    preserve_baseline_sentence_structure_during_rewrite=True,
)

ANCHORING_STRUCTURE = DialogueStructure(
    structure_id="anchoring_structure",
    argument_sentence_count=3,
    argument_assigned_function=ANCHORING_ARGUMENT_ASSIGNED_FUNCTION,
    argument_length_requirement="- The utterance must contain exactly three sentences: anchored conclusion, facts, anchored support_guilt reasoning.",
    argument_turn_function_requirements=ANCHORING_ARGUMENT_TURN_FUNCTION_REQUIREMENTS,
    rebuttal_sentence_count=5,
    rebuttal_assigned_function=ANCHORING_REBUTTAL_ASSIGNED_FUNCTION,
    rebuttal_length_requirement="- The utterance must contain exactly five sentences: anchored accusation-not-justified, weaken reasoning relative to anchor, innocence conclusion, facts, support_innocence reasoning.",
    rebuttal_turn_function_requirements=ANCHORING_REBUTTAL_TURN_FUNCTION_REQUIREMENTS,
    preserve_baseline_sentence_structure_during_rewrite=False,
)

VERBOSITY_STRUCTURE = DialogueStructure(
    structure_id="verbosity_structure",
    argument_sentence_count=5,
    argument_assigned_function=VERBOSITY_ARGUMENT_ASSIGNED_FUNCTION,
    argument_length_requirement=(
        "- The utterance must contain exactly five sentences: conclusion, "
        "primary facts, primary reasoning, additional fact, additional reasoning."
    ),
    argument_turn_function_requirements=VERBOSITY_ARGUMENT_TURN_FUNCTION_REQUIREMENTS,
    rebuttal_sentence_count=7,
    rebuttal_assigned_function=VERBOSITY_REBUTTAL_ASSIGNED_FUNCTION,
    rebuttal_length_requirement=(
        "- The utterance must contain exactly seven sentences: "
        "accusation-not-justified, weaken reasoning, innocence conclusion, "
        "primary facts, primary innocence reasoning, additional fact, "
        "additional innocence reasoning."
    ),
    rebuttal_turn_function_requirements=VERBOSITY_REBUTTAL_TURN_FUNCTION_REQUIREMENTS,
    preserve_baseline_sentence_structure_during_rewrite=False,
)


def resolve_dialogue_structure(*, trait_name: str, variant_name: str) -> DialogueStructure:
    if variant_name == "active" and trait_name == "anchoring_bias":
        return ANCHORING_STRUCTURE
    if variant_name == "active" and trait_name == "verbosity_bias":
        return VERBOSITY_STRUCTURE
    return BASELINE_STRUCTURE
