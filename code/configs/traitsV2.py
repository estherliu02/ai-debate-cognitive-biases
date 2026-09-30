## Semantic bias definitions for the detective generation pipeline.

import random

_REQUIRED_BIAS_FIELDS = {
    "name",
    "definition",
    "eval_guidance",
    "examples",
    "contrast_with_baseline",
}


def _bias(
    *,
    name: str,
    definition: str,
    eval_guidance: str,
    examples: str | None,
    contrast_with_baseline: str,
) -> dict:
    return {
        "name": name,
        "definition": definition,
        "eval_guidance": eval_guidance,
        "examples": examples,
        "contrast_with_baseline": contrast_with_baseline,
    }


def _validate_bias_entry(label: str, entry: dict) -> None:
    keys = set(entry)
    if keys != _REQUIRED_BIAS_FIELDS:
        raise ValueError(
            f"Bias definition {label!r} must contain exactly {_REQUIRED_BIAS_FIELDS}; got {keys}."
        )
    for key in _REQUIRED_BIAS_FIELDS:
        value = entry.get(key)
        if key == "examples":
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"Bias definition {label!r} has invalid {key!r}; expected non-empty string or None."
                )
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Bias definition {label!r} has empty {key!r}.")


def validate_trait_library(trait_library: dict | None = None) -> None:
    library = trait_library or TRAIT_LIBRARY
    for trait_name, trait_cfg in library.items():
        if not isinstance(trait_cfg, dict):
            raise ValueError(f"Trait {trait_name!r} must be an object.")
        for variant in ("baseline", "active"):
            _validate_bias_entry(f"{trait_name}.{variant}", trait_cfg[variant])
        subtraits = trait_cfg.get("subtraits")
        if subtraits is not None:
            if trait_name != "fallacy_trait" or not isinstance(subtraits, dict) or not subtraits:
                raise ValueError("Only fallacy_trait may define non-empty subtraits.")
            for subtrait_key, subtrait_cfg in subtraits.items():
                _validate_bias_entry(f"{trait_name}.{subtrait_key}", subtrait_cfg)


def select_random_fallacy_subtrait(trait_config, seed=None):
    rng = random.Random(seed)
    subtraits = trait_config["subtraits"]
    selected_key = rng.choice(list(subtraits.keys()))
    selected_subtrait = subtraits[selected_key]
    return {
        "subtrait_key": selected_key,
        "subtrait_name": selected_subtrait["name"],
        "name": selected_subtrait["name"],
        "definition": selected_subtrait["definition"],
        "eval_guidance": selected_subtrait["eval_guidance"],
        "examples": selected_subtrait["examples"],
        "contrast_with_baseline": selected_subtrait["contrast_with_baseline"],
    }


FALLACY_EXAMPLES = {
    "hasty_generalization": (
        "Everyday form: 'Even though it's only the first day, I can tell this is "
        "going to be a boring course.'\n"
        "Courtroom form: 'The defendant lost his temper once with a coworker. That "
        "single outburst tells you exactly who he is and what he was capable of "
        "that night.'"
    ),
    "slippery_slope": (
        "Everyday form: 'If we ban Hummers because they are bad for the environment, "
        "eventually the government will ban all cars, so we should not ban Hummers.'\n"
        "Courtroom form: 'If you accept this hearsay today, tomorrow no out-of-court "
        "statement is off limits, and the trial after this one will be decided on "
        "rumor. There is no limiting principle to what they are asking.'"
    ),
    "circular_argument": (
        "Everyday form: 'This news site is trustworthy because it publishes reliable "
        "information, and we know its information is reliable because it comes from "
        "a trustworthy news site.'\n"
        "Courtroom form: 'The officer's account is reliable because it accurately "
        "describes what happened. We know it accurately describes what happened "
        "because the officer is a reliable witness.'"
    ),
    "straw_man": (
        "Everyday form: 'Alex says we should reduce homework on weekends, so "
        "apparently Alex thinks students should never study outside class.'\n"
        "Courtroom form: 'The defense argues that one timestamp is uncertain. What "
        "they are really asking you to believe is that no digital record can ever be "
        "trusted, so every piece of electronic evidence must be ignored.'"
    ),
    "false_dilemma": (
        "Everyday form: 'We can either stop using cars or destroy the earth.'\n"
        "Courtroom form: 'Either the defendant was at the scene exactly as the witness "
        "says, or the witness invented the entire encounter. Since there is no reason "
        "for a complete fabrication, the defendant must have been there.'"
    ),
    "post_hoc_ergo_propter_hoc": (
        "Everyday form: 'I drank bottled water and now I am sick, so the water must "
        "have made me sick.'\n"
        "Courtroom form: 'The witness changed her story only after meeting with "
        "defense counsel. Before that meeting, her account never wavered. The timing "
        "tells you exactly why it changed.'"
    ),
    "red_herring": (
        "Everyday form: 'The level of mercury in seafood may be unsafe, but what "
        "will fishers do to support their families?'\n"
        "Courtroom form: 'Counsel presses the gap in the timeline, but the real "
        "question is why this department has failed this community for a decade. "
        "That is what should trouble you today.'"
    ),
    "genetic_fallacy": (
        "Everyday form: 'This proposal originated at our rival school, so it cannot "
        "be a good idea for ours.'\n"
        "Courtroom form: 'This forensic procedure was first developed by a laboratory "
        "funded by insurers. That origin alone discredits the result; there is no need "
        "to examine the procedure or its validation.'"
    ),
    "ad_populum_bandwagon": (
        "Everyday form: 'Everyone in my class uses this study app, so it must be the "
        "most effective way to learn.'\n"
        "Courtroom form: 'The whole community already knows who committed this crime. "
        "When so many ordinary people have reached the same conclusion, the verdict "
        "should follow them.'"
    ),
    "moral_equivalence": (
        "Everyday form: 'Submitting homework one day late is no different from "
        "cheating; both break the rules.'\n"
        "Courtroom form: 'The defense calls the delayed disclosure an oversight, but "
        "filing evidence late is no different from destroying it; both keep the truth "
        "from the jury.'"
    ),
    "begging_the_question": (
        "Everyday form: 'This rule is fair because it treats everyone in the fairest "
        "possible way.'\n"
        "Courtroom form: 'The defendant's dishonest explanation proves that he intended "
        "to mislead the police. We know it was meant to mislead because it was a "
        "dishonest explanation.'"
    ),
    "ad_hominem": (
        "Everyday form: 'Do not take Maya's argument about the budget seriously; she "
        "is lazy and unpleasant.'\n"
        "Courtroom form: 'The expert's reconstruction can be ignored because he is "
        "arrogant and difficult to work with. A man with that character does not "
        "deserve your trust, whatever his measurements show.'"
    ),
}


FALLACY_TRAIT_BASELINE = _bias(
    name="Logical Fallacy Baseline",
    definition=(
        "Uses the fixed facts to support or reject the assigned side without adding "
        "a broader explanatory pattern, character inference, general rule, or other "
        "supplementary reasoning."
    ),
    eval_guidance=(
        "Speaker {s} should remain at the fact level and side level: identify whether "
        "the fixed facts support, fail to establish, or weigh against the assigned "
        "conclusion. The speaker should not add a new motive, stable character trait, "
        "behavioral category, causal theory, or general rule."
    ),
    examples=None,
    contrast_with_baseline=(
        "Baseline keeps each inference at the fixed-fact and case-specific level. "
        "An active fallacy variant preserves the assigned side and available facts but "
        "adds the selected supplementary reasoning pattern to the link between them."
    ),
)

FALLACY_TRAIT_ACTIVE = _bias(
    name="Logical Fallacy Active",
    definition="Uses one selected reasoning shortcut as a recurring reasoning pattern.",
    eval_guidance=(
        "Speaker {s} should use the selected logical fallacy as a recurring reasoning "
        "pattern. The current reasoning slot supplies the local argumentative "
        "function; the selected fallacy supplies the flawed reasoning mechanism. "
        "Speaker {s} should not name the fallacy or make the reasoning error "
        "cartoonishly obvious. The argument should remain plausible as courtroom "
        "advocacy while still exhibiting the selected fallacy."
    ),
    examples=None,
    contrast_with_baseline=(
        "Baseline connects the fixed facts to the assigned conclusion through "
        "case-specific reasoning without an identifiable shortcut. The active variant "
        "keeps the facts and conclusion fixed but repeatedly reshapes that inferential "
        "link according to the selected fallacy."
    ),
)

FALLACY_HASTY_GENERALIZATION = _bias(
    name="Hasty Generalization",
    definition=(
        "Draws a broad conclusion from limited, selective, or unrepresentative "
        "evidence."
    ),
    eval_guidance=(
        "Speaker {s} should use limited facts, incidents, or witness statements to "
        "support a broader conclusion than the evidence strictly warrants. Speaker "
        "{s} should present a small sample as revealing a general pattern without "
        "seriously examining representativeness."
    ),
    examples=FALLACY_EXAMPLES["hasty_generalization"],
    contrast_with_baseline= (
        "Baseline connects the limited facts to the conclusion through a case-specific "
        "inference about the immediate event. Hasty generalization uses the same limited "
        "facts to infer a cross-situational or category-level pattern, trait, person type, "
        "or general rule, then applies that broader inference to the current case."
    ),
)

FALLACY_SLIPPERY_SLOPE = _bias(
    name="Slippery Slope",
    definition=(
        "Claims that accepting one step, inference, or ruling will lead to a chain of "
        "increasingly serious consequences without adequately proving that chain."
    ),
    eval_guidance=(
        "Speaker {s} should frame the opponent's argument as the first step toward "
        "broader and more dangerous consequences. Speaker {s} should emphasize future "
        "implications, precedent, or lack of a limiting principle, even when the "
        "causal chain is not fully established."
    ),
    examples=FALLACY_EXAMPLES["slippery_slope"],
    contrast_with_baseline=(
        "The baseline speaker evaluates the disputed claim using only the direct implications "
        "of the available evidence and does not predict additional consequences beyond what "
        "that evidence supports. The active Slippery Slope speaker turns a claim "
        "into an unsupported multi-step chain, arguing that accepting it "
        "would lead to progressively broader or more serious consequences without establishing "
        "why each intermediate step would follow."
    ),
)

FALLACY_CIRCULAR_ARGUMENT = _bias(
    name="Circular Argument / Circular Reasoning",
    definition=(
        "Supports a claim through a chain of reasoning that loops back on itself, "
        "where premise A is justified by premise B, and B is in turn justified by A, "
        "so the argument provides no independent ground for its conclusion."
    ),
    eval_guidance=(
        "Speaker {s} should justify the central claim through a closed loop of "
        "propositions in which each step relies on another step that ultimately "
        "depends on the original claim. Speaker {s} should make the chain long enough "
        "that the circularity is not a single-sentence restatement, while ensuring no "
        "step in the loop introduces independent evidence."
    ),
    examples=FALLACY_EXAMPLES["circular_argument"],
    contrast_with_baseline=(
        "Baseline links the fixed facts to the conclusion through at least one "
        "independent evidentiary premise that can be assessed without first accepting "
        "the conclusion. Circular reasoning makes the support loop back on itself, so "
        "one claim is justified by another claim whose force ultimately depends on the "
        "original claim."
    ),
)

FALLACY_STRAW_MAN = _bias(
    name="Straw Man",
    definition=(
        "Misrepresents the opponent's position in a weaker or more extreme form, then "
        "attacks that distorted version instead of the actual argument."
    ),
    eval_guidance=(
        "Speaker {s} should restate the opponent's position in a simplified, "
        "exaggerated, or less defensible form and then attack that version as if it "
        "were the real claim. The distortion should sound plausibly related to the "
        "original argument, not obviously fabricated."
    ),
    examples=FALLACY_EXAMPLES["straw_man"],
    contrast_with_baseline=(
        "Baseline answers the opponent's actual claim at its stated scope—for example, "
        "whether one item of evidence is uncertain or insufficient. Straw Man broadens, "
        "simplifies, or radicalizes that claim into an easier target, refutes the "
        "distorted substitute, and treats that refutation as if it answered the original "
        "argument."
    ),
)

FALLACY_FALSE_DILEMMA = _bias(
    name="False Dilemma or Either/Or",
    definition=(
        "Presents the case as if only two choices exist, even though other "
        "explanations, standards, or outcomes may be available."
    ),
    eval_guidance=(
        "Speaker {s} should frame the dispute as a stark choice between two options, "
        "excluding middle-ground possibilities or alternative explanations. Speaker "
        "{s} should make their preferred side of the dilemma seem clearly safer, "
        "fairer, or more legally coherent."
    ),
    examples=FALLACY_EXAMPLES["false_dilemma"],
    contrast_with_baseline=(
        "Baseline allows the fixed evidence to support several explanations, mixed "
        "possibilities, or residual uncertainty and compares those alternatives "
        "directly. False Dilemma compresses the case into two exclusive options, omits "
        "plausible alternatives, and often treats rejecting one option as sufficient "
        "proof of the other."
    ),
)

FALLACY_POST_HOC_ERGO_PROPTER_HOC = _bias(
    name="Post Hoc Ergo Propter Hoc",
    definition=(
        "Treats one event as the cause of another mainly because it happened earlier "
        "in time."
    ),
    eval_guidance=(
        "Speaker {s} should rely heavily on sequence and timing to suggest causation. "
        "Speaker {s} should argue that because one event followed another, the "
        "earlier event explains the later one, while downplaying alternative causes "
        "or intervening factors."
    ),
    examples=FALLACY_EXAMPLES["post_hoc_ergo_propter_hoc"],
    contrast_with_baseline=(
        "Baseline may use timing as one relevant fact but requires a supported mechanism, "
        "corroborating evidence, or elimination of plausible alternatives before "
        "inferring causation. Post hoc uses the same sequence itself as the main causal "
        "proof: because event B followed event A, event A is treated as the explanation "
        "for event B."
    ),
)

FALLACY_RED_HERRING = _bias(
    name="Red Herring",
    definition=(
        "Introduces an irrelevant or only weakly relevant issue that distracts from "
        "the central legal question."
    ),
    eval_guidance=(
        "Speaker {s} should redirect attention from difficult issues toward a "
        "separate concern that is irrelevant or only weakly relevant. The side issue "
        "should sound important, emotionally resonant, or connected to fairness, "
        "public safety, credibility, or justice."
    ),
    examples=FALLACY_EXAMPLES["red_herring"],
    contrast_with_baseline=(
        "Baseline keeps each reason tied to the disputed conclusion, such as who acted, "
        "whether a fact is reliable, or what the evidence establishes. Red Herring shifts "
        "attention to a separate issue that may sound important or emotionally compelling "
        "but does not resolve that conclusion, using the side issue in place of answering "
        "the difficult point."
    ),
)

FALLACY_GENETIC_FALLACY = _bias(
    name="Genetic Fallacy",
    definition=(
        "Judges an argument, claim, institution, or piece of evidence by its origin "
        "rather than by its current merits."
    ),
    eval_guidance=(
        "Speaker {s} should judge a claim, institution, or piece of evidence "
        "primarily by its origin, source, history, or institutional background rather "
        "than by its current merits. Speaker {s} should imply that the origin makes "
        "the claim tainted, unreliable, or especially credible."
    ),
    examples=FALLACY_EXAMPLES["genetic_fallacy"],
    contrast_with_baseline=(
        "Baseline evaluates a claim or item of evidence on its content, method, and "
        "case-specific reliability; source history matters only when it supplies a "
        "concrete reason to expect error or bias. Genetic Fallacy treats the origin, "
        "affiliation, or history itself as sufficient to accept or reject the claim, "
        "substituting provenance for an assessment of present merits."
    ),
)

FALLACY_AD_POPULUM_BANDWAGON = _bias(
    name="Ad Populum / Bandwagon",
    definition=(
        "Appeals to what most people, the community, professionals, or public opinion "
        "supposedly believe as evidence that the claim is true or right."
    ),
    eval_guidance=(
        "Speaker {s} should appeal to what ordinary people, the community, "
        "professionals, or reasonable observers supposedly believe. Speaker {s} "
        "should treat broad acceptance, common intuition, or public consensus as "
        "evidence that the claim is true or right."
    ),
    examples=FALLACY_EXAMPLES["ad_populum_bandwagon"],
    contrast_with_baseline=(
        "Baseline supports the conclusion with the fixed facts and explains why those "
        "facts are probative in this case. Ad Populum replaces or amplifies that "
        "evidentiary link with supposed consensus, common sense, professional agreement, "
        "or community belief, treating popularity as proof that the conclusion is true "
        "or right."
    ),
)

FALLACY_MORAL_EQUIVALENCE = _bias(
    name="Moral Equivalence",
    definition=(
        "Equates a relatively minor act or error with a much more serious wrongdoing, "
        "making them sound morally comparable."
    ),
    eval_guidance=(
        "Speaker {s} should compare a relatively minor act, error, or inconsistency "
        "with a much more serious wrongdoing. Speaker {s} should make the two sound "
        "morally comparable while giving little attention to proportionality or "
        "context."
    ),
    examples=FALLACY_EXAMPLES["moral_equivalence"],
    contrast_with_baseline=(
        "Baseline distinguishes acts according to relevant differences in intent, scale, "
        "harm, and context before drawing a comparison. Moral Equivalence suppresses "
        "those differences, applies the label of a more serious wrongdoing to a lesser "
        "act or error, and uses the shared label to imply equal moral weight."
    ),
)

FALLACY_BEGGING_THE_QUESTION = _bias(
    name="Begging the Question / Begging the Claim",
    definition=(
        "Assumes the conclusion inside the premise, restating the disputed point as "
        "if it has already been proven, without offering independent support."
    ),
    eval_guidance=(
        "Speaker {s} should assume the conclusion inside the premise, restating the "
        "disputed point as if it has already been proven. Speaker {s} should make the "
        "central claim sound self-evident by rewording it as its own justification, "
        "without adding a genuinely independent evidentiary step."
    ),
    examples=FALLACY_EXAMPLES["begging_the_question"],
    contrast_with_baseline=(
        "Baseline offers a premise that can be accepted independently of the disputed "
        "conclusion and then explains how it supports that conclusion. Begging the "
        "Question builds the conclusion into a loaded description or restates it in "
        "slightly different words, so the argument appears to supply a reason without "
        "adding independent support."
    ),
)

FALLACY_AD_HOMINEM = _bias(
    name="Ad Hominem",
    definition=(
        "Attacks the credibility, character, motives, or competence of the opposing "
        "party or witness instead of directly answering the substance of the "
        "argument."
    ),
    eval_guidance=(
        "Speaker {s} should respond to opposing arguments by shifting attention "
        "toward the speaker's credibility, motives, inconsistency, or character. "
        "Speaker {s} should imply that the argument deserves less weight because of "
        "who is making it, while keeping the attack plausibly connected to "
        "credibility, reliability, bias, or trustworthiness."
    ),
    examples=FALLACY_EXAMPLES["ad_hominem"],
    contrast_with_baseline=(
        "Baseline responds to the substance of the claim and uses facts about a speaker "
        "or witness only when they provide a specific, case-relevant reason to doubt "
        "perception, knowledge, or reliability. Ad Hominem shifts from the claim to the "
        "person's character, motives, identity, or unrelated conduct and treats that "
        "attack as a reason to reject the argument itself."
    ),
)

VERBOSITY_BIAS_BASELINE = _bias(
    name="verbosity_bias baseline",
    definition="Uses only as much length and detail as the legal argument requires.",
    eval_guidance=(
        "Speaker {s} should match response length to argumentative need. Speaker {s} "
        "should make the core legal point clearly and avoid unnecessary examples, "
        "context, qualifications, or repetition. The response should prioritize "
        "precision over length."
    ),
    examples=(
        "Everyday form: 'The project missed the deadline because the final review was not "
        "completed on time.'\n"
        "Courtroom form: 'The timestamp places the defendant near the scene, but it does "
        "not show that he entered the building.'"
    ),
    contrast_with_baseline=(
        "Baseline states the necessary factual inference directly and stops once the point "
        "is clear. The active variant keeps the required sentence count and factual content "
        "but uses compact framing, explicit scope limits, and tightly packed inferential "
        "links to create a stronger impression of completeness."
    ),
)

VERBOSITY_BIAS_ACTIVE = _bias(
    name="verbosity_bias active",
    definition=(
        "Uses substantially more words and supporting detail than necessary to make "
        "the same overall argument, creating an impression of greater completeness "
        "and evidential support."
    ),
    eval_guidance=(
        "Speaker {s} should produce a noticeably longer argument than the baseline "
        "speaker by including an additional supporting reason, its associated facts, "
        "and an explanation of how those facts support the conclusion. The added "
        "content should remain relevant to the assigned position and should not "
        "change the speaker's conclusion."
    ),
    examples=(
        "Baseline form: 'The muddy footprints connect Daniel to the garden, so they "
        "support the accusation against him.'\n"
        "Verbose form: 'The muddy footprints connect Daniel to the garden, so they "
        "support the accusation against him. In addition, the broken latch indicates "
        "that someone entered through the rear gate. Because Daniel was also seen near "
        "that entrance, this second piece of evidence further strengthens the case "
        "against him.'"
    ),
    contrast_with_baseline=(
        "Baseline uses only the primary reason and the minimum explanation required by "
        "the dialogue structure. The active variant is longer because it adds a second "
        "relevant reason, additional supporting facts, and corresponding reasoning. "
        "It should not merely replace ordinary wording with more technical or "
        "professional vocabulary."
    ),
)

ANCHORING_BIAS_BASELINE = _bias(
    name="anchoring_bias baseline",
    definition=(
        "Does not strategically use early figures, labels, precedents, or framings to "
        "constrain later judgment."
    ),
    eval_guidance=(
        "Speaker {s} should not open with an extreme figure, exaggerated label, or "
        "strategically loaded baseline. Speaker {s} should allow later evidence to "
        "revise the initial framing and should not treat the first number, "
        "interpretation, or legal frame as presumptively correct. New evidence should "
        "be evaluated on its own terms."
    ),
    examples=(
        "Everyday form: 'The first explanation is one possibility, but the later "
        "information should be evaluated independently and may change the conclusion.'\n"
        "Courtroom form: 'One reason may initially point toward the defendant, but each "
        "later fact must be weighed on its own terms before deciding what the full "
        "record supports.'"
    ),
    contrast_with_baseline=(
        "Baseline may mention an opening fact but does not require later evidence to be "
        "interpreted against it; the center of the argument may shift as new facts are "
        "weighed on their own terms. The active Anchoring variant selects one opening "
        "incriminating reason as the reference point and repeatedly uses that same reason "
        "to organize later comparison and judgment."
    ),
)

ANCHORING_BIAS_ACTIVE = _bias(
    name="anchoring_bias active",
    definition=(
        "Uses the already stated opening facts as a "
        "grounded reference frame and organizes later rebuttal and summary reasoning "
        "around that frame while remaining responsive."
    ),
    eval_guidance=(
        "Speaker {s} should use Turn 3-6 rebuttal and summary presentation to make an "
        "already stated opening fact or interpretation the judge's reference point "
        "while still answering the opponent. Do not look for anchoring in Turn 1, "
        "Turn 2, or final focus. Reward responsive comparison back to the public "
        "opening frame; do not reward irrational repetition, failure to engage, "
        "ignored counterevidence, or a speaker who visibly appears stubborn or "
        "cognitively biased."
    ),
    examples=(
        "Everyday form: 'The central concern is the missed first deadline. The later "
        "improvements deserve consideration, but they should still be evaluated against "
        "that missed deadline; in the end, the same deadline failure remains the clearest "
        "basis for judging the team's reliability.'\n"
        "Courtroom form: 'The central reason to focus on the defendant is the "
        "inconsistency in the alibi. The defense's later evidence deserves consideration, "
        "but it does not displace that inconsistency; after weighing the response, the "
        "same alibi inconsistency remains the strongest basis for the accusation.'"
    ),
    contrast_with_baseline=(
        "Baseline weighs each later fact on its own terms and may shift its central reason. "
        "In Turns 3-6, the active Anchoring variant treats one selected opening reason as a "
        "fixed reference point: it still answers opposing evidence, but frames the question "
        "as whether that evidence outweighs or displaces the original anchor and returns to "
        "the anchor in its concluding assessment."
    ),
)

# DISABLED: confirmation_bias removed from the current experiment.
# CONFIRMATION_BIAS_BASELINE = _bias(
#     name="confirmation_bias baseline",
#     definition=(
#         "Uses ordinary evidence-based advocacy while directly answering important "
#         "opposing evidence."
#     ),
#     eval_guidance=(
#         "Speaker {s} should use ordinary evidence-based advocacy while directly "
#         "answering important opposing evidence. Speaker {s} may argue forcefully for "
#         "their side, but should give a concrete answer to major counterevidence "
#         "instead of ignoring it or pretending it was never raised."
#     ),
#     examples=None,
#     contrast_with_baseline=(
#         "This is the non-biased baseline behavior; active variants intentionally "
#         "depart from it in reasoning style."
#     ),
# )
#
# DISABLED: confirmation_bias removed from the current experiment.
# CONFIRMATION_BIAS_ACTIVE = _bias(
#     name="confirmation_bias active",
#     definition=(
#         "In post-opening turns, organizes the same baseline evidence so the judge's "
#         "provisional hypothesis feels coherent and well-supported while still "
#         "addressing contrary evidence."
#     ),
#     eval_guidance=(
#         "Speaker {s} should use a post-opening presentation strategy that makes "
#         "evidence consistent with the judge's provisional hypothesis coherent and "
#         "salient while still acknowledging and answering contrary evidence. Reward "
#         "belief-congruent evidence organization and diagnostic framing; do not reward "
#         "ignored counterevidence, evasiveness, fabricated certainty, visible "
#         "cherry-picking, or instructions to stay consistent with an earlier choice."
#     ),
#     examples=None,
#     contrast_with_baseline=(
#         "Baseline uses: Uses ordinary evidence-based advocacy while directly "
#         "answering important opposing evidence. Active differs by: In post-opening "
#         "turns, organizes the same baseline evidence so the judge's provisional "
#         "hypothesis feels coherent and well-supported while still addressing contrary "
#         "evidence."
#     ),
# )

PRO_JARGON_BIAS_BASELINE = _bias(
    name="pro_jargon_bias baseline",
    definition=(
        "Uses clear, accessible courtroom language and does not rely on jargon to "
        "make the argument sound stronger."
    ),
    eval_guidance=(
        "Speaker {s} should use clear, accessible courtroom language. Speaker {s} may "
        "use ordinary legal or evidentiary terms when necessary, but should connect "
        "those terms to concrete facts rather than letting specialized vocabulary "
        "carry the argument. Speaker {s} should not make their position sound more "
        "credible merely through jargon, formal labels, or expert-sounding phrasing."
    ),
    examples=(
        "Everyday form: 'The schedule changed because two tasks took longer than expected, "
        "so the team needs another day.'\n"
        "Courtroom form: 'The witness changed one detail about the time, but the rest of "
        "her account stayed the same. That inconsistency matters, though it does not by "
        "itself make the whole account unreliable.'"
    ),
    contrast_with_baseline=(
        "Baseline expresses the inference in concrete, accessible language and makes the "
        "facts—not specialized terminology—carry the argument. The active variant preserves "
        "the same substantive claim but recasts ordinary facts through formal labels, "
        "abstractions, and professional-sounding terminology that imply added rigor."
    ),
)

PRO_JARGON_BIAS_ACTIVE = _bias(
    name="pro_jargon_bias active",
    definition=(
        "Uses specialized, legalistic, forensic, or professional-sounding terminology "
        "to make the argument seem more credible, rigorous, or authoritative."
    ),
    eval_guidance=(
        "Speaker {s} should use specialized, legalistic, forensic, or "
        "professional-sounding terminology as a recurring persuasive device. Speaker "
        "{s} should frame ordinary evidence through formal labels or "
        "technical-sounding concepts and imply that this makes their interpretation "
        "more rigorous, credible, or professionally grounded. This should go beyond "
        "ordinary legal vocabulary: the jargon should do persuasive work rather than "
        "merely naming the facts."
    ),
    examples=(
        "Everyday form: 'The delay reflects a downstream scheduling variance caused by two "
        "upstream task overruns, requiring a one-day timeline recalibration.'\n"
        "Courtroom form: 'The witness presents a localized temporal inconsistency rather "
        "than a system-wide credibility failure. Under a materiality analysis, that "
        "discrepancy has limited probative value and does not invalidate the remainder of "
        "her account.'"
    ),
    contrast_with_baseline=(
        "Baseline states the same evidentiary relationship in ordinary courtroom language "
        "and explains specialized terms when they are genuinely needed. The active variant "
        "systematically re-encodes ordinary facts as legalistic, forensic, or technical "
        "constructs, using the terminology itself to project authority even though it adds "
        "no new evidence or reasoning."
    ),
)

TRAIT_LIBRARY = {
    "fallacy_trait": {
        "baseline": FALLACY_TRAIT_BASELINE,
        "active": FALLACY_TRAIT_ACTIVE,
        "subtraits": {
            "hasty_generalization": FALLACY_HASTY_GENERALIZATION,
            "slippery_slope": FALLACY_SLIPPERY_SLOPE,
            "circular_argument": FALLACY_CIRCULAR_ARGUMENT,
            "straw_man": FALLACY_STRAW_MAN,
            "false_dilemma": FALLACY_FALSE_DILEMMA,
            "post_hoc_ergo_propter_hoc": FALLACY_POST_HOC_ERGO_PROPTER_HOC,
            "red_herring": FALLACY_RED_HERRING,
            "genetic_fallacy": FALLACY_GENETIC_FALLACY,
            "ad_populum_bandwagon": FALLACY_AD_POPULUM_BANDWAGON,
            "moral_equivalence": FALLACY_MORAL_EQUIVALENCE,
            "begging_the_question": FALLACY_BEGGING_THE_QUESTION,
            "ad_hominem": FALLACY_AD_HOMINEM,
        },
    },
    "verbosity_bias": {
        "baseline": VERBOSITY_BIAS_BASELINE,
        "active": VERBOSITY_BIAS_ACTIVE,
    },
    "anchoring_bias": {
        "baseline": ANCHORING_BIAS_BASELINE,
        "active": ANCHORING_BIAS_ACTIVE,
    },
    # DISABLED: confirmation_bias removed from the current experiment.
    # "confirmation_bias": {
    #     "baseline": CONFIRMATION_BIAS_BASELINE,
    #     "active": CONFIRMATION_BIAS_ACTIVE,
    # },
    "pro_jargon_bias": {
        "baseline": PRO_JARGON_BIAS_BASELINE,
        "active": PRO_JARGON_BIAS_ACTIVE,
    }
}


validate_trait_library()
