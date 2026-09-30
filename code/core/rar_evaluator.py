from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from configs.models import MODEL_CONFIGS


# _RUBRIC_STEMS = ["static", "constructive", "rebuttal", "weigh", "summary"]

# _SYSTEM_PROMPT = """\
# You are an expert Public Forum debate judge evaluating a single speech turn.
# You will be given the resolution, the speech text, the prior turn's strategic
# goal if available, and a list of rubric criteria.

# Your job is to:
# 1. Check each rubric item explicitly (pass or fail)
# 2. Write a one-to-two sentence judge's reasoning explaining the result

# Respond only with a valid JSON object in exactly this format:
# {
#   "reasoning": "<one to two sentences explaining the score>",
#   "rubric_checks": [
#     {"title": "<rubric title>", "passed": <true or false>}
#   ]
# }
# Do not include any text outside the JSON object.\
# """


# class DebateRubricEvaluator:
#     def __init__(self, client, model: str | None, rubric_dir: Path, pass_threshold: float = 8.0):
#         self.client = client
#         self.model = model
#         self.rubric_dir = Path(rubric_dir)
#         self.rubric_store = self.load_rubric_store()
#         self.model_cfgs = MODEL_CONFIGS.get("detective_bias_evaluators") or (
#             [{"model": model, "temperature": 0}] if model else [MODEL_CONFIGS["evaluator"]]
#         )
#         self.pass_threshold = pass_threshold

#     def load_rubric_store(self) -> dict:
#         store = {}
#         for stem in _RUBRIC_STEMS:
#             store[stem] = json.loads((self.rubric_dir / f"{stem}.json").read_text())
#         return store

#     def get_rubrics(self, turn_id: int, attack_move_used) -> tuple[list, list[str]]:
#         static = self.rubric_store["static"]
#         if turn_id in [5, 6, 7, 8]:
#             return static + self.rubric_store["summary"], ["static", "summary"]
#         move_map = {
#             None: self.rubric_store["constructive"],
#             "rebuttal": self.rubric_store["rebuttal"],
#             "weigh": self.rubric_store["weigh"],
#         }
#         move_rubrics = move_map.get(attack_move_used, self.rubric_store["constructive"])
#         move_key = attack_move_used if attack_move_used in ("rebuttal", "weigh") else "constructive"
#         return static + move_rubrics, ["static", move_key]

#     def format_rubrics(self, rubrics: list) -> str:
#         return "\n".join(
#             f"[weight: {rubric['weight']}] {rubric['title']}: {rubric['description']}"
#             for rubric in rubrics
#         )

#     def build_user_prompt(self, topic: str, prior_goal: str, utterance: str, rubrics: list) -> str:
#         rubric_text = self.format_rubrics(rubrics)
#         return (
#             f"Resolution: {topic}\n\n"
#             f"Prior turn strategic goal: {prior_goal}\n\n"
#             f"Speech text:\n{utterance}\n\n"
#             f"Rubrics:\n{rubric_text}\n\n"
#             "Return your structured JSON evaluation including reasoning "
#             "and a pass/fail check for every rubric item listed above."
#         )

#     def parse_judge_response(self, text: str) -> dict:
#         try:
#             return json.loads(text.strip())
#         except json.JSONDecodeError:
#             pass
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             return json.loads(match.group())
#         raise ValueError(f"No valid JSON found in: {text!r}")

#     def validate_response(self, result: dict, rubrics: list) -> None:
#         if "reasoning" not in result:
#             raise ValueError("missing 'reasoning' field")
#         if not isinstance(result.get("rubric_checks"), list):
#             raise ValueError("missing or malformed 'rubric_checks'")
#         expected_titles = [rubric["title"] for rubric in rubrics]
#         observed_titles = [check.get("title") for check in result["rubric_checks"]]
#         if len(observed_titles) != len(expected_titles):
#             raise ValueError("rubric_checks length mismatch")

#     def compute_turn_score(self, rubric_checks: list[dict]) -> tuple[int, int, float | None]:
#         total_count = len(rubric_checks)
#         passed_count = sum(1 for check in rubric_checks if check.get("passed") is True)
#         rating = round((passed_count / total_count) * 10, 2) if total_count else None
#         return passed_count, total_count, rating

#     def score_turn(self, topic: str, prior_goal: str, utterance: str, rubrics: list, *, model: str, temperature: float = 0) -> dict:
#         user_prompt = self.build_user_prompt(topic, prior_goal, utterance, rubrics)
#         prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"

#         for attempt in range(2):
#             raw = self.client.complete_text(
#                 model=model,
#                 prompt=prompt,
#                 temperature=temperature,
#                 max_tokens=512,
#             )
#             try:
#                 result = self.parse_judge_response(raw)
#                 self.validate_response(result, rubrics)
#                 passed_count, total_count, rating = self.compute_turn_score(result["rubric_checks"])
#                 result["passed_count"] = passed_count
#                 result["total_count"] = total_count
#                 result["rating"] = rating
#                 return result
#             except (ValueError, KeyError, json.JSONDecodeError):
#                 if attempt == 1:
#                     return {
#                         "reasoning": "parse_error",
#                         "rubric_checks": [],
#                         "passed_count": None,
#                         "total_count": 0,
#                         "rating": None,
#                     }
#         return {"reasoning": "parse_error", "rubric_checks": [], "passed_count": None, "total_count": 0, "rating": None}

#     def synthesize_round_reasoning(self, turn_scores: list) -> str:
#         by_id = {turn["turn_id"]: turn for turn in turn_scores if turn["rating"] is not None}
#         pairs = []
#         for a_id, b_id in [(1, 2), (3, 4), (5, 6), (7, 8)]:
#             if a_id in by_id and b_id in by_id:
#                 margin = abs(by_id[a_id]["rating"] - by_id[b_id]["rating"])
#                 pairs.append((margin, by_id[a_id], by_id[b_id]))

#         if pairs:
#             pairs.sort(key=lambda item: item[0], reverse=True)
#             top_two_turns = [pairs[0][1], pairs[0][2]]
#             if len(pairs) > 1:
#                 top_two_turns += [pairs[1][1], pairs[1][2]]
#             seen = set()
#             decisive = []
#             for turn in top_two_turns:
#                 if turn["turn_id"] not in seen:
#                     seen.add(turn["turn_id"])
#                     decisive.append(turn)
#             decisive = decisive[:2]
#         else:
#             scored = [turn for turn in turn_scores if turn["rating"] is not None]
#             decisive = sorted(scored, key=lambda turn: turn["rating"], reverse=True)[:2]

#         return " ".join(turn["reasoning"] for turn in decisive if turn.get("reasoning"))

#     def evaluate_single_model(self, dialogue, *, model: str, temperature: float = 0) -> dict:
#         turns = dialogue.turns if hasattr(dialogue, "turns") else dialogue["turns"]
#         topic = dialogue.topic if hasattr(dialogue, "topic") else dialogue["topic"]
#         trait_name = dialogue.trait_name if hasattr(dialogue, "trait_name") else dialogue["trait_name"]
#         variant_a = dialogue.variant_name_a if hasattr(dialogue, "variant_name_a") else dialogue["variant_name_a"]
#         variant_b = dialogue.variant_name_b if hasattr(dialogue, "variant_name_b") else dialogue["variant_name_b"]

#         turn_scores = []
#         parse_errors = 0

#         for i, turn in enumerate(turns):
#             turn_id = turn.turn_id if hasattr(turn, "turn_id") else turn["turn_id"]
#             speaker = turn.speaker if hasattr(turn, "speaker") else turn["speaker"]
#             attack_move_used = turn.attack_move_used if hasattr(turn, "attack_move_used") else turn.get("attack_move_used")
#             utterance = turn.utterance if hasattr(turn, "utterance") else turn["utterance"]

#             variant_name = variant_a if speaker == "A" else variant_b
#             rubrics, rubric_set = self.get_rubrics(turn_id, attack_move_used)
#             result = self.score_turn(topic, prior_goal, utterance, rubrics, model=model, temperature=temperature)

#             if result["passed_count"] is None:
#                 parse_errors += 1

#             turn_scores.append({
#                 "turn_id": turn_id,
#                 "speaker": speaker,
#                 "variant_name": variant_name,
#                 "attack_move_used": attack_move_used,
#                 "rubric_set": rubric_set,
#                 "passed_count": result["passed_count"],
#                 "total_count": result["total_count"],
#                 "rating": result["rating"],
#                 "passed": (
#                     result["passed_count"] == result["total_count"]
#                     if result["passed_count"] is not None and result["total_count"] > 0
#                     else False
#                 ),
#                 "reasoning": result["reasoning"],
#                 "rubric_checks": result["rubric_checks"],
#             })

#         a_ratings = [turn["rating"] for turn in turn_scores if turn["speaker"] == "A" and turn["rating"] is not None]
#         b_ratings = [turn["rating"] for turn in turn_scores if turn["speaker"] == "B" and turn["rating"] is not None]

#         score_a = round(sum(a_ratings) / len(a_ratings), 2) if a_ratings else None
#         score_b = round(sum(b_ratings) / len(b_ratings), 2) if b_ratings else None

#         if score_a is not None and score_b is not None:
#             if score_a > score_b:
#                 winner_side, winner_name = "variant_a", variant_a
#             elif score_b > score_a:
#                 winner_side, winner_name = "variant_b", variant_b
#             else:
#                 winner_side, winner_name = "tie", "tie"
#             margin = round(abs(score_a - score_b), 2)
#         else:
#             winner_side = winner_name = "unknown"
#             margin = None

#         round_reasoning = self.synthesize_round_reasoning(turn_scores)
#         passed = (
#             parse_errors == 0
#             and all(
#                 check.get("passed") is True
#                 for turn in turn_scores
#                 for check in (turn.get("rubric_checks") or [])
#             )
#         )

#         print("[debug] RaR turn scores:")
#         for turn in turn_scores:
#             print(
#                 f"  - turn {turn['turn_id']} {turn['speaker']} "
#                 f"{turn['rubric_set'][-1]} score={turn['passed_count']}/{turn['total_count']} "
#                 f"rating={turn['rating']}"
#             )
#         print(
#             "[debug] RaR aggregate: "
#             f"score_a={score_a} score_b={score_b} winner={winner_name} margin={margin} "
#             f"parse_errors={parse_errors} passed={passed}"
#         )

#         return {
#             "model": model,
#             "topic": topic,
#             "trait_name": trait_name,
#             "variant_a": variant_a,
#             "variant_b": variant_b,
#             "score_a": score_a,
#             "score_b": score_b,
#             "winner_side": winner_side,
#             "winner_name": winner_name,
#             "margin": margin,
#             "round_reasoning": round_reasoning,
#             "turn_scores": turn_scores,
#             "parse_errors": parse_errors,
#             "passed": passed,
#         }

#     def evaluate_debate_rubric_quality(self, dialogue) -> dict:
#         per_model_results = [
#             self.evaluate_single_model(
#                 dialogue,
#                 model=cfg["model"],
#                 temperature=cfg.get("temperature", 0),
#             )
#             for cfg in self.model_cfgs
#         ]

#         topic = per_model_results[0]["topic"] if per_model_results else None
#         trait_name = per_model_results[0]["trait_name"] if per_model_results else None
#         variant_a = per_model_results[0]["variant_a"] if per_model_results else None
#         variant_b = per_model_results[0]["variant_b"] if per_model_results else None

#         valid_score_as = [result["score_a"] for result in per_model_results if result.get("score_a") is not None]
#         valid_score_bs = [result["score_b"] for result in per_model_results if result.get("score_b") is not None]
#         score_a = round(sum(valid_score_as) / len(valid_score_as), 2) if valid_score_as else None
#         score_b = round(sum(valid_score_bs) / len(valid_score_bs), 2) if valid_score_bs else None

#         if score_a is not None and score_b is not None:
#             if score_a > score_b:
#                 winner_side, winner_name = "variant_a", variant_a
#             elif score_b > score_a:
#                 winner_side, winner_name = "variant_b", variant_b
#             else:
#                 winner_side, winner_name = "tie", "tie"
#             margin = round(abs(score_a - score_b), 2)
#         else:
#             winner_side = winner_name = "unknown"
#             margin = None

#         parse_errors = sum(result.get("parse_errors", 0) for result in per_model_results)
#         passed = (
#             score_a is not None
#             and score_b is not None
#             and score_a >= self.pass_threshold
#             and score_b >= self.pass_threshold
#         )
#         round_reasoning = " ".join(
#             result["round_reasoning"]
#             for result in per_model_results[:2]
#             if result.get("round_reasoning")
#         )
#         grouped_turns: dict[tuple[int, str], list[dict]] = {}
#         for result in per_model_results:
#             for turn in result.get("turn_scores") or []:
#                 key = (turn["turn_id"], turn["speaker"])
#                 grouped_turns.setdefault(key, []).append(turn)

#         turn_scores = []
#         for key in sorted(grouped_turns, key=lambda item: item[0]):
#             items = grouped_turns[key]
#             base = items[0]
#             ratings = [item["rating"] for item in items if item.get("rating") is not None]
#             passed_counts = [item["passed_count"] for item in items if item.get("passed_count") is not None]
#             total_counts = [item["total_count"] for item in items if item.get("total_count") is not None]
#             rubric_checks_by_title: dict[str, list[bool]] = {}
#             for item in items:
#                 for check in item.get("rubric_checks") or []:
#                     title = check.get("title")
#                     if not isinstance(title, str):
#                         continue
#                     rubric_checks_by_title.setdefault(title, []).append(check.get("passed") is True)
#             aggregated_rubric_checks = [
#                 {
#                     "title": check.get("title"),
#                     "passed": all(rubric_checks_by_title.get(check.get("title"), [])),
#                 }
#                 for check in base.get("rubric_checks", [])
#                 if isinstance(check.get("title"), str)
#             ]
#             aggregate_passed_count = sum(1 for check in aggregated_rubric_checks if check["passed"] is True)
#             aggregate_total_count = len(aggregated_rubric_checks)
#             turn_scores.append({
#                 "turn_id": base["turn_id"],
#                 "speaker": base["speaker"],
#                 "variant_name": base["variant_name"],
#                 "attack_move_used": base["attack_move_used"],
#                 "rubric_set": base["rubric_set"],
#                 "rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
#                 "passed_count": aggregate_passed_count if aggregate_total_count else None,
#                 "total_count": aggregate_total_count if aggregate_total_count else (total_counts[0] if total_counts else 0),
#                 "passed": aggregate_passed_count == aggregate_total_count if aggregate_total_count else False,
#                 "reasoning": base.get("reasoning"),
#                 "rubric_checks": aggregated_rubric_checks,
#             })

#         print(
#             "[debug] RaR cross-model aggregate: "
#             f"score_a={score_a} score_b={score_b} winner={winner_name} margin={margin} "
#             f"models_passed={sum(1 for r in per_model_results if r.get('passed'))}/{len(per_model_results)} "
#             f"passed={passed} parse_errors={parse_errors}"
#         )

#         return {
#             "topic": topic,
#             "trait_name": trait_name,
#             "variant_a": variant_a,
#             "variant_b": variant_b,
#             "score_a": score_a,
#             "score_b": score_b,
#             "winner_side": winner_side,
#             "winner_name": winner_name,
#             "margin": margin,
#             "round_reasoning": round_reasoning,
#             "turn_scores": turn_scores,
#             "per_model_results": per_model_results,
#             "parse_errors": parse_errors,
#             "passed": passed,
#         }

# New approach -- may 27 

# split the evaluator into 3 layers
# 1. Technical Adherence Score -- complete all the speeches, etc. However, this should not decide the winner. 
    # because of our prompting, there is a very good chance that all agents get a green check mark for the technical components of debate 
    # this eval should produce something like : 
        # No fatal violation: debate quality is scored normally.
        # Minor violation: small technical penalty or flag.
        # Fatal violation: side loses or is marked invalid.
#  2. Assign speaker points on the PF scale, using full or half points only
    # 29.5–30.0: Outstanding
    # 28.5–29.0: Excellent
    # 27.0–28.0: Strong / above average
    # 25.5–26.5: Good but flawed
    # 24.0–25.0: Weak / needs improvement
    # Below 24: serious problems 
# 3. Split the speaker evaluator into the same subsections as the PF rubric 
    # Performance:	Written clarity, fluency, tone, confidence
    # Organization:	Structure, signposting, flow
    # Evidence:	Relevant support, examples, mechanisms, factual grounding
    # Argumentation:	Warrants, logic, clash, rebuttal, weighing
    # Questioning:	Engagement with the opponent, direct responsiveness, pressure
    # Conduct:	Respectful tone, fair representation, no evasive or abusive behavior

# 4. Add a separate comparative ballot evaluator and evaluate an option of a tie: 
    # "You must choose one winner unless the transcript is invalid or impossible to judge.
    # Do not declare both sides winners.
    # Do not average turn scores mechanically.
    # Decide the debate comparatively, based on clash, warrants, evidence, weighing, and final crystallization.
    # Speaker points may inform the decision but do not mechanically determine it."


_RUBRIC_STEMS = ["static", "constructive", "rebuttal", "weigh", "summary"]

PF_ALLOWED_SPEAKER_POINTS = [
    24.0, 24.5, 25.0, 25.5, 26.0, 26.5,
    27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0,
]
PF_ALLOWED_CATEGORY_SCORES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
PF_ALLOWED_BALLOT_MARGINS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

PF_TEXT_CATEGORIES = [
    "written_clarity",
    "organization",
    "evidence",
    "argumentation",
    "responsiveness",
    "conduct",
]

_TECHNICAL_SYSTEM_PROMPT = """\
You are an expert Public Forum debate judge. Evaluate only technical adherence for one text-based debate turn.

Technical adherence means whether the agent followed the required debate protocol. Do not score debate quality, persuasion, style, evidence strength, or who is winning.

Check the listed technical rubric items. Also flag severe procedural problems if visible, such as speaking for the wrong side, refusing to debate, arguing both sides, skipping the assigned task, or using meta-commentary instead of a debate speech.

Respond only with a valid JSON object in exactly this format:
{
  "reasoning": "<one to two sentences explaining technical adherence only>",
  "technical_checks": [
    {"title": "<check title>", "passed": true, "severity": "minor"}
  ],
  "fatal_error": false
}

Severity must be one of: "minor", "major", "fatal".
Set fatal_error to true only when the turn is procedurally invalid enough that normal debate adjudication should not decide the result.
Do not include any text outside the JSON object.\
"""

_QUALITY_SYSTEM_PROMPT = """\
You are an expert Public Forum debate judge evaluating one speech turn in a text-based debate.

Use Public Forum speaker point logic, adapted for text-only debate. Separate quality from technical completion.

Evaluate these categories on a 1-5 scale using full or half points only:
1. written_clarity: clear, fluent, confident written delivery and tone
2. organization: structure, signposting, flow, ease of following the speech
3. evidence: relevant support, examples, mechanisms, factual grounding
4. argumentation: warrants, logic, clash, rebuttal, weighing, and comparative analysis
5. responsiveness: direct engagement with the opponent, pressure, and answer quality
6. conduct: respectful, fair, non-abusive, non-evasive conduct

If a side is explicitly labeled with a fallacy_trait variant, do not mark the
turn procedurally invalid or reject the transcript solely because the intended
fallacy is logically invalid. Continue to evaluate grounding, coherence,
readability, responsiveness, and stance consistency.

Assign speaker_points from 24.0 to 30.0 using only full or half points. Do not use arbitrary decimals.
PF scale:
24.0-25.0 = weak / needs improvement
25.5-26.5 = good but flawed
27.0-28.0 = strong / above average
28.5-29.0 = excellent
29.5-30.0 = outstanding

Respond only with a valid JSON object in exactly this format:
{
  "reasoning": "<two to four sentences explaining the speaker points>",
  "category_scores": {
    "written_clarity": 4.0,
    "organization": 4.0,
    "evidence": 3.5,
    "argumentation": 4.0,
    "responsiveness": 4.0,
    "conduct": 5.0
  },
  "speaker_points": 28.0
}

Do not include any text outside the JSON object.\
"""

_BALLOT_SYSTEM_PROMPT = """\
You are an expert Public Forum debate judge adjudicating the full text-based debate round.

You must choose one winner unless the debate is invalid because of a fatal technical violation. Do not declare both sides winners. Do not mechanically average turn scores. Speaker points may inform your view, but the ballot must be decided comparatively based on clash, warrants, evidence, weighing, responsiveness, and final crystallization.

If a side is explicitly labeled with a fallacy_trait variant, do not declare the
round invalid solely because the intended fallacy is logically invalid. Still
judge grounding, coherence, readability, responsiveness, and stance consistency.

Respond only with a valid JSON object in exactly this format:
{
  "winner_side": "A",
  "winner_name": "<variant name>",
  "loser_side": "B",
  "loser_name": "<variant name>",
  "decision_margin": 1.0,
  "main_voting_issue": "<the decisive issue>",
  "comparative_reasoning": "<three to six sentences explaining why the winner won>",
  "decisive_turns": [3, 7, 8]
}

winner_side must be "A", "B", or "invalid". Use "invalid" only for fatal technical violations or impossible-to-judge transcripts.
decision_margin must be one of: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0.
Do not include any text outside the JSON object.\
"""

_TECHNICAL_ROUND_SYSTEM_PROMPT = """\
You are an expert Public Forum debate judge. Evaluate technical adherence for the full text-based debate round.

Technical adherence means whether each agent followed the required debate protocol. Do not score debate quality, persuasion, style, evidence strength, or who is winning.

For each turn, check the listed technical rubric items. Also flag severe procedural problems if visible, such as speaking for the wrong side, refusing to debate, arguing both sides, skipping the assigned task, or using meta-commentary instead of a debate speech.

Respond only with a valid JSON object in exactly this format:
{
  "round_reasoning": "<two to four sentences summarizing technical adherence across the full round>",
  "turn_technical_scores": [
    {
      "turn_id": 1,
      "speaker": "A",
      "reasoning": "<one to two sentences explaining technical adherence for this turn only>",
      "technical_checks": [
        {"title": "<check title>", "passed": true, "severity": "minor"}
      ],
      "fatal_error": false
    }
  ],
  "round_fatal_error": false
}

Severity must be one of: "minor", "major", "fatal".
Set fatal_error or round_fatal_error to true only when the turn or round is procedurally invalid enough that normal debate adjudication should not decide the result.
Return exactly one turn_technical_scores item for every turn in the transcript.
Do not include any text outside the JSON object.\
"""

_QUALITY_ROUND_SYSTEM_PROMPT = """\
You are an expert Public Forum debate judge evaluating speech quality and speaker points for the full text-based debate round.

Use Public Forum speaker point logic, adapted for text-only debate. Separate quality from technical completion.

For each turn, evaluate these categories on a 1-5 scale using full or half points only:
1. written_clarity: clear, fluent, confident written delivery and tone
2. organization: structure, signposting, flow, ease of following the speech
3. evidence: relevant support, examples, mechanisms, factual grounding
4. argumentation: warrants, logic, clash, rebuttal, weighing, and comparative analysis
5. responsiveness: direct engagement with the opponent, pressure, and answer quality
6. conduct: respectful, fair, non-abusive, non-evasive conduct

If a side is explicitly labeled with a fallacy_trait variant, do not mark the
active dialogue invalid solely because the intended fallacy is logically invalid.
Still evaluate grounding, coherence, readability, responsiveness, and stance
consistency.

Assign speaker_points from 24.0 to 30.0 using only full or half points. Do not use arbitrary decimals.
PF scale:
24.0-25.0 = weak / needs improvement
25.5-26.5 = good but flawed
27.0-28.0 = strong / above average
28.5-29.0 = excellent
29.5-30.0 = outstanding

Respond only with a valid JSON object in exactly this format:
{
  "round_reasoning": "<two to four sentences summarizing speech quality across the full round>",
  "turn_quality_scores": [
    {
      "turn_id": 1,
      "speaker": "A",
      "reasoning": "<two to four sentences explaining the speaker points for this turn>",
      "category_scores": {
        "written_clarity": 4.0,
        "organization": 4.0,
        "evidence": 3.5,
        "argumentation": 4.0,
        "responsiveness": 4.0,
        "conduct": 5.0
      },
      "speaker_points": 28.0
    }
  ],
  "side_summaries": {
    "A": {"average_speaker_points": 28.0, "reasoning": "<side A quality summary>"},
    "B": {"average_speaker_points": 27.5, "reasoning": "<side B quality summary>"}
  }
}

Return exactly one turn_quality_scores item for every turn in the transcript.
Do not include any text outside the JSON object.\
"""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _nearest_allowed(value: Any, allowed: list[float], default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return min(allowed, key=lambda item: abs(item - numeric))


class DebateRubricEvaluator:
    """PF-style debate evaluator with separate technical, speaker-point, and ballot layers.

    Public interface is intentionally compatible with the previous RaR evaluator:
    - evaluate_debate_rubric_quality(dialogue) remains the main entry point.
    - score_a / score_b are now average PF speaker points, not 0-10 pass-ratio scores.
    - winner_side / winner_name now come from a comparative ballot, not arithmetic averaging.
    """

    def __init__(
        self,
        client,
        model: str | None,
        rubric_dir: Path,
        pass_threshold: float = 24.0,
        temperature: float = 0,
    ):
        self.client = client
        self.model = model
        self.rubric_dir = Path(rubric_dir)
        self.rubric_store = self.load_rubric_store()
        self.model_cfgs = [{"model": model, "temperature": temperature}] if model else [MODEL_CONFIGS["evaluator"]]
        self.pass_threshold = pass_threshold

    def load_rubric_store(self) -> dict[str, list[dict]]:
        store: dict[str, list[dict]] = {}
        for stem in _RUBRIC_STEMS:
            path = self.rubric_dir / f"{stem}.json"
            store[stem] = json.loads(path.read_text()) if path.exists() else []
        return store

    def get_turn_rubric_sets(self, turn_id: int, attack_move_used: Any) -> tuple[list[dict], list[dict], list[str]]:
        technical_rubrics = self.rubric_store.get("static", [])
        if turn_id in [5, 6, 7, 8]:
            return technical_rubrics, self.rubric_store.get("summary", []), ["technical", "summary"]

        move_map = {
            None: self.rubric_store.get("constructive", []),
            "rebuttal": self.rubric_store.get("rebuttal", []),
            "weigh": self.rubric_store.get("weigh", []),
        }
        quality_rubrics = move_map.get(attack_move_used, self.rubric_store.get("constructive", []))
        move_key = attack_move_used if attack_move_used in ("rebuttal", "weigh") else "constructive"
        return technical_rubrics, quality_rubrics, ["technical", move_key]

    def format_rubrics(self, rubrics: list[dict]) -> str:
        if not rubrics:
            return "N/A"
        lines = []
        for rubric in rubrics:
            weight = rubric.get("weight", "N/A")
            title = rubric.get("title", "Untitled")
            description = rubric.get("description", "")
            lines.append(f"[weight: {weight}] {title}: {description}")
        return "\n".join(lines)

    def parse_judge_response(self, text: str) -> dict:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in: {text!r}")

    def _complete_json(self, prompt: str, *, model: str, temperature: float, max_tokens: int) -> dict:
        last_error: Exception | None = None
        for _attempt in range(2):
            raw = self.client.complete_text(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                return self.parse_judge_response(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError(f"Could not parse judge response: {last_error}")

    def build_technical_prompt(
        self,
        topic: str,
        turn_id: int,
        speaker: str,
        prior_goal: str,
        utterance: str,
        technical_rubrics: list[dict],
    ) -> str:
        return (
            f"{_TECHNICAL_SYSTEM_PROMPT}\n\n"
            f"Resolution: {topic}\n\n"
            f"Turn ID: {turn_id}\n"
            f"Speaker side: {speaker}\n"
            f"Prior turn strategic goal: {prior_goal}\n\n"
            f"Speech text:\n{utterance}\n\n"
            f"Technical rubric items:\n{self.format_rubrics(technical_rubrics)}\n\n"
            "Return the technical adherence JSON."
        )

    def build_quality_prompt(
        self,
        topic: str,
        turn_id: int,
        speaker: str,
        prior_goal: str,
        utterance: str,
        quality_rubrics: list[dict],
    ) -> str:
        return (
            f"{_QUALITY_SYSTEM_PROMPT}\n\n"
            f"Resolution: {topic}\n\n"
            f"Turn ID: {turn_id}\n"
            f"Speaker side: {speaker}\n"
            f"Prior turn strategic goal: {prior_goal}\n\n"
            f"Speech text:\n{utterance}\n\n"
            f"Turn-specific debate-quality rubric context:\n{self.format_rubrics(quality_rubrics)}\n\n"
            "Return the PF-style speech quality JSON."
        )

    def validate_technical_response(self, result: dict, technical_rubrics: list[dict]) -> None:
        if not isinstance(result.get("reasoning"), str):
            raise ValueError("missing technical reasoning")
        if not isinstance(result.get("technical_checks"), list):
            raise ValueError("missing technical_checks")
        expected_min = len(technical_rubrics)
        if expected_min and len(result["technical_checks"]) < expected_min:
            raise ValueError("technical_checks shorter than supplied rubric list")
        if not isinstance(result.get("fatal_error"), bool):
            raise ValueError("missing fatal_error boolean")

    def validate_quality_response(self, result: dict) -> None:
        if not isinstance(result.get("reasoning"), str):
            raise ValueError("missing quality reasoning")
        if not isinstance(result.get("category_scores"), dict):
            raise ValueError("missing category_scores")
        missing = [category for category in PF_TEXT_CATEGORIES if category not in result["category_scores"]]
        if missing:
            raise ValueError(f"missing category scores: {missing}")
        if "speaker_points" not in result:
            raise ValueError("missing speaker_points")

    def compute_technical_score(self, technical_checks: list[dict]) -> tuple[int, int, float | None, bool]:
        total_count = len(technical_checks)
        passed_count = sum(1 for check in technical_checks if check.get("passed") is True)
        score = round((passed_count / total_count) * 100, 1) if total_count else None
        fatal_error = any(
            check.get("passed") is not True and check.get("severity") == "fatal"
            for check in technical_checks
        )
        return passed_count, total_count, score, fatal_error

    def score_technical_turn(
        self,
        topic: str,
        turn_id: int,
        speaker: str,
        prior_goal: str,
        utterance: str,
        technical_rubrics: list[dict],
        *,
        model: str,
        temperature: float = 0,
    ) -> dict:
        prompt = self.build_technical_prompt(topic, turn_id, speaker, prior_goal, utterance, technical_rubrics)
        try:
            result = self._complete_json(prompt, model=model, temperature=temperature, max_tokens=700)
            self.validate_technical_response(result, technical_rubrics)
            for check in result["technical_checks"]:
                if check.get("severity") not in {"minor", "major", "fatal"}:
                    check["severity"] = "minor"
            passed_count, total_count, score, fatal_error_from_checks = self.compute_technical_score(result["technical_checks"])
            result["passed_count"] = passed_count
            result["total_count"] = total_count
            result["score"] = score
            result["fatal_error"] = bool(result.get("fatal_error")) or fatal_error_from_checks
            return result
        except (ValueError, KeyError, json.JSONDecodeError):
            return {
                "reasoning": "parse_error",
                "technical_checks": [],
                "passed_count": None,
                "total_count": 0,
                "score": None,
                "fatal_error": True,
            }

    def score_quality_turn(
        self,
        topic: str,
        turn_id: int,
        speaker: str,
        prior_goal: str,
        utterance: str,
        quality_rubrics: list[dict],
        *,
        model: str,
        temperature: float = 0,
    ) -> dict:
        prompt = self.build_quality_prompt(topic, turn_id, speaker, prior_goal, utterance, quality_rubrics)
        try:
            result = self._complete_json(prompt, model=model, temperature=temperature, max_tokens=900)
            self.validate_quality_response(result)
            result["category_scores"] = {
                category: _nearest_allowed(
                    result["category_scores"].get(category), PF_ALLOWED_CATEGORY_SCORES, 3.0
                )
                for category in PF_TEXT_CATEGORIES
            }
            result["speaker_points"] = _nearest_allowed(
                result.get("speaker_points"), PF_ALLOWED_SPEAKER_POINTS, 27.0
            )
            return result
        except (ValueError, KeyError, json.JSONDecodeError):
            return {
                "reasoning": "parse_error",
                "category_scores": {category: None for category in PF_TEXT_CATEGORIES},
                "speaker_points": None,
            }

    def build_turn_contexts(self, turns: list[Any], variant_a: str, variant_b: str) -> list[dict]:
        contexts = []
        for i, turn in enumerate(turns):
            turn_id = _get(turn, "turn_id")
            speaker = _get(turn, "speaker")
            attack_move_used = _get(turn, "attack_move_used")
            technical_rubrics, quality_rubrics, rubric_set = self.get_turn_rubric_sets(turn_id, attack_move_used)
            contexts.append({
                "turn_id": turn_id,
                "speaker": speaker,
                "variant_name": variant_a if speaker == "A" else variant_b,
                "attack_move_used": attack_move_used,
                "utterance": _get(turn, "utterance", ""),
                "rubric_set": rubric_set,
                "technical_rubrics": technical_rubrics,
                "quality_rubrics": quality_rubrics,
            })
        return contexts

    def _prompt_turn_contexts(self, turn_contexts: list[dict], *, rubric_key: str) -> list[dict]:
        return [
            {
                "turn_id": item["turn_id"],
                "speaker": item["speaker"],
                "variant_name": item["variant_name"],
                "attack_move_used": item["attack_move_used"],
                "prior_goal": item["prior_goal"],
                "speech_text": item["utterance"],
                "rubric_set": item["rubric_set"],
                "rubrics": item[rubric_key],
            }
            for item in turn_contexts
        ]

    def build_technical_round_prompt(
        self,
        topic: str,
        variant_a: str,
        variant_b: str,
        turn_contexts: list[dict],
    ) -> str:
        return (
            f"{_TECHNICAL_ROUND_SYSTEM_PROMPT}\n\n"
            f"Resolution: {topic}\n\n"
            f"Side A variant name: {variant_a}\n"
            f"Side B variant name: {variant_b}\n\n"
            "Full-round turn contexts with technical rubric items:\n"
            f"{json.dumps(self._prompt_turn_contexts(turn_contexts, rubric_key='technical_rubrics'), ensure_ascii=False, indent=2)}\n\n"
            "Return the full-round technical adherence JSON."
        )

    def build_quality_round_prompt(
        self,
        topic: str,
        variant_a: str,
        variant_b: str,
        turn_contexts: list[dict],
    ) -> str:
        return (
            f"{_QUALITY_ROUND_SYSTEM_PROMPT}\n\n"
            f"Resolution: {topic}\n\n"
            f"Side A variant name: {variant_a}\n"
            f"Side B variant name: {variant_b}\n\n"
            "Full-round turn contexts with debate-quality rubric context:\n"
            f"{json.dumps(self._prompt_turn_contexts(turn_contexts, rubric_key='quality_rubrics'), ensure_ascii=False, indent=2)}\n\n"
            "Return the full-round PF-style speech quality JSON."
        )

    def validate_technical_round_response(self, result: dict) -> None:
        if not isinstance(result.get("round_reasoning"), str):
            raise ValueError("missing technical round_reasoning")
        if not isinstance(result.get("turn_technical_scores"), list):
            raise ValueError("missing turn_technical_scores")
        if not isinstance(result.get("round_fatal_error"), bool):
            raise ValueError("missing round_fatal_error boolean")

    def validate_quality_round_response(self, result: dict) -> None:
        if not isinstance(result.get("round_reasoning"), str):
            raise ValueError("missing quality round_reasoning")
        if not isinstance(result.get("turn_quality_scores"), list):
            raise ValueError("missing turn_quality_scores")

    def default_technical_result(self) -> dict:
        return {
            "reasoning": "parse_error",
            "technical_checks": [],
            "passed_count": None,
            "total_count": 0,
            "score": None,
            "fatal_error": True,
        }

    def default_quality_result(self) -> dict:
        return {
            "reasoning": "parse_error",
            "category_scores": {category: None for category in PF_TEXT_CATEGORIES},
            "speaker_points": None,
        }

    def normalize_technical_round(
        self,
        result: dict,
        turn_contexts: list[dict],
    ) -> tuple[dict[tuple[Any, Any], dict], int]:
        raw_items = {
            (item.get("turn_id"), item.get("speaker")): item
            for item in result.get("turn_technical_scores") or []
            if isinstance(item, dict)
        }
        normalized = {}
        parse_errors = 0
        round_fatal_error = bool(result.get("round_fatal_error"))

        for context in turn_contexts:
            key = (context["turn_id"], context["speaker"])
            item = raw_items.get(key)
            if not isinstance(item, dict) or not isinstance(item.get("technical_checks"), list):
                normalized[key] = self.default_technical_result()
                parse_errors += 1
                continue

            for check in item["technical_checks"]:
                if check.get("severity") not in {"minor", "major", "fatal"}:
                    check["severity"] = "minor"
            passed_count, total_count, score, fatal_error_from_checks = self.compute_technical_score(item["technical_checks"])
            fatal_error = bool(item.get("fatal_error")) or fatal_error_from_checks or round_fatal_error
            normalized[key] = {
                "reasoning": item.get("reasoning") if isinstance(item.get("reasoning"), str) else "",
                "technical_checks": item["technical_checks"],
                "passed_count": passed_count,
                "total_count": total_count,
                "score": score,
                "fatal_error": fatal_error,
            }
        return normalized, parse_errors

    def normalize_quality_round(
        self,
        result: dict,
        turn_contexts: list[dict],
    ) -> tuple[dict[tuple[Any, Any], dict], int]:
        raw_items = {
            (item.get("turn_id"), item.get("speaker")): item
            for item in result.get("turn_quality_scores") or []
            if isinstance(item, dict)
        }
        normalized = {}
        parse_errors = 0

        for context in turn_contexts:
            key = (context["turn_id"], context["speaker"])
            item = raw_items.get(key)
            if not isinstance(item, dict):
                normalized[key] = self.default_quality_result()
                parse_errors += 1
                continue
            category_scores = item.get("category_scores")
            if not isinstance(category_scores, dict) or "speaker_points" not in item:
                normalized[key] = self.default_quality_result()
                parse_errors += 1
                continue
            normalized[key] = {
                "reasoning": item.get("reasoning") if isinstance(item.get("reasoning"), str) else "",
                "category_scores": {
                    category: _nearest_allowed(category_scores.get(category), PF_ALLOWED_CATEGORY_SCORES, 3.0)
                    for category in PF_TEXT_CATEGORIES
                },
                "speaker_points": _nearest_allowed(item.get("speaker_points"), PF_ALLOWED_SPEAKER_POINTS, 27.0),
            }
        return normalized, parse_errors

    def score_technical_round(
        self,
        topic: str,
        variant_a: str,
        variant_b: str,
        turn_contexts: list[dict],
        *,
        model: str,
        temperature: float = 0,
    ) -> dict:
        prompt = self.build_technical_round_prompt(topic, variant_a, variant_b, turn_contexts)
        try:
            result = self._complete_json(prompt, model=model, temperature=temperature, max_tokens=4500)
            self.validate_technical_round_response(result)
            turn_results, parse_errors = self.normalize_technical_round(result, turn_contexts)
            return {
                "round_reasoning": result.get("round_reasoning"),
                "turn_results": turn_results,
                "round_fatal_error": bool(result.get("round_fatal_error")),
                "parse_errors": parse_errors,
            }
        except (ValueError, KeyError, json.JSONDecodeError):
            return {
                "round_reasoning": "parse_error",
                "turn_results": {
                    (context["turn_id"], context["speaker"]): self.default_technical_result()
                    for context in turn_contexts
                },
                "round_fatal_error": True,
                "parse_errors": len(turn_contexts),
            }

    def score_quality_round(
        self,
        topic: str,
        variant_a: str,
        variant_b: str,
        turn_contexts: list[dict],
        *,
        model: str,
        temperature: float = 0,
    ) -> dict:
        prompt = self.build_quality_round_prompt(topic, variant_a, variant_b, turn_contexts)
        try:
            result = self._complete_json(prompt, model=model, temperature=temperature, max_tokens=4500)
            self.validate_quality_round_response(result)
            turn_results, parse_errors = self.normalize_quality_round(result, turn_contexts)
            return {
                "round_reasoning": result.get("round_reasoning"),
                "turn_results": turn_results,
                "side_summaries": result.get("side_summaries") if isinstance(result.get("side_summaries"), dict) else {},
                "parse_errors": parse_errors,
            }
        except (ValueError, KeyError, json.JSONDecodeError):
            return {
                "round_reasoning": "parse_error",
                "turn_results": {
                    (context["turn_id"], context["speaker"]): self.default_quality_result()
                    for context in turn_contexts
                },
                "side_summaries": {},
                "parse_errors": len(turn_contexts),
            }

    def build_transcript(self, turns: list[Any]) -> str:
        lines = []
        for turn in turns:
            lines.append(
                f"Turn {_get(turn, 'turn_id')} | Side {_get(turn, 'speaker')} | "
                f"Move {_get(turn, 'attack_move_used', 'N/A')}\n{_get(turn, 'utterance', '')}"
            )
        return "\n\n".join(lines)

    def build_ballot_prompt(
        self,
        topic: str,
        variant_a: str,
        variant_b: str,
        transcript: str,
        turn_scores: list[dict],
        technical_adherence: dict[str, dict],
    ) -> str:
        compact_turn_scores = [
            {
                "turn_id": turn["turn_id"],
                "speaker": turn["speaker"],
                "speaker_points": turn.get("speaker_points"),
                "quality_reasoning": turn.get("quality_reasoning"),
                "technical_score": turn.get("technical_score"),
                "fatal_error": turn.get("fatal_error"),
            }
            for turn in turn_scores
        ]
        return (
            f"{_BALLOT_SYSTEM_PROMPT}\n\n"
            f"Resolution: {topic}\n\n"
            f"Side A variant name: {variant_a}\n"
            f"Side B variant name: {variant_b}\n\n"
            f"Technical adherence summary:\n{json.dumps(technical_adherence, ensure_ascii=False, indent=2)}\n\n"
            f"Turn speaker-point summaries:\n{json.dumps(compact_turn_scores, ensure_ascii=False, indent=2)}\n\n"
            f"Full transcript:\n{transcript}\n\n"
            "Return the comparative PF ballot JSON."
        )

    def validate_ballot_response(self, result: dict) -> None:
        if result.get("winner_side") not in {"A", "B", "invalid"}:
            raise ValueError("winner_side must be A, B, or invalid")
        if result.get("winner_side") in {"A", "B"}:
            expected_loser = "B" if result["winner_side"] == "A" else "A"
            if result.get("loser_side") != expected_loser:
                raise ValueError("loser_side inconsistent with winner_side")
        if "decision_margin" not in result:
            raise ValueError("missing decision_margin")
        if not isinstance(result.get("comparative_reasoning"), str):
            raise ValueError("missing comparative_reasoning")
        if not isinstance(result.get("decisive_turns"), list):
            raise ValueError("missing decisive_turns")

    def adjudicate_round(
        self,
        topic: str,
        variant_a: str,
        variant_b: str,
        turns: list[Any],
        turn_scores: list[dict],
        technical_adherence: dict[str, dict],
        *,
        model: str,
        temperature: float = 0,
    ) -> dict:
        fatal_override = None
        if technical_adherence["A"].get("fatal_error") and not technical_adherence["B"].get("fatal_error"):
            fatal_override = {
                "winner_side": "B",
                "winner_name": variant_b,
                "loser_side": "A",
                "loser_name": variant_a,
                "decision_margin": 3.0,
                "main_voting_issue": "Fatal technical violation by side A",
                "comparative_reasoning": "Side A had a fatal technical violation, so the round cannot be awarded to A on normal debate quality.",
                "decisive_turns": [turn["turn_id"] for turn in turn_scores if turn["speaker"] == "A" and turn.get("fatal_error")],
            }
        elif technical_adherence["B"].get("fatal_error") and not technical_adherence["A"].get("fatal_error"):
            fatal_override = {
                "winner_side": "A",
                "winner_name": variant_a,
                "loser_side": "B",
                "loser_name": variant_b,
                "decision_margin": 3.0,
                "main_voting_issue": "Fatal technical violation by side B",
                "comparative_reasoning": "Side B had a fatal technical violation, so the round cannot be awarded to B on normal debate quality.",
                "decisive_turns": [turn["turn_id"] for turn in turn_scores if turn["speaker"] == "B" and turn.get("fatal_error")],
            }
        elif technical_adherence["A"].get("fatal_error") and technical_adherence["B"].get("fatal_error"):
            fatal_override = {
                "winner_side": "invalid",
                "winner_name": "invalid",
                "loser_side": "invalid",
                "loser_name": "invalid",
                "decision_margin": 3.0,
                "main_voting_issue": "Fatal technical violations by both sides",
                "comparative_reasoning": "Both sides had fatal technical violations, so the transcript is invalid for normal adjudication.",
                "decisive_turns": [],
            }

        transcript = self.build_transcript(turns)
        prompt = self.build_ballot_prompt(topic, variant_a, variant_b, transcript, turn_scores, technical_adherence)
        try:
            result = self._complete_json(prompt, model=model, temperature=temperature, max_tokens=1400)
            self.validate_ballot_response(result)
            result["decision_margin"] = _nearest_allowed(
                result.get("decision_margin"), PF_ALLOWED_BALLOT_MARGINS, 1.0
            )
            if fatal_override is not None:
                fatal_override["judge_ballot"] = result
                return fatal_override
            if result["winner_side"] == "A":
                result["winner_name"] = variant_a
                result["loser_side"] = "B"
                result["loser_name"] = variant_b
            elif result["winner_side"] == "B":
                result["winner_name"] = variant_b
                result["loser_side"] = "A"
                result["loser_name"] = variant_a
            else:
                result["winner_name"] = "invalid"
                result["loser_side"] = "invalid"
                result["loser_name"] = "invalid"
            return result
        except (ValueError, KeyError, json.JSONDecodeError):
            if fatal_override is not None:
                return fatal_override
            return self.fallback_ballot(variant_a, variant_b, turn_scores)

    def fallback_ballot(self, variant_a: str, variant_b: str, turn_scores: list[dict]) -> dict:
        a_points = [turn["speaker_points"] for turn in turn_scores if turn["speaker"] == "A" and turn.get("speaker_points") is not None]
        b_points = [turn["speaker_points"] for turn in turn_scores if turn["speaker"] == "B" and turn.get("speaker_points") is not None]
        a_avg = sum(a_points) / len(a_points) if a_points else 0
        b_avg = sum(b_points) / len(b_points) if b_points else 0
        if a_avg >= b_avg:
            winner_side, winner_name, loser_side, loser_name = "A", variant_a, "B", variant_b
        else:
            winner_side, winner_name, loser_side, loser_name = "B", variant_b, "A", variant_a
        return {
            "winner_side": winner_side,
            "winner_name": winner_name,
            "loser_side": loser_side,
            "loser_name": loser_name,
            "decision_margin": 0.5,
            "main_voting_issue": "Fallback decision from average speaker points after ballot parse error",
            "comparative_reasoning": "The comparative ballot judge returned an invalid response, so this fallback chooses the side with the higher average PF speaker points. Re-run the evaluation for a full comparative reason.",
            "decisive_turns": [],
        }

    def aggregate_side_technical(self, turn_scores: list[dict], speaker: str) -> dict:
        side_turns = [turn for turn in turn_scores if turn["speaker"] == speaker]
        passed = sum(turn.get("technical_passed_count") or 0 for turn in side_turns)
        total = sum(turn.get("technical_total_count") or 0 for turn in side_turns)
        score = round((passed / total) * 100, 1) if total else None
        fatal_error = any(turn.get("fatal_error") for turn in side_turns)
        return {
            "passed_count": passed,
            "total_count": total,
            "score": score,
            "fatal_error": fatal_error,
        }

    def evaluate_single_model(self, dialogue, *, model: str, temperature: float = 0) -> dict:
        turns = _get(dialogue, "turns", [])
        topic = _get(dialogue, "topic")
        trait_name = _get(dialogue, "trait_name")
        variant_a = _get(dialogue, "variant_name_a")
        variant_b = _get(dialogue, "variant_name_b")

        turn_contexts = self.build_turn_contexts(turns, variant_a, variant_b)
        technical_round = self.score_technical_round(
            topic, variant_a, variant_b, turn_contexts,
            model=model, temperature=temperature,
        )
        quality_round = self.score_quality_round(
            topic, variant_a, variant_b, turn_contexts,
            model=model, temperature=temperature,
        )

        turn_scores = []
        parse_errors = technical_round.get("parse_errors", 0) + quality_round.get("parse_errors", 0)

        for context in turn_contexts:
            key = (context["turn_id"], context["speaker"])
            technical = technical_round["turn_results"][key]
            quality = quality_round["turn_results"][key]

            turn_scores.append({
                "turn_id": context["turn_id"],
                "speaker": context["speaker"],
                "variant_name": context["variant_name"],
                "attack_move_used": context["attack_move_used"],
                "rubric_set": context["rubric_set"],

                "technical_passed_count": technical["passed_count"],
                "technical_total_count": technical["total_count"],
                "technical_score": technical["score"],
                "technical_reasoning": technical["reasoning"],
                "technical_checks": technical["technical_checks"],
                "fatal_error": technical["fatal_error"],

                "category_scores": quality["category_scores"],
                "speaker_points": quality["speaker_points"],
                "quality_reasoning": quality["reasoning"],

                # Backward-compatible aliases. These now refer to technical checks and PF points.
                "passed_count": technical["passed_count"],
                "total_count": technical["total_count"],
                "rating": quality["speaker_points"],
                "passed": (
                    technical["passed_count"] == technical["total_count"]
                    if technical["passed_count"] is not None and technical["total_count"] > 0
                    else False
                ),
                "reasoning": quality["reasoning"],
                "rubric_checks": technical["technical_checks"],
                "evaluation_granularity": "round",
            })

        a_points = [turn["speaker_points"] for turn in turn_scores if turn["speaker"] == "A" and turn.get("speaker_points") is not None]
        b_points = [turn["speaker_points"] for turn in turn_scores if turn["speaker"] == "B" and turn.get("speaker_points") is not None]
        score_a = round(sum(a_points) / len(a_points), 2) if a_points else None
        score_b = round(sum(b_points) / len(b_points), 2) if b_points else None

        technical_adherence = {
            "A": self.aggregate_side_technical(turn_scores, "A"),
            "B": self.aggregate_side_technical(turn_scores, "B"),
        }
        ballot = self.adjudicate_round(
            topic, variant_a, variant_b, turns, turn_scores, technical_adherence,
            model=model, temperature=temperature,
        )

        winner_side = (
            "variant_a" if ballot["winner_side"] == "A"
            else "variant_b" if ballot["winner_side"] == "B"
            else "invalid"
        )
        winner_name = ballot.get("winner_name")
        margin = ballot.get("decision_margin")
        round_reasoning = ballot.get("comparative_reasoning")
        passed = (
            parse_errors == 0
            and not technical_adherence["A"]["fatal_error"]
            and not technical_adherence["B"]["fatal_error"]
            and score_a is not None
            and score_b is not None
            and score_a >= self.pass_threshold
            and score_b >= self.pass_threshold
        )

        print("[debug] PF turn scores:")
        for turn in turn_scores:
            print(
                f"  - turn {turn['turn_id']} {turn['speaker']} "
                f"{turn['rubric_set'][-1]} tech={turn['technical_passed_count']}/{turn['technical_total_count']} "
                f"tech_score={turn['technical_score']} sp={turn['speaker_points']} fatal={turn['fatal_error']}"
            )
        print(
            "[debug] PF aggregate: "
            f"score_a={score_a} score_b={score_b} ballot_winner={winner_name} margin={margin} "
            f"parse_errors={parse_errors} passed={passed}"
        )

        return {
            "model": model,
            "topic": topic,
            "trait_name": trait_name,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "score_a": score_a,
            "score_b": score_b,
            "speaker_points": {"side_a_avg": score_a, "side_b_avg": score_b},
            "technical_adherence": technical_adherence,
            "technical_round_reasoning": technical_round.get("round_reasoning"),
            "quality_round_reasoning": quality_round.get("round_reasoning"),
            "quality_side_summaries": quality_round.get("side_summaries", {}),
            "ballot": ballot,
            "winner_side": winner_side,
            "winner_name": winner_name,
            "margin": margin,
            "round_reasoning": round_reasoning,
            "turn_scores": turn_scores,
            "parse_errors": parse_errors,
            "passed": passed,
            "evaluation_granularity": "round",
            "llm_call_count": 3,
        }

    def aggregate_turns_across_models(self, per_model_results: list[dict]) -> list[dict]:
        grouped_turns: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for result in per_model_results:
            for turn in result.get("turn_scores") or []:
                grouped_turns[(turn["turn_id"], turn["speaker"])].append(turn)

        turn_scores = []
        for key in sorted(grouped_turns, key=lambda item: item[0]):
            items = grouped_turns[key]
            base = items[0]
            speaker_points = [item["speaker_points"] for item in items if item.get("speaker_points") is not None]
            technical_scores = [item["technical_score"] for item in items if item.get("technical_score") is not None]

            checks_by_title: dict[str, list[bool]] = defaultdict(list)
            checks_by_severity: dict[str, str] = {}
            for item in items:
                for check in item.get("technical_checks") or []:
                    title = check.get("title")
                    if not isinstance(title, str):
                        continue
                    checks_by_title[title].append(check.get("passed") is True)
                    checks_by_severity[title] = check.get("severity", checks_by_severity.get(title, "minor"))

            aggregated_checks = [
                {
                    "title": title,
                    "passed": all(values),
                    "severity": checks_by_severity.get(title, "minor"),
                }
                for title, values in checks_by_title.items()
            ]
            technical_passed = sum(1 for check in aggregated_checks if check["passed"] is True)
            technical_total = len(aggregated_checks)

            category_scores: dict[str, float | None] = {}
            for category in PF_TEXT_CATEGORIES:
                values = [
                    item.get("category_scores", {}).get(category)
                    for item in items
                    if item.get("category_scores", {}).get(category) is not None
                ]
                category_scores[category] = round(sum(values) / len(values), 2) if values else None

            turn_scores.append({
                "turn_id": base["turn_id"],
                "speaker": base["speaker"],
                "variant_name": base["variant_name"],
                "attack_move_used": base["attack_move_used"],
                "rubric_set": base["rubric_set"],
                "technical_passed_count": technical_passed if technical_total else None,
                "technical_total_count": technical_total,
                "technical_score": round(sum(technical_scores) / len(technical_scores), 1) if technical_scores else None,
                "technical_checks": aggregated_checks,
                "fatal_error": any(item.get("fatal_error") for item in items),
                "category_scores": category_scores,
                "speaker_points": round(sum(speaker_points) / len(speaker_points), 2) if speaker_points else None,
                "quality_reasoning": base.get("quality_reasoning"),
                # Backward-compatible aliases.
                "passed_count": technical_passed if technical_total else None,
                "total_count": technical_total,
                "rating": round(sum(speaker_points) / len(speaker_points), 2) if speaker_points else None,
                "passed": technical_passed == technical_total if technical_total else False,
                "reasoning": base.get("quality_reasoning"),
                "rubric_checks": aggregated_checks,
                "evaluation_granularity": base.get("evaluation_granularity", "turn"),
            })
        return turn_scores

    def aggregate_ballot_across_models(
        self,
        per_model_results: list[dict],
        score_a: float | None,
        score_b: float | None,
        variant_a: str,
        variant_b: str,
    ) -> dict:
        ballots = [result.get("ballot") for result in per_model_results if isinstance(result.get("ballot"), dict)]
        valid_sides = [ballot.get("winner_side") for ballot in ballots if ballot.get("winner_side") in {"A", "B", "invalid"}]
        if not valid_sides:
            return {
                "winner_side": "invalid",
                "winner_name": "invalid",
                "loser_side": "invalid",
                "loser_name": "invalid",
                "decision_margin": 3.0,
                "main_voting_issue": "No valid ballot produced",
                "comparative_reasoning": "No evaluator produced a valid comparative ballot.",
                "decisive_turns": [],
            }

        counts = Counter(valid_sides)
        top_count = max(counts.values())
        candidates = [side for side, count in counts.items() if count == top_count]

        if len(candidates) == 1:
            winning_side = candidates[0]
        elif score_a is not None and score_b is not None and score_a != score_b:
            winning_side = "A" if score_a > score_b else "B"
        else:
            # Deterministic fallback: use the first non-invalid ballot rather than returning a tie.
            winning_side = next((side for side in valid_sides if side in {"A", "B"}), "invalid")

        side_ballots = [ballot for ballot in ballots if ballot.get("winner_side") == winning_side]
        representative = side_ballots[0] if side_ballots else ballots[0]
        margins = [
            _nearest_allowed(ballot.get("decision_margin"), PF_ALLOWED_BALLOT_MARGINS, 1.0)
            for ballot in side_ballots
        ]
        avg_margin = _nearest_allowed(sum(margins) / len(margins), PF_ALLOWED_BALLOT_MARGINS, 1.0) if margins else 1.0

        if winning_side == "A":
            winner_name, loser_side, loser_name = variant_a, "B", variant_b
        elif winning_side == "B":
            winner_name, loser_side, loser_name = variant_b, "A", variant_a
        else:
            winner_name, loser_side, loser_name = "invalid", "invalid", "invalid"

        return {
            "winner_side": winning_side,
            "winner_name": winner_name,
            "loser_side": loser_side,
            "loser_name": loser_name,
            "decision_margin": avg_margin,
            "main_voting_issue": representative.get("main_voting_issue", "N/A"),
            "comparative_reasoning": representative.get("comparative_reasoning", ""),
            "decisive_turns": representative.get("decisive_turns", []),
            "model_vote_counts": dict(counts),
        }

    def evaluate_debate_rubric_quality(self, dialogue) -> dict:
        per_model_results = [
            self.evaluate_single_model(
                dialogue,
                model=cfg["model"],
                temperature=cfg.get("temperature", 0),
            )
            for cfg in self.model_cfgs
        ]

        topic = per_model_results[0]["topic"] if per_model_results else None
        trait_name = per_model_results[0]["trait_name"] if per_model_results else None
        variant_a = per_model_results[0]["variant_a"] if per_model_results else None
        variant_b = per_model_results[0]["variant_b"] if per_model_results else None

        valid_score_as = [result["score_a"] for result in per_model_results if result.get("score_a") is not None]
        valid_score_bs = [result["score_b"] for result in per_model_results if result.get("score_b") is not None]
        score_a = round(sum(valid_score_as) / len(valid_score_as), 2) if valid_score_as else None
        score_b = round(sum(valid_score_bs) / len(valid_score_bs), 2) if valid_score_bs else None

        turn_scores = self.aggregate_turns_across_models(per_model_results)
        technical_adherence = {
            "A": self.aggregate_side_technical(turn_scores, "A"),
            "B": self.aggregate_side_technical(turn_scores, "B"),
        }
        ballot = self.aggregate_ballot_across_models(per_model_results, score_a, score_b, variant_a, variant_b)

        winner_side = (
            "variant_a" if ballot["winner_side"] == "A"
            else "variant_b" if ballot["winner_side"] == "B"
            else "invalid"
        )
        winner_name = ballot.get("winner_name")
        margin = ballot.get("decision_margin")
        parse_errors = sum(result.get("parse_errors", 0) for result in per_model_results)
        passed = (
            parse_errors == 0
            and not technical_adherence["A"].get("fatal_error")
            and not technical_adherence["B"].get("fatal_error")
            and score_a is not None
            and score_b is not None
            and score_a >= self.pass_threshold
            and score_b >= self.pass_threshold
            and ballot.get("winner_side") in {"A", "B"}
        )
        round_reasoning = ballot.get("comparative_reasoning")
        technical_round_reasoning = " ".join(
            result.get("technical_round_reasoning", "")
            for result in per_model_results[:2]
            if result.get("technical_round_reasoning")
        )
        quality_round_reasoning = " ".join(
            result.get("quality_round_reasoning", "")
            for result in per_model_results[:2]
            if result.get("quality_round_reasoning")
        )

        print(
            "[debug] PF cross-model aggregate: "
            f"score_a={score_a} score_b={score_b} winner={winner_name} margin={margin} "
            f"models_passed={sum(1 for r in per_model_results if r.get('passed'))}/{len(per_model_results)} "
            f"passed={passed} parse_errors={parse_errors}"
        )

        return {
            "topic": topic,
            "trait_name": trait_name,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "score_a": score_a,
            "score_b": score_b,
            "speaker_points": {"side_a_avg": score_a, "side_b_avg": score_b},
            "technical_adherence": technical_adherence,
            "technical_round_reasoning": technical_round_reasoning,
            "quality_round_reasoning": quality_round_reasoning,
            "ballot": ballot,
            "winner_side": winner_side,
            "winner_name": winner_name,
            "margin": margin,
            "round_reasoning": round_reasoning,
            "turn_scores": turn_scores,
            "per_model_results": per_model_results,
            "parse_errors": parse_errors,
            "passed": passed,
            "evaluation_granularity": "round",
            "llm_call_count": sum(result.get("llm_call_count", 0) for result in per_model_results),
        }
