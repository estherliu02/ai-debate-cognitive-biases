from __future__ import annotations

from abc import ABC, abstractmethod
from math import floor

from configs import bias_runtime, traitsV2
from utils.text_metrics import count_words


def split_turns_into_rounds(turns) -> list[list]:
    rounds: list[list] = []
    turn_list = list(turns)
    for idx in range(0, len(turn_list), 2):
        rounds.append(turn_list[idx:idx + 2])
    return rounds


def _is_verbosity_target_turn(turn) -> bool:
    return getattr(turn, "turn_id", None) in {3, 4, 5, 6}


def aggregate_round_scores(round_scores: list[dict]) -> tuple[int, str]:
    if not round_scores:
        return 1, "No rounds available for trait evaluation."
    numeric_scores = [item["score"] for item in round_scores if isinstance(item.get("score"), int)]
    if not numeric_scores:
        return 1, "No valid round trait scores were produced."
    overall = min(numeric_scores)
    summary = ", ".join(f"R{item['round_idx']}={item['score']}" for item in round_scores)
    return overall, f"Per-round trait scores: {summary}. Overall score is the minimum round score."


def collect_speaker_word_stats(dialogue, *, target_turns_only: bool = False) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for turn in dialogue.turns:
        if target_turns_only and not _is_verbosity_target_turn(turn):
            continue
        speaker = turn.speaker
        words = count_words(turn.utterance)
        bucket = stats.setdefault(
            speaker,
            {"total_words": 0, "turn_count": 0, "average_turn_words": 0.0, "turn_word_counts": []},
        )
        bucket["total_words"] += words
        bucket["turn_count"] += 1
        bucket["turn_word_counts"].append(words)

    for bucket in stats.values():
        turn_count = bucket["turn_count"]
        bucket["average_turn_words"] = bucket["total_words"] / turn_count if turn_count else 0.0
    return stats


def build_verbosity_gap_report(dialogue, speaker_variants: dict[str, str]) -> dict:
    stats = collect_speaker_word_stats(dialogue, target_turns_only=True)
    measure = "paired_turn_3_4_and_5_6_word_gap"
    threshold = bias_runtime.get_verbosity_word_gap_threshold()
    min_gap, max_gap = bias_runtime.get_verbosity_active_pair_gap_range()
    ratio_target = bias_runtime.get_verbosity_word_ratio_target()
    turns_by_id = {getattr(turn, "turn_id", None): turn for turn in dialogue.turns}

    speaker_payload = {}
    for speaker in sorted(speaker_variants):
        bucket = stats.get(
            speaker,
            {"total_words": 0, "turn_count": 0, "average_turn_words": 0.0, "turn_word_counts": []},
        )
        speaker_payload[speaker] = {
            "variant": speaker_variants[speaker],
            "total_words": bucket["total_words"],
            "turn_count": bucket["turn_count"],
            "average_turn_words": round(bucket["average_turn_words"], 2),
            "turn_word_counts": list(bucket["turn_word_counts"]),
        }

    def turn_words(turn_id: int) -> int:
        turn = turns_by_id.get(turn_id)
        return count_words(turn.utterance) if turn is not None else 0

    pair_word_counts = {
        "3_vs_4": {
            "A": turn_words(3),
            "B": turn_words(4),
            "absolute_gap": abs(turn_words(3) - turn_words(4)),
            "a_minus_b": turn_words(3) - turn_words(4),
        },
        "5_vs_6": {
            "A": turn_words(5),
            "B": turn_words(6),
            "absolute_gap": abs(turn_words(5) - turn_words(6)),
            "a_minus_b": turn_words(5) - turn_words(6),
        },
    }
    round_gaps = [
        {
            "round_idx": 1,
            "turn_ids": [3, 4],
            "words_a": pair_word_counts["3_vs_4"]["A"],
            "words_b": pair_word_counts["3_vs_4"]["B"],
            "a_minus_b": pair_word_counts["3_vs_4"]["a_minus_b"],
            "absolute_gap": pair_word_counts["3_vs_4"]["absolute_gap"],
            "comparable": True,
        },
        {
            "round_idx": 2,
            "turn_ids": [5, 6],
            "words_a": pair_word_counts["5_vs_6"]["A"],
            "words_b": pair_word_counts["5_vs_6"]["B"],
            "a_minus_b": pair_word_counts["5_vs_6"]["a_minus_b"],
            "absolute_gap": pair_word_counts["5_vs_6"]["absolute_gap"],
            "comparable": True,
        },
    ]

    words_a = int(speaker_payload.get("A", {}).get("total_words", 0))
    words_b = int(speaker_payload.get("B", {}).get("total_words", 0))
    speaker_signed_gap = words_a - words_b
    absolute_gap = max((item["absolute_gap"] for item in round_gaps), default=0)
    if all(item["a_minus_b"] > threshold for item in round_gaps):
        verbosity_signature_speaker = "A"
    elif all(-item["a_minus_b"] > threshold for item in round_gaps):
        verbosity_signature_speaker = "B"
    else:
        verbosity_signature_speaker = None

    mode = "other"
    passed = True
    gap = 0.0
    signed_gap = speaker_signed_gap
    reason = "Verbosity paired Turn 3/4 and Turn 5/6 check not applicable for this variant pairing."
    active_speakers = [speaker for speaker, variant in speaker_variants.items() if variant == "active"]
    baseline_speakers = [speaker for speaker, variant in speaker_variants.items() if variant == "baseline"]

    if len(baseline_speakers) == len(speaker_variants) == 2:
        mode = "baseline_vs_baseline"
        gap = absolute_gap
        passed = all(item["absolute_gap"] <= threshold for item in round_gaps)
        reason = (
            "Baseline verbosity balance check: "
            f"each paired absolute gap for Turn 3/4 and Turn 5/6 must be <= {threshold}; "
            f"pair_gaps={[item['absolute_gap'] for item in round_gaps]}."
        )
    elif len(active_speakers) == 1 and len(baseline_speakers) == 1:
        mode = "active_vs_baseline"
        active_speaker = active_speakers[0]
        baseline_speaker = baseline_speakers[0]
        active_words = int(speaker_payload[active_speaker]["total_words"])
        baseline_words = int(speaker_payload[baseline_speaker]["total_words"])
        signed_gap = active_words - baseline_words
        gap = signed_gap
        if active_speaker == "A":
            speaker_signed_gap = signed_gap
            active_pair_gaps = [
                pair_word_counts["3_vs_4"]["a_minus_b"],
                pair_word_counts["5_vs_6"]["a_minus_b"],
            ]
        else:
            speaker_signed_gap = -signed_gap
            active_pair_gaps = [
                -pair_word_counts["3_vs_4"]["a_minus_b"],
                -pair_word_counts["5_vs_6"]["a_minus_b"],
            ]
        for idx, active_pair_gap in enumerate(active_pair_gaps):
            round_gaps[idx]["active_minus_baseline"] = active_pair_gap
        gap = min(active_pair_gaps) if active_pair_gaps else 0
        if traitsV2.TRAIT_LIBRARY.get("verbosity_bias") and traitsV2.TRAIT_LIBRARY["verbosity_bias"]:
            passed = all(min_gap <= pair_gap <= max_gap for pair_gap in active_pair_gaps)
            reason = (
                "Active verbosity compactness check: "
                f"{active_speaker} active must stay comparable to {baseline_speaker} baseline in both paired rounds "
                f"with active_minus_baseline gaps within [{min_gap}, {max_gap}] so bias is not expressed through added length; "
                f"pair_gaps={active_pair_gaps}."
            )

    return {
        "measure": measure,
        "threshold": threshold,
        "min_gap": min_gap,
        "max_gap": max_gap,
        "active_baseline_word_ratio_target": ratio_target,
        "mode": mode,
        "passed": passed,
        "gap": gap,
        "signed_gap": signed_gap,
        "speaker_signed_gap_a_minus_b": speaker_signed_gap,
        "absolute_gap": absolute_gap,
        "words_a": words_a,
        "words_b": words_b,
        "verbosity_signature_speaker": verbosity_signature_speaker,
        "pair_word_counts": pair_word_counts,
        "round_gaps": round_gaps,
        "speaker_stats": speaker_payload,
        "reason": reason,
    }


class TraitEvaluator(ABC):
    """Base class for per-trait evaluation logic.

    Subclasses that evaluate programmatically set uses_llm = False.
    The main evaluator will skip asking the LLM for target_trait_score
    and call score() instead.

    Subclasses that delegate to the LLM set uses_llm = True, and
    score() will never be called — the LLM handles it via the prompt.

    generation_constraint() returns a hard instruction injected into the
    speaker turn prompt so generation and evaluation use the same rule.
    """

    uses_llm: bool = False

    @abstractmethod
    def score(self, dialogue, speaker_variants: dict[str, str]) -> tuple[int, str]:
        """Return (target_trait_score 1-5, reason string).
        speaker_variants maps speaker name to their variant, e.g. {"A": "active", "B": "baseline"}.
        """
        ...

    def score_rounds(self, dialogue, speaker_variants: dict[str, str]) -> list[dict]:
        """Return per-round trait results.

        Each item is a dict with:
        - round_idx: 1-based round number
        - score: integer 1-5
        - reason: short explanation
        """
        score, reason = self.score(dialogue, speaker_variants)
        return [{"round_idx": 1, "score": score, "reason": reason}]

    def generation_constraint(self, variant_name: str) -> str | None:
        """Return a hard constraint string to inject into the speaker turn prompt.
        Return None if no extra constraint is needed beyond the style rules."""
        return None

    def check_turn(self, turn, variant_name: str) -> tuple[bool, str]:
        """Validate a single generated turn against the trait constraint.
        Return (passed, reason). Default: always pass (no per-turn check needed)."""
        return True, ""

    def evaluation_guidance(self, speaker_variants: dict[str, str]) -> str | None:
        """Return trait-specific scoring guidance to inject into the evaluator prompt.
        Describes what the LLM should look for when scoring target_trait_score.
        Return None to use the generic guidance."""
        return None


class GenericLLMTraitEvaluator(TraitEvaluator):
    """LLM-evaluated trait that reads all guidance from TRAIT_LIBRARY.

    To add a new LLM-evaluated trait, just add it to TRAIT_LIBRARY with
    eval_guidance per variant.
    No code changes needed here.
    """

    uses_llm = True

    def __init__(self, trait_name: str):
        self.trait_name = trait_name

    def score(self, dialogue, speaker_variants: dict[str, str]) -> tuple[int, str]:
        raise NotImplementedError

    def evaluation_guidance(self, speaker_variants: dict[str, str]) -> str | None:
        trait = traitsV2.TRAIT_LIBRARY[self.trait_name]
        header = f"Evaluate whether each speaker demonstrates '{self.trait_name}' as assigned."
        lines = [
            trait[variant]["eval_guidance"].format(s=s)
            for s, variant in speaker_variants.items()
        ]
        return (
            header
            + "\n"
            + "\n".join(lines)
            + "\nScore the target trait round by round. For each round, judge whether both speakers match their assigned variants in that round. "
              "Then set the overall target_trait_score to the minimum of the per-round scores."
        )


class VerbosityEvaluator(TraitEvaluator):
    """Programmatic evaluator for verbosity_bias — checks word counts directly."""

    uses_llm = False

    def generation_constraint(self, variant_name: str) -> str | None:
        low, high = bias_runtime.get_verbosity_word_range(variant_name)
        return (
            f"For rebuttal and summary only, your response MUST be exactly two sentences and between {low} and {high} words. "
            "Final focus is exempt from this range and must remain one sentence."
        )

    def check_turn(self, turn, variant_name: str) -> tuple[bool, str]:
        if not _is_verbosity_target_turn(turn):
            return True, ""
        low, high = bias_runtime.get_verbosity_word_range(variant_name)
        n = count_words(turn.utterance)
        if n < low or n > high + 15:
            return False, f"Turn {turn.turn_id} (speaker {turn.speaker}) has {n} words, outside [{low}-{high}]."
        return True, ""

    def score(self, dialogue, speaker_variants: dict[str, str]) -> tuple[int, str]:
        round_scores = self.score_rounds(dialogue, speaker_variants)
        overall, reason = aggregate_round_scores(round_scores)
        gap_report = build_verbosity_gap_report(dialogue, speaker_variants)
        if gap_report["mode"] in {"active_vs_baseline", "baseline_vs_baseline"}:
            if not gap_report["passed"]:
                overall = 1
            reason = f"{reason} {gap_report['reason']}"
        return overall, reason

    def score_rounds(self, dialogue, speaker_variants: dict[str, str]) -> list[dict]:
        word_ranges = {
            speaker: bias_runtime.get_verbosity_word_range(variant)
            for speaker, variant in speaker_variants.items()
        }
        round_scores: list[dict] = []
        for round_idx, round_turns in enumerate(split_turns_into_rounds(dialogue.turns), start=1):
            if not any(_is_verbosity_target_turn(turn) for turn in round_turns):
                continue
            violations: list[str] = []
            severities: list[float] = []
            for turn in round_turns:
                if turn.speaker not in word_ranges:
                    continue
                low, high = word_ranges[turn.speaker]
                n = count_words(turn.utterance)
                if n < low:
                    gap = low - n
                    violations.append(f"{turn.speaker} has {n} words below {low}")
                    severities.append(gap / max(low, 1))
                elif n > high + 15:
                    gap = n - (high + 15)
                    violations.append(f"{turn.speaker} has {n} words above {high}")
                    severities.append(gap / max(high, 1))

            if not violations:
                round_scores.append({
                    "round_idx": round_idx,
                    "score": 5,
                    "reason": "All turns in this round are within the expected word-count range.",
                })
                continue

            max_severity = max(severities, default=1.0)
            penalty_steps = min(4, max(1, floor(max_severity * 4) + 1))
            score = max(1, 5 - penalty_steps)
            round_scores.append({
                "round_idx": round_idx,
                "score": score,
                "reason": "; ".join(violations) + ".",
            })

        return round_scores


# Registry is auto-populated from TRAIT_LIBRARY.
# Traits with a programmatic evaluator override the generic entry below.
def _build_registry() -> dict[str, TraitEvaluator]:
    registry = {
        trait_name: GenericLLMTraitEvaluator(trait_name)
        for trait_name in traitsV2.TRAIT_LIBRARY
    }
    registry["verbosity_bias"] = VerbosityEvaluator()
    return registry


TRAIT_EVALUATOR_REGISTRY: dict[str, TraitEvaluator] = _build_registry()
