"""Runtime settings for bias generation and evaluation.

These values are operational pipeline configuration, not semantic bias
definitions.
"""

PAIRWISE_BIAS_TURN_TYPES = ("argument", "rebuttal")
PAIRWISE_REASONING_SENTENCE_ONLY = True
PAIRWISE_FACTS_AND_CONCLUSIONS_FIXED = True
PAIRWISE_STYLE_PLAN_OUTPUT = "reasoning_guidelines"

VERBOSITY_WORD_GAP_MEASURE = "average_turn_words"
VERBOSITY_RELATIVE_WORD_GAP_THRESHOLD = 25
VERBOSITY_ACTIVE_BASELINE_WORD_RATIO_TARGET = 1.5
VERBOSITY_BASELINE_WORD_RANGE = (25, 55)
VERBOSITY_ACTIVE_WORD_RANGE = (25, 55)
VERBOSITY_ACTIVE_PAIR_WORD_GAP_RANGE = (0, 10)


def get_verbosity_word_gap_measure() -> str:
    return VERBOSITY_WORD_GAP_MEASURE


def get_verbosity_word_gap_threshold() -> int:
    return VERBOSITY_RELATIVE_WORD_GAP_THRESHOLD


def get_verbosity_active_pair_gap_range() -> tuple[int, int]:
    return VERBOSITY_ACTIVE_PAIR_WORD_GAP_RANGE


def get_verbosity_word_ratio_target() -> float:
    return VERBOSITY_ACTIVE_BASELINE_WORD_RATIO_TARGET


def get_verbosity_word_range(variant_name: str) -> tuple[int, int]:
    if variant_name == "active":
        return VERBOSITY_ACTIVE_WORD_RANGE
    return VERBOSITY_BASELINE_WORD_RANGE
