from __future__ import annotations

import re


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def dialogue_to_text(dialogue) -> str:
    lines = []
    for turn in dialogue.turns:
        lines.append(f"Turn {turn.turn_id} | {turn.speaker}: {turn.utterance}")
    return "\n".join(lines)
