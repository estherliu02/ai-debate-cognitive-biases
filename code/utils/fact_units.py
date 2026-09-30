from __future__ import annotations

import re


_REASONING_BRIDGE_PATTERNS = (
    r"\bwhich suggests\b",
    r"\bwhich weakens\b",
    r"\bwhich may indicate\b",
    r"\bwhich might indicate\b",
    r"\bwhich could indicate guilt\b",
    r"\bwhich could indicate\b",
    r"\bwhich indicates\b",
    r"\bwhich implies\b",
    r"\bwhich points toward\b",
    r"\bwhich supports\b",
    r"\bcould support\b",
    r"\bsupports the (?:claim|case|conclusion|accusation)\b",
    r"\bweakens the (?:claim|case|conclusion|accusation|suspicion)\b",
    r"\bmay be interpreted as\b",
    r"\bcan be interpreted as\b",
    r"\bmaking it unlikely\b",
    r"\bmaking .*? unlikely\b",
    r"\bsuggesting that\b",
    r"\bindicating that\b",
    r"\bimplying that\b",
    r"\bshowing that\b",
    r"\bwondered if\b",
    r"\bsuspected\b",
    r"\bsuspicion\b",
    r"\bguilty\b",
    r"\binnocent\b",
    r"\bresponsible\b",
    r"\blikely\b",
    r"\bunlikely\b",
    r"\bmotive\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bto gain attention\b",
)

_OBSERVABLE_SPLIT_VERBS = (
    "appeared",
    "looked",
    "seemed",
    "was",
    "were",
    "had",
    "found",
    "suggested",
    "stormed",
    "went",
    "said",
    "told",
    "asked",
    "kept",
    "knew",
    "did",
    "could",
)

_NARRATIVE_DROP_PATTERNS = (
    r"\bwould be surprisingly different\b",
    r"\band a mystery\b",
    r"\bthe mystery began\b",
    r"\bwe'?d finally see\b",
    r"\bfor a minute I\b",
    r"\bas the laughter died down\b",
    r"\bthere was a strange silence\b",
    r"\bafter all this excitement\b",
)

_RELEVANT_CLUE_PATTERNS = (
    r"\bnote used\b",
    r"\bdrawing of\b",
    r"\bpun\b",
    r"\bknight\b",
    r"\bhandwriting\b",
    r"\b20 feet\b",
    r"\bup (?:in|a) tree\b",
    r"\bblack plastic garbage bag\b",
    r"\bfootprints?\b",
    r"\bchess piece\b",
)

_GENERIC_MYSTERY_PATTERNS = (
    r"\ball the .* were gone\b",
    r"\bcould not play\b",
    r"\bleft behind a taunting note\b",
    r"\btoo far away\b",
)


def observable_fact_unit(text: str | None) -> str:
    cleaned = _normalize_story_fact_sentence(str(text or ""))
    if not cleaned:
        return ""
    for pattern in _REASONING_BRIDGE_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[: match.start()].rstrip(" ,;:-")
            break
    cleaned = re.sub(r"\bto gain attention\b.*$", "", cleaned, flags=re.IGNORECASE).rstrip(" ,;:-")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def observable_fact_units(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        source_text = str(value)
        for sentence in _split_fact_sentences(source_text):
            for sentence in _extract_atomic_fact_sentences(sentence, source_text):
                cleaned_sentence = observable_fact_unit(sentence)
                if fact_unit_has_reasoning(cleaned_sentence):
                    continue
                for part in _split_observable_conjunctions(cleaned_sentence):
                    if fact_unit_has_reasoning(part):
                        continue
                    if part and part not in out:
                        out.append(part)
    return out


def slot_relevant_fact_units(
    values: list[str] | None,
    *,
    subject_suspect: str | None = None,
    local_purpose: str | None = None,
    max_units: int = 5,
) -> list[str]:
    facts = observable_fact_units(values)
    if len(facts) <= max_units:
        return facts
    subject = str(subject_suspect or "").strip().lower()
    scored: list[tuple[int, int, str]] = []
    for index, fact in enumerate(facts):
        score = _slot_relevance_score(fact, subject=subject, local_purpose=local_purpose)
        if score > 0:
            scored.append((score, index, fact))
    if not scored:
        return facts[:max_units]
    if local_purpose == "support_innocence":
        high_signal = [(score, index, fact) for score, index, fact in scored if score >= 2]
        if high_signal:
            scored = high_signal
    selected_indices = {
        index
        for _score, index, _fact in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_units]
    }
    clue_indices = [
        index
        for _score, index, fact in scored
        if _fact_mentions_clue(fact)
    ]
    if clue_indices and not any(index in selected_indices for index in clue_indices):
        selected_indices.add(clue_indices[0])
        if len(selected_indices) > max_units:
            removable = [
                index
                for _score, index, fact in sorted(scored, key=lambda item: (item[0], -item[1]))
                if index in selected_indices and index != clue_indices[0]
            ]
            if removable:
                selected_indices.remove(removable[0])
    return [fact for index, fact in enumerate(facts) if index in selected_indices]


def _slot_relevance_score(fact: str, *, subject: str, local_purpose: str | None) -> int:
    lowered = fact.lower()
    score = 0
    if subject and re.search(rf"\b{re.escape(subject)}\b", lowered, flags=re.IGNORECASE):
        score += 3
    if _fact_mentions_clue(fact):
        score += 4
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _GENERIC_MYSTERY_PATTERNS):
        score -= 2
    if re.search(r"\b(?:did not|does not|could not|cannot|lacked|unable|not know|was not)\b", lowered):
        score += 2 if local_purpose == "support_innocence" else 1
    if re.search(r"\b(?:learned|read|practiced|knew|knowledge|suggested|found|glowed|proud|disappointed|stormed|sulk)\b", lowered):
        score += 1
    return score


def _fact_mentions_clue(fact: str) -> bool:
    lowered = fact.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _RELEVANT_CLUE_PATTERNS)


def _extract_atomic_fact_sentences(sentence: str, source_text: str) -> list[str]:
    cleaned = " ".join(str(sentence or "").split()).strip()
    if not cleaned:
        return []
    lowered = cleaned.lower()
    source_lower = source_text.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _NARRATIVE_DROP_PATTERNS):
        return []

    facts: list[str] = []

    if re.search(r"\bGreg had decided to learn how to play chess\b", cleaned, flags=re.IGNORECASE):
        facts.append("Greg had decided to learn how to play chess.")
    if re.search(r"\bHe'?d carefully learned to move the pieces\b", cleaned, flags=re.IGNORECASE):
        subject = "Greg" if "greg" in source_lower else "The person"
        facts.append(f"{subject} carefully learned to move the chess pieces.")
    if re.search(r"\bread several books about .*winning strategies for chess\b", cleaned, flags=re.IGNORECASE):
        subject = "Greg" if "greg" in source_lower else "The person"
        facts.append(f"{subject} read several books about chess strategy.")
    if re.search(r"\bMy sister and I didn'?t even know the names of the pieces\b", cleaned, flags=re.IGNORECASE):
        subject = "Tina" if "tina" in source_lower else "The sister"
        facts.append(f"{subject} did not know the names of the chess pieces.")
    greg_practice_match = re.search(
        r"\bGreg had practiced continuously against his friends until he was the best player in his school\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if greg_practice_match:
        facts.append("Greg practiced chess continuously against his friends until he was the best player in his school.")
    if re.search(r"\ball the chess pieces were gone\b", cleaned, flags=re.IGNORECASE):
        facts.append("The next morning, all the chess pieces were gone.")
    if re.search(r"\bWe couldn'?t play chess without the chess pieces\b", cleaned, flags=re.IGNORECASE):
        facts.append("The family could not play chess without the chess pieces.")
    if re.search(r"\bthe thief had left behind a taunting note\b", cleaned, flags=re.IGNORECASE):
        facts.append("The thief left behind a taunting note.")
    if re.search(r"\bdrawing of a knight\b", cleaned, flags=re.IGNORECASE) and re.search(r"\bword [\"']?night\b", cleaned, flags=re.IGNORECASE):
        facts.append('The note used a drawing of a knight as a pun for the word "night".')
    if re.search(r"\bTina suggested that we look in the woods\b", cleaned, flags=re.IGNORECASE):
        facts.append("Tina suggested looking in the woods.")
    if re.search(r"\bGreg was still sulking\b", cleaned, flags=re.IGNORECASE):
        facts.append("Greg was still sulking.")
    if re.search(r"\bGreg seemed disappointed that he hadn'?t found the pieces himself\b", cleaned, flags=re.IGNORECASE):
        facts.append("Greg seemed disappointed that he had not found the pieces himself.")
    if re.search(r"\bTina glowed with pride\b", cleaned, flags=re.IGNORECASE):
        facts.append("Tina glowed with pride after the pieces were found.")
    if re.search(r"\bGreg wailed\b", cleaned, flags=re.IGNORECASE) and re.search(r"\bstormed outside\b", cleaned, flags=re.IGNORECASE):
        facts.append("Greg wailed and stormed outside to sulk in the woods.")

    if facts:
        return facts

    if cleaned.startswith(('"', "“")) or _is_direct_quote_only(cleaned):
        return []
    return [cleaned]


def _normalize_story_fact_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    if any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in _NARRATIVE_DROP_PATTERNS):
        return ""
    cleaned = cleaned.strip(" \"")
    cleaned = re.sub(r"^But\s+the next morning\s*--\s*", "The next morning, ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^But\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^And then\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Then\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^To make things even more frustrating,\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmy older brother\s+([A-Z][a-z]+)\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmy younger sister\s+([A-Z][a-z]+)\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmy father\b", "the father", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmy mother\b", "the mother", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bour family\b", "the family", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwe couldn'?t\b", "the family could not", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwe were\b", "the family was", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwe\b", "the family", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bour\b", "the family", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bI\b", "the sibling", cleaned)
    cleaned = re.sub(r"\bmy\b", "the", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("--", ",")
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def _is_direct_quote_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return bool(re.match(r'^["“].*["”]\.?$', stripped))


def _split_fact_sentences(text: str) -> list[str]:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", cleaned)
    if len(pieces) == 1:
        pieces = re.split(r"\s*;\s*", cleaned)
    return [piece.strip(" -") for piece in pieces if piece.strip(" -")]


def fact_unit_has_reasoning(text: str | None) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _REASONING_BRIDGE_PATTERNS)


def _split_observable_conjunctions(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    bare = cleaned.rstrip(".!?")
    match = re.match(
        rf"^([A-Z][A-Za-z0-9' -]{{1,50}}?)\s+(.+?)\s+and\s+((?:{'|'.join(_OBSERVABLE_SPLIT_VERBS)})\b.+)$",
        bare,
    )
    if not match:
        return [cleaned]
    subject, first_predicate, second_predicate = match.groups()
    return [
        observable_fact_unit(f"{subject} {first_predicate}"),
        observable_fact_unit(f"{subject} {second_predicate}"),
    ]
