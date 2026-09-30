from __future__ import annotations

import re


_EVIDENCE_REF = r"(?:[Ee]vidence\s+)?[Ee]?\d+"
_EVIDENCE_REF_LIST = rf"{_EVIDENCE_REF}(?:\s*(?:,|and|&)\s*{_EVIDENCE_REF})*"
_SCAFFOLD_VERB = (
    r"show(?:s)?|suggest(?:s)?|indicate(?:s)?|note(?:s)?|state(?:s)?|"
    r"confirm(?:s)?|describe(?:s)?|establish(?:es)?|introduce(?:s)?|has|have"
)
_LEADING_SCAFFOLD_RE = re.compile(
    rf"^\s*(?:{_EVIDENCE_REF_LIST})\s+(?:{_SCAFFOLD_VERB})\s+(?:that\s+)?(?P<claim>.+)$",
    re.IGNORECASE,
)
_MID_SCAFFOLD_RE = re.compile(
    rf"\b(?P<connector>and|while|but)\s+(?:{_EVIDENCE_REF_LIST})\s+"
    rf"(?:{_SCAFFOLD_VERB})\s+(?:that\s+)?",
    re.IGNORECASE,
)
_PARENTHETICAL_CITATION_RE = re.compile(
    rf"\s*\((?:{_EVIDENCE_REF_LIST})\)",
    re.IGNORECASE,
)
_BRACKETED_CITATION_RE = re.compile(r"\s*\[[Ee]\d+\]")


def normalize_anchor_reason(reason_statement: str | None) -> str:
    """Remove evidence-ID scaffolding from a reason claim without changing its facts."""

    text = " ".join(str(reason_statement or "").split()).strip()
    if not text:
        return ""

    text = _BRACKETED_CITATION_RE.sub("", text)
    text = _PARENTHETICAL_CITATION_RE.sub("", text)

    leading_match = _LEADING_SCAFFOLD_RE.match(text)
    if leading_match:
        text = leading_match.group("claim").strip()

    text = _MID_SCAFFOLD_RE.sub(lambda match: f"{match.group('connector')} ", text)

    text = re.sub(
        r"^([A-Z][^,.;:]{0,120}?)\s+suggesting\s+that\b",
        r"\1 suggested that",
        text,
    )
    text = re.sub(
        r"^([A-Z][^,.;:]{0,120}?)\s+claiming\s+that\b",
        r"\1 claimed that",
        text,
    )
    text = re.sub(
        r"\b(his|her|their|its|the)\s+([^,.;:]{1,80}?)\s+claiming\s+that\b",
        r"\1 \2 claimed that",
        text,
    )

    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-")
    return text
