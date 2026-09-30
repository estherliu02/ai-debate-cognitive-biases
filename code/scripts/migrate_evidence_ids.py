#!/usr/bin/env python3
"""One-time evidence ID migration after changing detective sentence splitting.

This migrates generated artifacts without regenerating dialogues:

1. Load each detective case's original context and old evidence bank.
2. Regenerate evidence spans with the current pySBD splitter.
3. Map old evidence IDs to new IDs by maximum character-span overlap.
4. Copy an outputs directory to a new destination while rewriting citations.

Run from the repo root, for example:

    .venv/bin/python code/scripts/migrate_evidence_ids.py \
      outputs/<old_run_dir> \
      --output-dir outputs/<old_run_dir>_evidence
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from convert_true_detective import split_sentence_spans


INLINE_CITATION_RE = re.compile(r"\[E(\d+)\]")
EXACT_EID_RE = re.compile(r"^E(\d+)$")
ADJACENT_DUPLICATE_CITATION_RE = re.compile(r"\[E(\d+)\](?:\s*,?\s*\[E\1\])+")
TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
FALSE_EVIDENCE_ID_KEYS = {
    "evidence_bank",
    "evidence_count",
}


@dataclass(frozen=True)
class EvidenceItem:
    eid: int
    text: str
    start_char: int | None
    end_char: int | None


@dataclass
class MappingRecord:
    old_id: int
    old_text: str
    old_span: tuple[int | None, int | None]
    candidate_new_ids: list[int]
    candidate_new_texts: list[str]
    overlap_score: float
    mapping_status: str

    @property
    def applied_new_ids(self) -> list[int]:
        if self.mapping_status in {"unmatched", "low_confidence", "old_span_unmatched"}:
            return []
        return self.candidate_new_ids


@dataclass
class CaseEvidenceMapping:
    case_id: str
    context: str
    old_items: list[EvidenceItem]
    new_items: list[EvidenceItem]
    records: list[MappingRecord]
    old_to_new: dict[int, list[int]]

    @property
    def new_evidence_bank(self) -> list[dict[str, Any]]:
        return [
            {
                "index": item.eid,
                "text": item.text,
                "start_char": item.start_char,
                "end_char": item.end_char,
            }
            for item in self.new_items
        ]

    @property
    def status_counts(self) -> Counter:
        return Counter(record.mapping_status for record in self.records)


@dataclass
class RewriteStats:
    inline_citation_replacements: int = 0
    exact_json_eid_replacements: int = 0
    evidence_id_list_replacements: int = 0
    evidence_bank_rewrites: int = 0
    evidence_count_updates: int = 0
    duplicate_citations_collapsed: int = 0
    unresolved_citations: int = 0

    def total_citation_replacements(self) -> int:
        return (
            self.inline_citation_replacements
            + self.exact_json_eid_replacements
            + self.evidence_id_list_replacements
        )

    def add(self, other: "RewriteStats") -> None:
        self.inline_citation_replacements += other.inline_citation_replacements
        self.exact_json_eid_replacements += other.exact_json_eid_replacements
        self.evidence_id_list_replacements += other.evidence_id_list_replacements
        self.evidence_bank_rewrites += other.evidence_bank_rewrites
        self.evidence_count_updates += other.evidence_count_updates
        self.duplicate_citations_collapsed += other.duplicate_citations_collapsed
        self.unresolved_citations += other.unresolved_citations


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def normalized_with_char_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    char_map: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if not in_whitespace and normalized_chars:
                normalized_chars.append(" ")
                char_map.append(index)
            in_whitespace = True
            continue
        normalized_chars.append(char)
        char_map.append(index)
        in_whitespace = False
    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        char_map.pop()
    return "".join(normalized_chars), char_map


def find_normalized_span(
    source: str,
    snippet: str,
    *,
    start_after: int = 0,
) -> tuple[int | None, int | None]:
    normalized_source, char_map = normalized_with_char_map(source)
    normalized_snippet = normalize_text(snippet)
    if not normalized_snippet:
        return None, None

    normalized_start_after = 0
    for normalized_index, source_index in enumerate(char_map):
        if source_index >= start_after:
            normalized_start_after = normalized_index
            break

    found = normalized_source.find(normalized_snippet, normalized_start_after)
    if found == -1:
        found = normalized_source.find(normalized_snippet)
    if found == -1:
        return None, None

    start = char_map[found]
    end = char_map[found + len(normalized_snippet) - 1] + 1
    return start, end


def recover_old_evidence_items(context: str, old_bank: list[dict[str, Any]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    cursor = 0
    for raw_item in old_bank:
        eid = int(raw_item["index"])
        text = str(raw_item["text"])
        start, end = find_normalized_span(context, text, start_after=cursor)
        if end is not None:
            cursor = end
        items.append(EvidenceItem(eid=eid, text=normalize_text(text), start_char=start, end_char=end))
    return items


def build_new_evidence_items(context: str) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    cursor = 0
    for eid, span in enumerate(split_sentence_spans(context), start=1):
        start = span.start_char
        end = span.end_char
        if start is None or end is None:
            start, end = find_normalized_span(context, span.text, start_after=cursor)
        if end is not None:
            cursor = end
        items.append(
            EvidenceItem(
                eid=eid,
                text=normalize_text(span.text),
                start_char=start,
                end_char=end,
            )
        )
    return items


def overlap_len(
    old_span: tuple[int | None, int | None],
    new_span: tuple[int | None, int | None],
) -> int:
    old_start, old_end = old_span
    new_start, new_end = new_span
    if old_start is None or old_end is None or new_start is None or new_end is None:
        return 0
    return max(0, min(old_end, new_end) - max(old_start, new_start))


def build_case_mapping(
    case_id: str,
    case_data: dict[str, Any],
    *,
    min_overlap_score: float,
) -> CaseEvidenceMapping:
    context = str(case_data["context"])
    old_items = recover_old_evidence_items(context, list(case_data["evidence_bank"]))
    new_items = build_new_evidence_items(context)

    records: list[MappingRecord] = []
    for old_item in old_items:
        old_span = (old_item.start_char, old_item.end_char)
        if old_item.start_char is None or old_item.end_char is None:
            records.append(
                MappingRecord(
                    old_id=old_item.eid,
                    old_text=old_item.text,
                    old_span=old_span,
                    candidate_new_ids=[],
                    candidate_new_texts=[],
                    overlap_score=0.0,
                    mapping_status="old_span_unmatched",
                )
            )
            continue

        overlaps = [
            (new_item, overlap_len(old_span, (new_item.start_char, new_item.end_char)))
            for new_item in new_items
        ]
        overlapping_items = [(item, size) for item, size in overlaps if size > 0]
        overlapping_items.sort(key=lambda pair: (pair[0].start_char or -1, pair[0].eid))
        old_length = max(1, old_item.end_char - old_item.start_char)
        overlap_score = sum(size for _, size in overlapping_items) / old_length

        if not overlapping_items:
            status = "unmatched"
        elif overlap_score < min_overlap_score:
            status = "low_confidence"
        else:
            status = "pending"

        records.append(
            MappingRecord(
                old_id=old_item.eid,
                old_text=old_item.text,
                old_span=old_span,
                candidate_new_ids=[item.eid for item, _ in overlapping_items],
                candidate_new_texts=[item.text for item, _ in overlapping_items],
                overlap_score=overlap_score,
                mapping_status=status,
            )
        )

    single_new_counts = Counter(
        record.candidate_new_ids[0]
        for record in records
        if record.mapping_status == "pending" and len(record.candidate_new_ids) == 1
    )
    old_to_new: dict[int, list[int]] = {}
    for record in records:
        if record.mapping_status != "pending":
            continue
        if len(record.candidate_new_ids) > 1:
            record.mapping_status = "one_old_to_many_new"
        elif single_new_counts[record.candidate_new_ids[0]] > 1:
            record.mapping_status = "many_old_to_one_new"
        else:
            record.mapping_status = "one_to_one"
        old_to_new[record.old_id] = record.candidate_new_ids

    return CaseEvidenceMapping(
        case_id=case_id,
        context=context,
        old_items=old_items,
        new_items=new_items,
        records=records,
        old_to_new=old_to_new,
    )


def load_case_mappings(
    cases_dir: Path,
    *,
    min_overlap_score: float,
    selected_cases: set[str] | None = None,
) -> dict[str, CaseEvidenceMapping]:
    mappings: dict[str, CaseEvidenceMapping] = {}
    for path in sorted(cases_dir.glob("*.json")):
        if path.name == "_index.json":
            continue
        case_data = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(case_data["case_id"])
        if selected_cases is not None and case_id not in selected_cases:
            continue
        mappings[case_id] = build_case_mapping(
            case_id,
            case_data,
            min_overlap_score=min_overlap_score,
        )
    return mappings


def write_case_reports(mappings: dict[str, CaseEvidenceMapping], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    banks_dir = reports_dir / "new_evidence_banks"
    banks_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "old_id",
        "old_text",
        "old_span",
        "new_id",
        "new_ids",
        "new_text",
        "overlap_score",
        "mapping_status",
    ]
    for case_id, mapping in sorted(mappings.items()):
        report_path = reports_dir / f"{case_id}_evidence_id_mapping.csv"
        with report_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in mapping.records:
                old_start, old_end = record.old_span
                writer.writerow(
                    {
                        "old_id": record.old_id,
                        "old_text": record.old_text,
                        "old_span": format_span(old_start, old_end),
                        "new_id": record.candidate_new_ids[0] if len(record.candidate_new_ids) == 1 else "",
                        "new_ids": ";".join(str(eid) for eid in record.candidate_new_ids),
                        "new_text": " || ".join(record.candidate_new_texts),
                        "overlap_score": f"{record.overlap_score:.4f}",
                        "mapping_status": record.mapping_status,
                    }
                )
        (banks_dir / f"{case_id}_new_evidence_bank.json").write_text(
            json.dumps(mapping.new_evidence_bank, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def format_span(start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return ""
    return f"{start}:{end}"


def is_evidence_id_key(key: str | None) -> bool:
    if key is None:
        return False
    lowered = key.lower()
    if lowered in FALSE_EVIDENCE_ID_KEYS:
        return False
    return "evidence" in lowered or "citation" in lowered or lowered.endswith("_eid") or lowered.endswith("_eids")


def collapse_duplicate_adjacent_citations(text: str) -> tuple[str, int]:
    total = 0
    while True:
        text, count = ADJACENT_DUPLICATE_CITATION_RE.subn(
            lambda match: f"[E{match.group(1)}]",
            text,
        )
        total += count
        if count == 0:
            return text, total


def replace_inline_citations(
    text: str,
    mapping: CaseEvidenceMapping,
    stats: RewriteStats,
) -> str:
    def replace(match: re.Match[str]) -> str:
        old_id = int(match.group(1))
        new_ids = mapping.old_to_new.get(old_id)
        if not new_ids:
            stats.unresolved_citations += 1
            return match.group(0)
        stats.inline_citation_replacements += 1
        return "".join(f"[E{new_id}]" for new_id in new_ids)

    replaced = INLINE_CITATION_RE.sub(replace, text)
    replaced, collapsed = collapse_duplicate_adjacent_citations(replaced)
    stats.duplicate_citations_collapsed += collapsed
    return replaced


def map_eid_number(old_id: int, mapping: CaseEvidenceMapping) -> list[int] | None:
    return mapping.old_to_new.get(old_id)


def collapse_adjacent_duplicates(items: list[Any]) -> list[Any]:
    collapsed: list[Any] = []
    for item in items:
        if collapsed and collapsed[-1] == item:
            continue
        collapsed.append(item)
    return collapsed


def transform_evidence_id_value(
    value: Any,
    mapping: CaseEvidenceMapping,
    stats: RewriteStats,
    *,
    list_context: bool,
) -> list[Any] | Any:
    if isinstance(value, int):
        new_ids = map_eid_number(value, mapping)
        if not new_ids:
            return value
        stats.evidence_id_list_replacements += 1
        if list_context or len(new_ids) > 1:
            return new_ids
        return new_ids[0]

    if isinstance(value, str):
        exact = EXACT_EID_RE.match(value)
        if exact:
            new_ids = map_eid_number(int(exact.group(1)), mapping)
            if not new_ids:
                return value
            stats.exact_json_eid_replacements += 1
            formatted = [f"E{new_id}" for new_id in new_ids]
            if list_context or len(formatted) > 1:
                return formatted
            return formatted[0]

    return value


def transform_json_value(
    value: Any,
    mapping: CaseEvidenceMapping,
    stats: RewriteStats,
    *,
    parent_key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        transformed: dict[str, Any] = {}
        for key, child in value.items():
            if key == "evidence_bank" and isinstance(child, list):
                transformed[key] = mapping.new_evidence_bank
                stats.evidence_bank_rewrites += 1
            elif key == "evidence_count" and isinstance(child, int):
                transformed[key] = len(mapping.new_items)
                stats.evidence_count_updates += 1
            elif is_evidence_id_key(key):
                transformed[key] = transform_json_value(child, mapping, stats, parent_key=key)
            else:
                transformed[key] = transform_json_value(child, mapping, stats, parent_key=key)
        return transformed

    if isinstance(value, list):
        if is_evidence_id_key(parent_key):
            transformed_list: list[Any] = []
            for item in value:
                mapped = transform_evidence_id_value(item, mapping, stats, list_context=True)
                if isinstance(mapped, list):
                    transformed_list.extend(mapped)
                else:
                    transformed_list.append(transform_json_value(mapped, mapping, stats, parent_key=parent_key))
            return collapse_adjacent_duplicates(transformed_list)
        return [transform_json_value(item, mapping, stats, parent_key=parent_key) for item in value]

    if isinstance(value, str):
        exact_mapped = transform_evidence_id_value(value, mapping, stats, list_context=False)
        if exact_mapped != value:
            return exact_mapped
        return replace_inline_citations(value, mapping, stats)

    if isinstance(value, int) and is_evidence_id_key(parent_key):
        return transform_evidence_id_value(value, mapping, stats, list_context=False)

    return value


def collect_case_ids_from_json(value: Any, known_case_ids: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        case_id = value.get("case_id")
        if isinstance(case_id, str) and case_id in known_case_ids:
            found.add(case_id)
        for child in value.values():
            found.update(collect_case_ids_from_json(child, known_case_ids))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_case_ids_from_json(child, known_case_ids))
    return found


def infer_case_id_from_path_or_text(
    rel_path: Path,
    text: str,
    known_case_ids: set[str],
) -> str | None:
    path_matches = [case_id for case_id in known_case_ids if case_id in str(rel_path)]
    if len(path_matches) == 1:
        return path_matches[0]

    text_matches = [case_id for case_id in known_case_ids if case_id in text]
    if len(text_matches) == 1:
        return text_matches[0]
    return None


def transform_json_file(
    source: Path,
    destination: Path,
    rel_path: Path,
    mappings: dict[str, CaseEvidenceMapping],
    skipped_files: list[dict[str, str]],
) -> tuple[RewriteStats, bool]:
    stats = RewriteStats()
    text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return transform_text_file(source, destination, rel_path, mappings, skipped_files)

    case_ids = collect_case_ids_from_json(payload, set(mappings))
    if len(case_ids) != 1:
        inferred = infer_case_id_from_path_or_text(rel_path, text, set(mappings))
        if inferred is not None:
            case_ids = {inferred}

    if len(case_ids) != 1:
        has_citation = bool(INLINE_CITATION_RE.search(text) or EXACT_EID_RE.search(text))
        if has_citation:
            skipped_files.append(
                {
                    "path": str(rel_path),
                    "reason": f"ambiguous_case_id:{sorted(case_ids)}",
                }
            )
        destination.write_text(text, encoding="utf-8")
        return stats, False

    mapping = mappings[next(iter(case_ids))]
    transformed = transform_json_value(payload, mapping, stats)
    destination.write_text(json.dumps(transformed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats, True


def transform_text_file(
    source: Path,
    destination: Path,
    rel_path: Path,
    mappings: dict[str, CaseEvidenceMapping],
    skipped_files: list[dict[str, str]],
) -> tuple[RewriteStats, bool]:
    stats = RewriteStats()
    text = source.read_text(encoding="utf-8")
    case_id = infer_case_id_from_path_or_text(rel_path, text, set(mappings))
    if case_id is None:
        if INLINE_CITATION_RE.search(text):
            skipped_files.append(
                {
                    "path": str(rel_path),
                    "reason": "ambiguous_or_missing_case_id",
                }
            )
        destination.write_text(text, encoding="utf-8")
        return stats, False

    transformed = replace_inline_citations(text, mappings[case_id], stats)
    destination.write_text(transformed, encoding="utf-8")
    return stats, True


def migrate_output_tree(
    input_dir: Path,
    output_dir: Path,
    mappings: dict[str, CaseEvidenceMapping],
) -> tuple[dict[str, RewriteStats], list[dict[str, str]], int]:
    stats_by_suffix: dict[str, RewriteStats] = defaultdict(RewriteStats)
    skipped_files: list[dict[str, str]] = []
    transformed_file_count = 0

    for source in sorted(input_dir.rglob("*")):
        rel_path = source.relative_to(input_dir)
        destination = output_dir / rel_path
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            shutil.copy2(source, destination)
            continue

        try:
            if suffix == ".json":
                file_stats, transformed = transform_json_file(
                    source,
                    destination,
                    rel_path,
                    mappings,
                    skipped_files,
                )
            else:
                file_stats, transformed = transform_text_file(
                    source,
                    destination,
                    rel_path,
                    mappings,
                    skipped_files,
                )
        except UnicodeDecodeError:
            shutil.copy2(source, destination)
            continue

        stats_by_suffix[suffix or "[no_suffix]"].add(file_stats)
        if transformed:
            transformed_file_count += 1

    return stats_by_suffix, skipped_files, transformed_file_count


def mapping_summary(mappings: dict[str, CaseEvidenceMapping]) -> dict[str, Any]:
    overall = Counter()
    case_summaries: dict[str, dict[str, int]] = {}
    for case_id, mapping in sorted(mappings.items()):
        counts = mapping.status_counts
        case_summary = {
            "old_evidence_items": len(mapping.old_items),
            "mapped_one_to_one": counts["one_to_one"],
            "mapped_many_to_one": counts["many_old_to_one_new"],
            "mapped_one_to_many": counts["one_old_to_many_new"],
            "low_confidence": counts["low_confidence"],
            "unmatched": counts["unmatched"] + counts["old_span_unmatched"],
            "new_evidence_items": len(mapping.new_items),
        }
        case_summaries[case_id] = case_summary
        overall.update(case_summary)
    return {
        "overall": dict(overall),
        "cases": case_summaries,
    }


def write_summary_log(
    output_dir: Path,
    input_dir: Path,
    cases_dir: Path,
    mappings: dict[str, CaseEvidenceMapping],
    stats_by_suffix: dict[str, RewriteStats],
    skipped_files: list[dict[str, str]],
    transformed_file_count: int,
) -> None:
    summary = mapping_summary(mappings)
    overall = summary["overall"]
    lines = [
        "Evidence ID migration summary",
        f"input_dir: {input_dir}",
        f"output_dir: {output_dir}",
        f"cases_dir: {cases_dir}",
        f"cases_processed: {len(mappings)}",
        f"files_transformed: {transformed_file_count}",
        "",
        "Mapping statistics",
        f"old_evidence_items: {overall.get('old_evidence_items', 0)}",
        f"mapped_one_to_one: {overall.get('mapped_one_to_one', 0)}",
        f"mapped_many_to_one: {overall.get('mapped_many_to_one', 0)}",
        f"mapped_one_to_many: {overall.get('mapped_one_to_many', 0)}",
        f"low_confidence: {overall.get('low_confidence', 0)}",
        f"unmatched: {overall.get('unmatched', 0)}",
        f"new_evidence_items: {overall.get('new_evidence_items', 0)}",
        "",
        "Citation replacements per file type",
    ]
    replacement_summary: dict[str, dict[str, int]] = {}
    for suffix, stats in sorted(stats_by_suffix.items()):
        replacement_summary[suffix] = {
            "total_citation_replacements": stats.total_citation_replacements(),
            "inline_citation_replacements": stats.inline_citation_replacements,
            "exact_json_eid_replacements": stats.exact_json_eid_replacements,
            "evidence_id_list_replacements": stats.evidence_id_list_replacements,
            "evidence_bank_rewrites": stats.evidence_bank_rewrites,
            "evidence_count_updates": stats.evidence_count_updates,
            "duplicate_citations_collapsed": stats.duplicate_citations_collapsed,
            "unresolved_citations": stats.unresolved_citations,
        }
        lines.append(
            f"{suffix}: total={stats.total_citation_replacements()}, "
            f"inline={stats.inline_citation_replacements}, "
            f"exact_json={stats.exact_json_eid_replacements}, "
            f"lists={stats.evidence_id_list_replacements}, "
            f"evidence_bank={stats.evidence_bank_rewrites}, "
            f"evidence_count={stats.evidence_count_updates}, "
            f"collapsed_duplicates={stats.duplicate_citations_collapsed}, "
            f"unresolved={stats.unresolved_citations}"
        )

    lines.extend(["", "Skipped ambiguous citation files"])
    if skipped_files:
        for item in skipped_files:
            lines.append(f"{item['path']}: {item['reason']}")
    else:
        lines.append("none")

    (output_dir / "evidence_id_migration_summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "evidence_id_migration_summary.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "cases_dir": str(cases_dir),
                "cases_processed": len(mappings),
                "files_transformed": transformed_file_count,
                "mapping": summary,
                "citation_replacements_by_file_type": replacement_summary,
                "skipped_files": skipped_files,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Existing generated outputs directory to migrate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <input_dir>_evidence_id_migrated.",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=ROOT / "data" / "detective_cases",
        help="Directory containing old detective case JSON files.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Restrict migration mapping to one case_id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--min-overlap-score",
        type=float,
        default=0.5,
        help="Minimum old-span coverage required to apply an old-ID to new-ID mapping.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the destination directory first if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir.with_name(f"{input_dir.name}_evidence_id_migrated")).resolve()
    cases_dir = args.cases_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")
    if not cases_dir.exists() or not cases_dir.is_dir():
        raise SystemExit(f"Cases directory does not exist or is not a directory: {cases_dir}")
    if output_dir == input_dir:
        raise SystemExit("Refusing to overwrite the input directory.")
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output directory already exists: {output_dir}. Use --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    selected_cases = set(args.cases) if args.cases else None
    mappings = load_case_mappings(
        cases_dir,
        min_overlap_score=args.min_overlap_score,
        selected_cases=selected_cases,
    )
    if selected_cases:
        missing = selected_cases - set(mappings)
        if missing:
            raise SystemExit(f"Requested case_id(s) were not found: {sorted(missing)}")

    reports_dir = output_dir / "evidence_id_migration_reports"
    write_case_reports(mappings, reports_dir)
    stats_by_suffix, skipped_files, transformed_file_count = migrate_output_tree(input_dir, output_dir, mappings)
    write_summary_log(
        output_dir,
        input_dir,
        cases_dir,
        mappings,
        stats_by_suffix,
        skipped_files,
        transformed_file_count,
    )

    print(f"Migrated outputs written to: {output_dir}")
    print(f"Case mapping reports written to: {reports_dir}")
    print(f"Summary log: {output_dir / 'evidence_id_migration_summary.log'}")


if __name__ == "__main__":
    main()
