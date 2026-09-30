#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "code" / "configs" / "detective_cases.py").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
REPO_PATH_ANCHORS = {
    "experiment",
    "llm_participant",
    "outputs",
}

from configs.detective_cases import (  # noqa: E402
    GROUND_TRUTH_SIDE_ORDER_CHOICES,
    load_case,
)


CASE_SPECS = [
    "our-quarterback-is-missing",
    "the-diamond-necklace",
    "the-missing-briefcase",
    "the-mystery-of-the-leprechaun-s-trophy",
]
EXPECTED_CASE_COUNT = len(CASE_SPECS)


BIAS_SUMMARY_FIELDS = [
    "motion",
    "trait",
    "variant_a",
    "variant_b",
    "gt_order",
    "overall_passing_result",
    "verbosity_passing_result",
    "llm_as_judge_passing_result",
    "expected_biased_side",
    "expected_bias",
    "num_passing_judges",
    "gemini_passed",
    "gemini_predicted_biased_side",
    "gemini_predicted_bias",
    "gemini_reasoning",
    "claude_passed",
    "claude_predicted_biased_side",
    "claude_predicted_bias",
    "claude_reasoning",
    "openai_passed",
    "openai_predicted_biased_side",
    "openai_predicted_bias",
    "openai_reasoning",
    "failure_mode",
    "eval_file",
    "dialogue_file",
]


RunKey = tuple[str, str, str, str, str]
ComboKey = tuple[str, str, str, str]


@dataclass
class RunAvailability:
    case_id: str
    trait: str
    variant_a: str
    variant_b: str
    gt_order: str
    dialogue_files: set[str] = field(default_factory=set)
    bias_eval_files: set[str] = field(default_factory=set)
    passing_bias_eval_files: set[str] = field(default_factory=set)
    rar_eval_files: set[str] = field(default_factory=set)
    passing_rar_eval_files: set[str] = field(default_factory=set)

    @property
    def has_dialogue(self) -> bool:
        return bool(self.dialogue_files)

    @property
    def bias_passed(self) -> bool:
        return bool(self.passing_bias_eval_files)

    @property
    def rar_passed(self) -> bool:
        return bool(self.passing_rar_eval_files)

    @property
    def available(self) -> bool:
        return self.has_dialogue and self.bias_passed and self.rar_passed


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def resolve_output_root(reference_output: str) -> Path:
    path = Path(reference_output)
    if path.is_absolute():
        return resolve_repo_path(path)
    if path.parts and path.parts[0] == "outputs":
        return REPO_ROOT / path
    return REPO_ROOT / "outputs" / path


def repo_relative_candidate(path: Path) -> Path | None:
    for index, part in enumerate(path.parts):
        if part in REPO_PATH_ANCHORS:
            return Path(*path.parts[index:])
    return None


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        return REPO_ROOT / resolved
    if resolved.is_relative_to(REPO_ROOT):
        return resolved
    candidate = repo_relative_candidate(resolved)
    if candidate is not None:
        return REPO_ROOT / candidate
    return resolved


def repo_display_path(path: str | Path) -> str:
    resolved = Path(path)
    if resolved.is_absolute():
        if resolved.is_relative_to(REPO_ROOT):
            return resolved.relative_to(REPO_ROOT).as_posix()
        candidate = repo_relative_candidate(resolved)
        if candidate is not None:
            return candidate.as_posix()
    return resolved.as_posix()


@cache
def case_by_topic() -> dict[str, str]:
    return {load_case(case_id)["topic"]: case_id for case_id in CASE_SPECS}


def infer_case_id(payload: dict[str, Any]) -> str | None:
    return (
        payload.get("case_id")
        or payload.get("case_meta", {}).get("case_id")
        or case_by_topic().get(payload.get("topic"))
    )


def infer_gt_order(payload: dict[str, Any]) -> str:
    explicit = payload.get("ground_truth_side_order") or payload.get("case_meta", {}).get("ground_truth_side_order")
    if explicit in GROUND_TRUTH_SIDE_ORDER_CHOICES:
        return explicit
    supporting_speaker = payload.get("ground_truth_supporting_speaker") or payload.get("case_meta", {}).get("ground_truth_supporting_speaker")
    if supporting_speaker == "B":
        return "gt_second"
    return "gt_first"


def metadata_key(payload: dict[str, Any]) -> RunKey | None:
    case_id = infer_case_id(payload)
    trait = payload.get("trait_name")
    variant_a = payload.get("variant_name_a") or payload.get("variant_a")
    variant_b = payload.get("variant_name_b") or payload.get("variant_b")
    gt_order = infer_gt_order(payload)
    if not all([case_id, trait, variant_a, variant_b, gt_order]):
        return None
    return (case_id, trait, variant_a, variant_b, gt_order)


def run_prefix(path: Path, suffix: str) -> str | None:
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else None


def dialogue_run_prefix(path: Path) -> str | None:
    return run_prefix(path, "_accepted_dialogue") or run_prefix(path, "_dialogue")


def eval_passed(payload: dict[str, Any]) -> bool:
    if payload.get("trait_name") == "verbosity_bias":
        return bool(payload.get("final_bias_eval_passed")) and bool(payload.get("llm_judge_passed"))
    if payload.get("final_bias_eval_passed") is not None:
        return bool(payload["final_bias_eval_passed"])
    return bool(payload.get("passed"))


def rar_passed(payload: dict[str, Any]) -> bool:
    return bool(payload.get("passed"))


def get_record(records: dict[RunKey, RunAvailability], key: RunKey) -> RunAvailability:
    if key not in records:
        records[key] = RunAvailability(
            case_id=key[0],
            trait=key[1],
            variant_a=key[2],
            variant_b=key[3],
            gt_order=key[4],
        )
    return records[key]


def collect_records(output_root: Path) -> dict[RunKey, RunAvailability]:
    records: dict[RunKey, RunAvailability] = {}

    for subdir, passed_func, all_attr, pass_attr in [
        ("dialogues", None, "dialogue_files", None),
        ("accepted", None, "dialogue_files", None),
        ("evals", eval_passed, "bias_eval_files", "passing_bias_eval_files"),
        ("rar_evals", rar_passed, "rar_eval_files", "passing_rar_eval_files"),
    ]:
        search_dir = output_root / subdir
        if not search_dir.exists():
            continue
        for path in sorted(search_dir.glob("*.json")):
            payload = load_json(path)
            if not payload:
                continue
            key = metadata_key(payload)
            if key is None:
                continue
            record = get_record(records, key)
            getattr(record, all_attr).add(path.name)
            if passed_func is not None and pass_attr is not None and passed_func(payload):
                getattr(record, pass_attr).add(path.name)

    return records


def join_values(values: list[str] | set[str]) -> str:
    return ";".join(sorted(values))


def build_detail_rows(records: dict[RunKey, RunAvailability]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records.values():
        rows.append(
            {
                "case_id": record.case_id,
                "trait": record.trait,
                "variant_a": record.variant_a,
                "variant_b": record.variant_b,
                "gt_order": record.gt_order,
                "available": int(record.available),
                "has_dialogue": int(record.has_dialogue),
                "bias_passed": int(record.bias_passed),
                "rar_passed": int(record.rar_passed),
                "dialogue_count": len(record.dialogue_files),
                "bias_eval_count": len(record.bias_eval_files),
                "passing_bias_eval_count": len(record.passing_bias_eval_files),
                "rar_eval_count": len(record.rar_eval_files),
                "passing_rar_eval_count": len(record.passing_rar_eval_files),
                "dialogue_files": join_values(record.dialogue_files),
                "passing_bias_eval_files": join_values(record.passing_bias_eval_files),
                "passing_rar_eval_files": join_values(record.passing_rar_eval_files),
            }
        )
    return sorted(rows, key=lambda row: (row["trait"], row["variant_a"], row["variant_b"], row["gt_order"], row["case_id"]))


def build_aggregate_rows(records: dict[RunKey, RunAvailability]) -> list[dict[str, Any]]:
    grouped: dict[ComboKey, list[RunAvailability]] = defaultdict(list)
    for record in records.values():
        grouped[(record.trait, record.variant_a, record.variant_b, record.gt_order)].append(record)

    rows: list[dict[str, Any]] = []
    for (trait, variant_a, variant_b, gt_order), items in grouped.items():
        observed_cases = {item.case_id for item in items if item.has_dialogue}
        observed_bias_eval_cases = {item.case_id for item in items if item.bias_eval_files}
        observed_rar_eval_cases = {item.case_id for item in items if item.rar_eval_files}
        passing_cases = {item.case_id for item in items if item.available}
        bias_passed_cases = {item.case_id for item in items if item.bias_passed}
        rar_passed_cases = {item.case_id for item in items if item.rar_passed}
        missing_dialogue_cases = set(CASE_SPECS) - observed_cases
        missing_or_failing_cases = set(CASE_SPECS) - passing_cases

        rows.append(
            {
                "trait": trait,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "gt_order": gt_order,
                "passing_data": len(passing_cases),
                "max_data": EXPECTED_CASE_COUNT,
                "observed_cases": len(observed_cases),
                "observed_bias_eval_cases": len(observed_bias_eval_cases),
                "observed_rar_eval_cases": len(observed_rar_eval_cases),
                "bias_passed_cases": len(bias_passed_cases),
                "rar_passed_cases": len(rar_passed_cases),
                "missing_dialogue_cases": join_values(missing_dialogue_cases),
                "passing_case_ids": join_values(passing_cases),
                "missing_or_failing_case_ids": join_values(missing_or_failing_cases),
            }
        )

    return sorted(rows, key=lambda row: (row["trait"], row["variant_a"], row["variant_b"], row["gt_order"]))


def build_dialogue_indices(
    dialogue_paths: list[Path],
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], dict[RunKey, list[tuple[Path, dict[str, Any]]]]]:
    by_prefix: dict[str, tuple[Path, dict[str, Any]]] = {}
    by_metadata: dict[RunKey, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(dialogue_paths, key=lambda p: p.stat().st_mtime, reverse=True):
        payload = load_json(path)
        if not payload:
            continue
        prefix = dialogue_run_prefix(path)
        if prefix and prefix not in by_prefix:
            by_prefix[prefix] = (path, payload)
        key = metadata_key(payload)
        if key:
            by_metadata.setdefault(key, []).append((path, payload))
    return by_prefix, by_metadata


def find_dialogue_for_eval(
    eval_path: Path,
    eval_payload: dict[str, Any],
    dialogue_by_prefix: dict[str, tuple[Path, dict[str, Any]]],
    dialogues_by_metadata: dict[RunKey, list[tuple[Path, dict[str, Any]]]],
) -> tuple[Path | None, dict[str, Any] | None]:
    prefix = run_prefix(eval_path, "_eval")
    if prefix and prefix in dialogue_by_prefix:
        return dialogue_by_prefix[prefix]

    key = metadata_key(eval_payload)
    if key:
        matches = dialogues_by_metadata.get(key) or []
        if matches:
            return matches[0]
    return None, None


def bool_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return str(value)


def classify_error_modes(eval_payload: dict[str, Any] | None) -> str:
    if not eval_payload:
        return "missing_eval"
    if eval_passed(eval_payload):
        return "pass"

    modes: list[str] = []
    raw_modes = set(eval_payload.get("failure_modes") or [])
    if eval_payload.get("computational_verbosity_passed") is False or "computational_verbosity_failed" in raw_modes:
        modes.append("verbosity_failed")
    if eval_payload.get("side_selection_correct") is False or raw_modes.intersection(
        {"biased_side_not_identified", "wrong_biased_speaker_selected"}
    ):
        modes.append("side_miss")
    if eval_payload.get("bias_type_correct") is False or raw_modes.intersection(
        {"bias_type_not_identified", "wrong_bias_type_selected"}
    ):
        modes.append("bias_categorization_miss")
    if eval_payload.get("llm_judge_passed") is False and not any(
        mode in modes for mode in {"side_miss", "bias_categorization_miss"}
    ):
        modes.append("llm_judge_failed")
    if not modes and raw_modes:
        modes.extend(sorted(raw_modes))
    return "; ".join(modes) if modes else "unknown_failure"


def format_name(path: Path | None) -> str:
    return path.name if path else ""


def judge_count_text(eval_payload: dict[str, Any] | None) -> str:
    if not eval_payload:
        return ""
    match_count = eval_payload.get("judge_match_count")
    if match_count is None:
        return ""
    return str(match_count)


def normalize_model_family(model_name: str | None) -> str | None:
    normalized = (model_name or "").lower()
    if "gemini" in normalized or "google" in normalized:
        return "gemini"
    if "claude" in normalized or "anthropic" in normalized:
        return "claude"
    if "openai" in normalized or normalized.startswith("o3") or "/o3" in normalized:
        return "openai"
    return None


def model_results_by_family(eval_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not eval_payload:
        return results
    for item in eval_payload.get("per_model_results") or []:
        if not isinstance(item, dict):
            continue
        family = normalize_model_family(item.get("model"))
        if family and family not in results:
            results[family] = item
    return results


def model_result_fields(model_result: dict[str, Any] | None, prefix: str) -> dict[str, str]:
    if not model_result:
        return {
            f"{prefix}_passed": "",
            f"{prefix}_predicted_biased_side": "",
            f"{prefix}_predicted_bias": "",
            f"{prefix}_reasoning": "",
        }
    return {
        f"{prefix}_passed": bool_text(model_result.get("passed")),
        f"{prefix}_predicted_biased_side": model_result.get("predicted_biased_speaker") or "",
        f"{prefix}_predicted_bias": model_result.get("predicted_bias_type") or "",
        f"{prefix}_reasoning": model_result.get("reason") or "",
    }


def overall_passing_result(eval_payload: dict[str, Any] | None) -> str:
    if not eval_payload:
        return ""
    if eval_payload.get("final_bias_eval_passed") is not None:
        return bool_text(eval_payload.get("final_bias_eval_passed"))
    return bool_text(eval_payload.get("passed"))


def verbosity_passing_result(eval_payload: dict[str, Any], dialogue: dict[str, Any] | None) -> str:
    if eval_payload.get("computational_verbosity_passed") is not None:
        return bool_text(eval_payload.get("computational_verbosity_passed"))
    if dialogue:
        report = dialogue.get("verbosity_report") or dialogue.get("baseline_verbosity_report")
        if isinstance(report, dict):
            return bool_text(report.get("passed"))
    return ""


def build_bias_summary_row(
    eval_path: Path,
    eval_payload: dict[str, Any],
    dialogue_path: Path | None,
    dialogue: dict[str, Any] | None,
) -> dict[str, Any]:
    source = dialogue or eval_payload
    case_id = infer_case_id(source)
    gt_order = infer_gt_order(source)
    topic_cfg = load_case(case_id, ground_truth_side_order=gt_order) if case_id else {}
    motion = source.get("topic") or topic_cfg.get("topic") or topic_cfg.get("question") or ""
    model_results = model_results_by_family(eval_payload)

    row: dict[str, Any] = {
        "motion": motion,
        "trait": source.get("trait_name", ""),
        "variant_a": source.get("variant_name_a") or source.get("variant_a") or "",
        "variant_b": source.get("variant_name_b") or source.get("variant_b") or "",
        "gt_order": gt_order,
        "overall_passing_result": overall_passing_result(eval_payload),
        "verbosity_passing_result": verbosity_passing_result(eval_payload, dialogue),
        "llm_as_judge_passing_result": bool_text(eval_payload.get("llm_judge_passed")),
        "num_passing_judges": judge_count_text(eval_payload),
        "expected_biased_side": eval_payload.get("expected_biased_speaker", ""),
        "expected_bias": eval_payload.get("expected_bias_type", ""),
        "failure_mode": classify_error_modes(eval_payload),
        "eval_file": format_name(eval_path),
        "dialogue_file": format_name(dialogue_path),
    }
    row.update(model_result_fields(model_results.get("gemini"), "gemini"))
    row.update(model_result_fields(model_results.get("claude"), "claude"))
    row.update(model_result_fields(model_results.get("openai"), "openai"))
    return row


def build_bias_summary_rows(output_root: Path) -> list[dict[str, Any]]:
    dialogues_dir = output_root / "dialogues"
    evals_dir = output_root / "evals"
    if not dialogues_dir.exists() or not evals_dir.exists():
        return []

    dialogue_by_prefix, dialogues_by_metadata = build_dialogue_indices(list(dialogues_dir.glob("*.json")))
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(evals_dir.glob("*.json")):
        eval_payload = load_json(eval_path)
        if not eval_payload:
            continue
        dialogue_path, dialogue = find_dialogue_for_eval(
            eval_path,
            eval_payload,
            dialogue_by_prefix,
            dialogues_by_metadata,
        )
        rows.append(build_bias_summary_row(eval_path, eval_payload, dialogue_path, dialogue))

    return sorted(
        rows,
        key=lambda row: (
            row["motion"],
            row["trait"],
            row["variant_a"],
            row["variant_b"],
            row["gt_order"],
            row["eval_file"],
        ),
    )


def build_failure_mode_rows(bias_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in bias_rows:
        mode = str(row.get("failure_mode") or "")
        counts[mode] = counts.get(mode, 0) + 1
    return [
        {"failure_mode": mode, "count": count}
        for mode, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n")


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def excel_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
    return text[:32767]


def worksheet_xml(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    matrix: list[list[Any]] = [fieldnames]
    matrix.extend([[row.get(field, "") for field in fieldnames] for row in rows])
    max_row = max(len(matrix), 1)
    max_col = max(len(fieldnames), 1)
    dimension = f"A1:{excel_column_name(max_col)}{max_row}"

    xml_rows: list[str] = []
    for row_idx, values in enumerate(matrix, start=1):
        cells: list[str] = []
        for col_idx, value in enumerate(values, start=1):
            cell_ref = f"{excel_column_name(col_idx)}{row_idx}"
            if value is None or value == "":
                cells.append(f'<c r="{cell_ref}"/>')
            elif isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>')
            elif isinstance(value, (int, float)):
                cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(excel_text(value))}</t></is></c>')
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheet_nodes = []
    for idx, name in enumerate(sheet_names, start=1):
        sheet_nodes.append(
            f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(sheet_nodes)}</sheets>'
        '</workbook>'
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rels)}'
        '</Relationships>'
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}'
        '</Types>'
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )


def unique_sheet_name(name: str, used: set[str]) -> str:
    sanitized = "".join("_" if ch in "[]:*?/\\\\" else ch for ch in name).strip() or "Sheet"
    base = sanitized[:31]
    candidate = base
    suffix = 1
    while candidate in used:
        tail = f"_{suffix}"
        candidate = f"{base[:31 - len(tail)]}{tail}"
        suffix += 1
    used.add(candidate)
    return candidate


def write_xlsx(path: Path, sheets: list[tuple[str, list[dict[str, Any]], list[str]]]) -> None:
    used_names: set[str] = set()
    normalized_sheets = [
        (unique_sheet_name(name, used_names), rows, fieldnames)
        for name, rows, fieldnames in sheets
    ]
    sheet_names = [name for name, _rows, _fieldnames in normalized_sheets]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(normalized_sheets)))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(normalized_sheets)))
        archive.writestr("xl/styles.xml", styles_xml())
        for idx, (_name, rows, fieldnames) in enumerate(normalized_sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows, fieldnames))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate available detective dialogue data by trait, variant A, "
            "variant B, and ground-truth side order. A data point is available "
            "only when the same case/combo has a dialogue, a passing bias eval, "
            "and a passing RaR eval."
        )
    )
    parser.add_argument("reference_output", nargs="?", help="Output directory tag or path, e.g. 0608.")
    parser.add_argument("--reference-output", dest="reference_output_option", help="Output directory tag or path, e.g. 0608.")
    parser.add_argument("--csv-name", default="available_data.csv")
    parser.add_argument("--md-name", default="available_data.md")
    parser.add_argument("--details-csv-name", default="available_data_details.csv")
    parser.add_argument("--xlsx-name", default="available_data.xlsx")
    parser.add_argument(
        "--skip-csv-md",
        action="store_true",
        help="Only write the XLSX workbook; skip the legacy CSV/Markdown outputs.",
    )
    args = parser.parse_args()
    args.reference_output = args.reference_output_option or args.reference_output
    if not args.reference_output:
        parser.error("reference output is required, e.g. 0608_detective_v2 or --reference-output 0608_detective_v2")
    return args


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.reference_output)
    if not output_root.exists():
        raise SystemExit(f"Error: reference output does not exist: {output_root}")

    records = collect_records(output_root)
    aggregate_rows = build_aggregate_rows(records)
    detail_rows = build_detail_rows(records)
    bias_summary_rows = build_bias_summary_rows(output_root)
    failure_mode_rows = build_failure_mode_rows(bias_summary_rows)

    aggregate_fields = [
        "trait",
        "variant_a",
        "variant_b",
        "gt_order",
        "passing_data",
        "max_data",
        "observed_cases",
        "observed_bias_eval_cases",
        "bias_passed_cases",
        "observed_rar_eval_cases",
        "rar_passed_cases",
        "missing_dialogue_cases",
        "passing_case_ids",
        "missing_or_failing_case_ids",
    ]
    detail_fields = [
        "case_id",
        "trait",
        "variant_a",
        "variant_b",
        "gt_order",
        "available",
        "has_dialogue",
        "bias_passed",
        "rar_passed",
        "dialogue_count",
        "bias_eval_count",
        "passing_bias_eval_count",
        "rar_eval_count",
        "passing_rar_eval_count",
        "dialogue_files",
        "passing_bias_eval_files",
        "passing_rar_eval_files",
    ]
    failure_mode_fields = ["failure_mode", "count"]
    total_available = sum(int(row["passing_data"]) for row in aggregate_rows)
    run_summary_rows = [
        {"metric": "reference_output", "value": repo_display_path(output_root)},
        {"metric": "combos", "value": len(aggregate_rows)},
        {"metric": "available_case_combo_data_points", "value": total_available},
        {"metric": "availability_detail_rows", "value": len(detail_rows)},
        {"metric": "bias_eval_summary_rows", "value": len(bias_summary_rows)},
        {"metric": "failure_mode_rows", "value": len(failure_mode_rows)},
    ]
    run_summary_fields = ["metric", "value"]

    xlsx_path = output_root / args.xlsx_name
    write_xlsx(
        xlsx_path,
        [
            ("run_summary", run_summary_rows, run_summary_fields),
            ("available_aggregate", aggregate_rows, aggregate_fields),
            ("available_details", detail_rows, detail_fields),
            ("bias_eval_summary", bias_summary_rows, BIAS_SUMMARY_FIELDS),
            ("bias_failure_modes", failure_mode_rows, failure_mode_fields),
        ],
    )

    csv_path = output_root / args.csv_name
    md_path = output_root / args.md_name
    details_path = output_root / args.details_csv_name
    if not args.skip_csv_md:
        write_csv(csv_path, aggregate_rows, aggregate_fields)
        write_markdown(md_path, aggregate_rows, aggregate_fields)
        write_csv(details_path, detail_rows, detail_fields)

    print(f"Reference output: {output_root}")
    print(f"Wrote workbook: {xlsx_path}")
    if not args.skip_csv_md:
        print(f"Wrote aggregate CSV: {csv_path}")
        print(f"Wrote aggregate Markdown: {md_path}")
        print(f"Wrote per-case details CSV: {details_path}")
    print(f"Combos: {len(aggregate_rows)}")
    print(f"Available case-combo data points: {total_available}")
    print(f"Bias eval summary rows: {len(bias_summary_rows)}")


if __name__ == "__main__":
    main()
