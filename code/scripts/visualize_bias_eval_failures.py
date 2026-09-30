from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PATH_ANCHORS = {
    "experiment",
    "llm_participant",
    "outputs",
}
DEFAULT_HTML_NAME = "bias_eval_failure_report.html"
DEFAULT_JSON_NAME = "bias_eval_failure_report.json"
NO_BIAS_LABEL = "NO_BIAS_DETECTED"
UNTYPED_LABEL = "BIAS_DETECTED_BUT_UNTYPED"
MALFORMED_Q1_LABEL = "MALFORMED_Q1"
MALFORMED_Q2_LABEL = "MALFORMED_Q2"
GUIDANCE_NOT_SATISFIED_LABEL = "GUIDANCE_MARKERS_NOT_SATISFIED"
GUIDANCE_SATISFIED_LABEL = "GUIDANCE_MARKERS_SATISFIED"
STRICT_GUIDANCE_NOT_SATISFIED_LABEL = "STRICT_GUIDANCE_NOT_SATISFIED"
GUIDANCE_VIOLATION_LABEL = "GUIDANCE_VIOLATION"
UNKNOWN_LABEL = "UNKNOWN"


def repo_relative_candidate(path: Path) -> Path | None:
    for index, part in enumerate(path.parts):
        if part in REPO_PATH_ANCHORS:
            return Path(*path.parts[index:])
    return None


def repo_display_path(path: str | Path) -> str:
    resolved = Path(path)
    if resolved.is_absolute():
        if resolved.is_relative_to(REPO_ROOT):
            return resolved.relative_to(REPO_ROOT).as_posix()
        candidate = repo_relative_candidate(resolved)
        if candidate is not None:
            return candidate.as_posix()
    return resolved.as_posix()


@dataclass
class EvalFileRecord:
    path: Path
    payload: dict

    @property
    def case_id(self) -> str:
        return self.payload.get("case_id") or "unknown-case"

    @property
    def trait_name(self) -> str:
        return self.payload.get("trait_name") or "unknown-trait"

    @property
    def variants_label(self) -> str:
        return (
            f"{self.payload.get('variant_name_a', '?')}/"
            f"{self.payload.get('variant_name_b', '?')}"
        )

    @property
    def side_order(self) -> str:
        return self.payload.get("ground_truth_side_order") or "unknown-order"

    @property
    def failed(self) -> bool:
        return bool(self.payload.get("passed") is False)


def resolve_output_root(output_arg: str) -> Path:
    path = Path(output_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return REPO_ROOT / path
    return REPO_ROOT / "outputs" / path


def load_eval_records(output_root: Path) -> tuple[list[EvalFileRecord], int, int]:
    eval_dir = output_root / "evals"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    records: list[EvalFileRecord] = []
    legacy_count = 0
    deterministic_count = 0
    for path in sorted(eval_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if "per_model_results" not in payload:
            legacy_count += 1
            continue
        if payload.get("evaluation_type") == "deterministic":
            deterministic_count += 1
            continue
        records.append(EvalFileRecord(path=path, payload=payload))
    return records, legacy_count, deterministic_count


def _predicted_label_for_failed_turn(turn_score: dict) -> str:
    failure_modes = set(turn_score.get("failure_modes") or [])
    if "malformed_question_1" in failure_modes:
        return MALFORMED_Q1_LABEL

    if turn_score.get("expected_has_bias") is False:
        if turn_score.get("question_2_asked") is not True:
            return UNTYPED_LABEL
        predicted = turn_score.get("question_2_predicted_bias_type")
        if predicted:
            return str(predicted)
        if "malformed_question_2" in failure_modes:
            return MALFORMED_Q2_LABEL
        return UNTYPED_LABEL

    if turn_score.get("question_1_predicted_has_bias") is not True:
        return NO_BIAS_LABEL
    predicted = turn_score.get("question_2_predicted_bias_type")
    if predicted:
        return str(predicted)
    if "malformed_question_2" in failure_modes:
        return MALFORMED_Q2_LABEL
    return UNKNOWN_LABEL


def _predicted_guidance_label_for_failed_turn(turn_score: dict) -> str:
    failure_modes = set(turn_score.get("failure_modes") or [])
    if "malformed_marker_check" in failure_modes or "malformed_question_1" in failure_modes:
        return MALFORMED_Q1_LABEL
    if "assigned_guidance_violation" in failure_modes:
        return GUIDANCE_VIOLATION_LABEL
    if "strict_guidance_not_satisfied" in failure_modes:
        return STRICT_GUIDANCE_NOT_SATISFIED_LABEL
    guidance_satisfied = turn_score.get(
        "question_1_guidance_satisfied",
        turn_score.get(
            "question_1_predicted_guidance_satisfied",
            turn_score.get("question_1_predicted_has_bias"),
        ),
    )
    if guidance_satisfied is True:
        return GUIDANCE_SATISFIED_LABEL
    if guidance_satisfied is False:
        return GUIDANCE_NOT_SATISFIED_LABEL
    return UNKNOWN_LABEL


def _normalize_expected_bias(turn_score: dict) -> str:
    return str(turn_score.get("expected_bias_type") or UNKNOWN_LABEL)


def _normalize_expected_guidance(turn_score: dict) -> str:
    return str(
        turn_score.get("expected_guidance_type")
        or turn_score.get("expected_bias_type")
        or turn_score.get("variant_name")
        or UNKNOWN_LABEL
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_failure_summary(records: list[EvalFileRecord], legacy_count: int, deterministic_count: int) -> dict:
    total_eval_files = len(records)
    failed_records = [record for record in records if record.failed]

    aggregate_failure_modes = Counter()
    failed_files_by_trait = Counter()
    failed_files_by_variant_pair = Counter()
    failed_turn_outcomes = Counter()
    failed_model_turns = Counter()
    baseline_false_positive_types = Counter()
    active_failure_kinds = Counter()
    guidance_failure_kinds = Counter()
    active_confusion = Counter()
    baseline_confusion = Counter()
    guidance_confusion = Counter()
    model_error_counts = Counter()
    file_failure_rows: list[dict] = []
    marker_eval_files = sum(1 for record in records if record.payload.get("evaluation_type") == "llm_marker")

    total_turn_outcomes_failed = 0
    total_model_turn_failures = 0

    for record in failed_records:
        payload = record.payload
        is_marker_eval = payload.get("evaluation_type") == "llm_marker"
        aggregate_failure_modes.update(payload.get("failure_modes") or [])
        failed_files_by_trait[record.trait_name] += 1
        failed_files_by_variant_pair[record.variants_label] += 1

        turn_outcomes = payload.get("turn_outcomes") or []
        baseline_turn_failures = 0
        active_turn_failures = 0
        for turn_outcome in turn_outcomes:
            if turn_outcome.get("passed") is True:
                continue
            total_turn_outcomes_failed += 1
            if is_marker_eval:
                variant_name = turn_outcome.get("variant_name") or "unknown_variant"
                failed_turn_outcomes[f"{variant_name}_guidance_failures"] += 1
                if variant_name == "baseline":
                    baseline_turn_failures += 1
                elif variant_name == "active":
                    active_turn_failures += 1
            elif turn_outcome.get("expected_has_bias") is True:
                active_turn_failures += 1
                failed_turn_outcomes["active_turn_failures"] += 1
            else:
                baseline_turn_failures += 1
                failed_turn_outcomes["baseline_turn_failures"] += 1

        for model_result in payload.get("per_model_results") or []:
            model_name = model_result.get("model") or "unknown-model"
            for turn_score in model_result.get("turn_scores") or []:
                if turn_score.get("required_for_pass") is False:
                    continue
                if turn_score.get("passed") is True:
                    continue
                total_model_turn_failures += 1
                model_error_counts[model_name] += 1
                if is_marker_eval:
                    variant_name = turn_score.get("variant_name") or "unknown_variant"
                    predicted_label = _predicted_guidance_label_for_failed_turn(turn_score)
                    failed_model_turns[f"{variant_name}_guidance_failures"] += 1
                    if predicted_label == GUIDANCE_NOT_SATISFIED_LABEL:
                        guidance_failure_kinds["assigned_guidance_markers_missing"] += 1
                    elif predicted_label == STRICT_GUIDANCE_NOT_SATISFIED_LABEL:
                        guidance_failure_kinds["strict_guidance_not_satisfied"] += 1
                    elif predicted_label == GUIDANCE_VIOLATION_LABEL:
                        guidance_failure_kinds["assigned_guidance_violation"] += 1
                    elif predicted_label in {MALFORMED_Q1_LABEL, UNKNOWN_LABEL}:
                        guidance_failure_kinds["malformed_or_unknown_guidance_check"] += 1
                    else:
                        guidance_failure_kinds["unexpected_guidance_failure"] += 1
                    guidance_confusion[(_normalize_expected_guidance(turn_score), predicted_label)] += 1
                    continue

                expected_has_bias = turn_score.get("expected_has_bias") is True
                predicted_label = _predicted_label_for_failed_turn(turn_score)
                if expected_has_bias:
                    failed_model_turns["active_turn_failures"] += 1
                    if predicted_label == NO_BIAS_LABEL:
                        active_failure_kinds["missed_bias_detection"] += 1
                    elif predicted_label in {MALFORMED_Q1_LABEL, MALFORMED_Q2_LABEL, UNKNOWN_LABEL}:
                        active_failure_kinds["malformed_or_untyped"] += 1
                    else:
                        active_failure_kinds["wrong_bias_type"] += 1
                    active_confusion[(_normalize_expected_bias(turn_score), predicted_label)] += 1
                else:
                    failed_model_turns["baseline_turn_failures"] += 1
                    if predicted_label == MALFORMED_Q1_LABEL:
                        baseline_false_positive_types[MALFORMED_Q1_LABEL] += 1
                    else:
                        baseline_false_positive_types[predicted_label] += 1
                    baseline_confusion[("baseline_clean_turn", predicted_label)] += 1

        file_failure_rows.append(
            {
                "file": record.path.name,
                "case_id": record.case_id,
                "trait_name": record.trait_name,
                "variant_pair": record.variants_label,
                "ground_truth_side_order": record.side_order,
                "failed_turns": sum(1 for item in turn_outcomes if item.get("passed") is False),
                "baseline_turn_failures": baseline_turn_failures,
                "active_turn_failures": active_turn_failures,
                "models_with_all_turns_passing": payload.get("models_with_all_turns_passing"),
                "models_total": len(payload.get("per_model_results") or []),
                "failure_modes": payload.get("failure_modes") or [],
            }
        )

    file_failure_rows.sort(
        key=lambda row: (
            row["failed_turns"],
            row["active_turn_failures"],
            row["baseline_turn_failures"],
        ),
        reverse=True,
    )

    return {
        "output_root": repo_display_path(records[0].path.parents[1]) if records else "",
        "total_eval_files": total_eval_files,
        "legacy_eval_files_skipped": legacy_count,
        "deterministic_eval_files_skipped": deterministic_count,
        "failed_eval_files": len(failed_records),
        "marker_eval_files": marker_eval_files,
        "semantic_eval_files": total_eval_files - marker_eval_files,
        "pass_rate": round(
            100.0 * _safe_ratio(total_eval_files - len(failed_records), total_eval_files),
            2,
        ),
        "failed_turn_outcomes_total": total_turn_outcomes_failed,
        "failed_model_turns_total": total_model_turn_failures,
        "aggregate_failure_modes": dict(aggregate_failure_modes.most_common()),
        "failed_files_by_trait": dict(failed_files_by_trait.most_common()),
        "failed_files_by_variant_pair": dict(failed_files_by_variant_pair.most_common()),
        "failed_turn_outcomes": dict(failed_turn_outcomes),
        "failed_model_turns": dict(failed_model_turns),
        "baseline_false_positive_types": dict(baseline_false_positive_types.most_common()),
        "active_failure_kinds": dict(active_failure_kinds.most_common()),
        "guidance_failure_kinds": dict(guidance_failure_kinds.most_common()),
        "active_confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in active_confusion.most_common()
        ],
        "baseline_confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in baseline_confusion.most_common()
        ],
        "guidance_confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in guidance_confusion.most_common()
        ],
        "model_error_counts": dict(model_error_counts.most_common()),
        "failed_file_rows": file_failure_rows,
    }


def render_cards(summary: dict) -> str:
    cards = [
        ("Eval files", str(summary["total_eval_files"])),
        ("Marker eval files", str(summary["marker_eval_files"])),
        ("Semantic eval files", str(summary["semantic_eval_files"])),
        ("Legacy skipped", str(summary["legacy_eval_files_skipped"])),
        ("Deterministic skipped", str(summary["deterministic_eval_files_skipped"])),
        ("Failed eval files", str(summary["failed_eval_files"])),
        ("Pass rate", f"{summary['pass_rate']}%"),
        ("Failed turn outcomes", str(summary["failed_turn_outcomes_total"])),
        ("Failed model-turn decisions", str(summary["failed_model_turns_total"])),
    ]
    return "".join(
        f'<div class="card"><div class="card-label">{escape(label)}</div><div class="card-value">{escape(value)}</div></div>'
        for label, value in cards
    )


def render_bar_chart(title: str, data: dict[str, int], *, empty_message: str) -> str:
    if not data:
        return f'<section class="panel"><h2>{escape(title)}</h2><p class="muted">{escape(empty_message)}</p></section>'

    max_value = max(data.values()) or 1
    rows = []
    for label, value in data.items():
        width_pct = round(100.0 * value / max_value, 2)
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{escape(str(label))}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width_pct}%'></div></div>"
            f"<div class='bar-value'>{value}</div>"
            "</div>"
        )
    return f"<section class='panel'><h2>{escape(title)}</h2>{''.join(rows)}</section>"


def render_confusion_table(
    title: str,
    rows: list[dict],
    *,
    empty_message: str,
) -> str:
    if not rows:
        return f'<section class="panel"><h2>{escape(title)}</h2><p class="muted">{escape(empty_message)}</p></section>'

    expected_labels = sorted({row["expected"] for row in rows})
    predicted_labels = sorted({row["predicted"] for row in rows})
    counts = {(row["expected"], row["predicted"]): row["count"] for row in rows}
    max_count = max(counts.values()) or 1

    header = "".join(f"<th>{escape(label)}</th>" for label in predicted_labels)
    body_rows = []
    for expected in expected_labels:
        cells = []
        for predicted in predicted_labels:
            count = counts.get((expected, predicted), 0)
            alpha = 0.08 + (0.92 * count / max_count if count else 0.0)
            bg = f"rgba(196, 78, 82, {alpha:.3f})" if count else "rgba(255,255,255,0.02)"
            cells.append(
                f"<td style='background:{bg}' title='{escape(expected)} -> {escape(predicted)} = {count}'>{count}</td>"
            )
        body_rows.append(f"<tr><th>{escape(expected)}</th>{''.join(cells)}</tr>")

    return (
        f"<section class='panel'><h2>{escape(title)}</h2>"
        "<div class='table-wrap'><table class='matrix'>"
        f"<thead><tr><th>Expected \\ Predicted</th>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div></section>"
    )


def render_failed_files_table(rows: list[dict]) -> str:
    if not rows:
        return "<section class='panel'><h2>Failed Eval Files</h2><p class='muted'>No failed eval files.</p></section>"

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{escape(row['file'])}</td>"
            f"<td>{escape(row['case_id'])}</td>"
            f"<td>{escape(row['trait_name'])}</td>"
            f"<td>{escape(row['variant_pair'])}</td>"
            f"<td>{escape(row['ground_truth_side_order'])}</td>"
            f"<td>{row['failed_turns']}</td>"
            f"<td>{row['baseline_turn_failures']}</td>"
            f"<td>{row['active_turn_failures']}</td>"
            f"<td>{row['models_with_all_turns_passing']}/{row['models_total']}</td>"
            f"<td>{escape(', '.join(row['failure_modes']))}</td>"
            "</tr>"
        )
    return (
        "<section class='panel wide'><h2>Failed Eval Files</h2>"
        "<div class='table-wrap'><table>"
        "<thead><tr>"
        "<th>File</th><th>Case</th><th>Trait</th><th>Variants</th><th>Order</th>"
        "<th>Failed turns</th><th>Baseline guidance/bias failed turns</th><th>Active guidance/bias failed turns</th>"
        "<th>Models passed</th><th>Failure modes</th>"
        "</tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table></div></section>"
    )


def render_html(summary: dict, output_root: Path) -> str:
    title = f"Guidance/Bias Eval Failure Report: {output_root.name}"
    sections = [
        render_bar_chart(
            "Aggregate Failure Modes",
            summary["aggregate_failure_modes"],
            empty_message="No aggregate failure modes recorded.",
        ),
        render_bar_chart(
            "Failed Eval Files by Trait",
            summary["failed_files_by_trait"],
            empty_message="No failed eval files.",
        ),
        render_bar_chart(
            "Failed Turn Outcomes by Variant/Expectation",
            summary["failed_turn_outcomes"],
            empty_message="No failed turn outcomes.",
        ),
        render_bar_chart(
            "Model-Level Failed Turn Decisions by Variant/Expectation",
            summary["failed_model_turns"],
            empty_message="No model-level failed turn decisions.",
        ),
        render_bar_chart(
            "Marker Guidance Failure Kinds",
            summary["guidance_failure_kinds"],
            empty_message="No marker-guidance failures.",
        ),
        render_bar_chart(
            "Legacy/Semantic Baseline False Positives: Predicted As",
            summary["baseline_false_positive_types"],
            empty_message="No baseline false-positive errors.",
        ),
        render_bar_chart(
            "Legacy/Semantic Active Failure Kinds",
            summary["active_failure_kinds"],
            empty_message="No active-turn failures.",
        ),
        render_bar_chart(
            "Model Error Counts",
            summary["model_error_counts"],
            empty_message="No model error counts.",
        ),
        render_confusion_table(
            "Marker Guidance Satisfaction Matrix",
            summary["guidance_confusion"],
            empty_message="No marker-guidance confusions.",
        ),
        render_confusion_table(
            "Legacy/Semantic Active Bias Confusion Matrix",
            summary["active_confusion"],
            empty_message="No active-bias confusions.",
        ),
        render_confusion_table(
            "Legacy/Semantic Baseline False-Positive Confusion Matrix",
            summary["baseline_confusion"],
            empty_message="No baseline false-positive confusions.",
        ),
        render_failed_files_table(summary["failed_file_rows"]),
    ]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f1a14;
      --muted: #6e6559;
      --accent: #9f3d2f;
      --accent-soft: #d77b59;
      --line: #ddd1be;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: Georgia, "Iowan Old Style", serif;
      background: linear-gradient(180deg, #f5f1e8 0%, #ece3d4 100%);
      color: var(--ink);
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 20px; }}
    p {{ line-height: 1.45; }}
    .subhead {{ color: var(--muted); margin-top: 8px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 24px rgba(31, 26, 20, 0.06);
    }}
    .card {{
      padding: 18px 20px;
    }}
    .card-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .card-value {{
      margin-top: 10px;
      font-size: 30px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px;
    }}
    .panel {{
      padding: 18px 20px 20px;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    .muted {{
      color: var(--muted);
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 190px 1fr 64px;
      gap: 12px;
      align-items: center;
      margin: 10px 0;
    }}
    .bar-label {{
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    .bar-track {{
      height: 18px;
      border-radius: 999px;
      background: #efe7d9;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent-soft));
    }}
    .bar-value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      background: rgba(0,0,0,0.02);
    }}
    .matrix td {{
      text-align: center;
      min-width: 72px;
      font-variant-numeric: tabular-nums;
    }}
    .matrix th:first-child {{
      min-width: 180px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="subhead">Analyzed directory: {escape(repo_display_path(output_root))}. Marker-based eval statistics describe assigned-guidance satisfaction failures by variant; legacy semantic sections retain the older bias-presence interpretation.</p>
  </header>
  <section class="cards">{render_cards(summary)}</section>
  <section class="grid">
    {''.join(sections)}
  </section>
</body>
</html>"""


def write_outputs(summary: dict, output_root: Path, html_name: str, json_name: str) -> tuple[Path, Path]:
    html_path = output_root / html_name
    json_path = output_root / json_name
    html_path.write_text(render_html(summary, output_root))
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return html_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize detective bias-eval failures for a specified output directory."
    )
    parser.add_argument(
        "output_arg",
        nargs="?",
        help="Output run directory to analyze, e.g. 0509_detective_v2 or outputs/0509_detective_v2.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        help="Output run directory to analyze, e.g. 0509_detective_v2 or outputs/0509_detective_v2.",
    )
    parser.add_argument(
        "--html-name",
        default=DEFAULT_HTML_NAME,
        help=f"Filename for the generated HTML report inside the output directory. Default: {DEFAULT_HTML_NAME}",
    )
    parser.add_argument(
        "--json-name",
        default=DEFAULT_JSON_NAME,
        help=f"Filename for the generated JSON summary inside the output directory. Default: {DEFAULT_JSON_NAME}",
    )
    args = parser.parse_args()
    if args.output and args.output_arg:
        parser.error("provide the output directory either positionally or with --output, not both")
    args.output = args.output or args.output_arg
    if not args.output:
        parser.error("the following arguments are required: output or --output")
    return args


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output)
    records, legacy_count, deterministic_count = load_eval_records(output_root)
    summary = build_failure_summary(records, legacy_count, deterministic_count)
    html_path, json_path = write_outputs(
        summary,
        output_root,
        html_name=args.html_name,
        json_name=args.json_name,
    )
    print(f"[bias-eval-failures] analyzed output: {output_root}")
    print(f"[bias-eval-failures] eval files: {summary['total_eval_files']}")
    print(f"[bias-eval-failures] failed eval files: {summary['failed_eval_files']}")
    print(f"[bias-eval-failures] legacy eval files skipped: {summary['legacy_eval_files_skipped']}")
    print(f"[bias-eval-failures] deterministic eval files skipped: {summary['deterministic_eval_files_skipped']}")
    print(f"[bias-eval-failures] failed turn outcomes: {summary['failed_turn_outcomes_total']}")
    print(f"[bias-eval-failures] failed model-turn decisions: {summary['failed_model_turns_total']}")
    print(f"[bias-eval-failures] html report: {html_path}")
    print(f"[bias-eval-failures] json summary: {json_path}")


if __name__ == "__main__":
    main()
