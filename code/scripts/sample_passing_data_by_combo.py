#!/usr/bin/env python3
#
# Example:
# .venv/bin/python code/scripts/sample_passing_data_by_combo.py \
#   outputs/0610_detective_v2_evidence \
#   --seed 0 \
#   --samples-per-combo 4
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from available_data import (
    ComboKey,
    RunAvailability,
    collect_records,
    dialogue_run_prefix,
    repo_display_path,
    resolve_output_root,
    run_prefix,
)


SAMPLE_FIELDS = [
    "trait",
    "variant_a",
    "variant_b",
    "gt_order",
    "case_id",
    "dialogue_file",
    "bias_eval_file",
    "rar_eval_file",
    "dialogue_path",
    "bias_eval_path",
    "rar_eval_path",
]


DEFAULT_IGNORED_TRAITS = {
    "fallacy_trait__ad_hominem",
    "fallacy_trait__begging_the_question",
    "fallacy_trait__hasty_generalization",
    "fallacy_trait__post_hoc_ergo_propter_hoc",
}


def choose_one(values: set[str], rng: random.Random) -> str:
    return rng.choice(sorted(values))


def prefix_map(files: set[str], suffix: str) -> dict[str, list[str]]:
    by_prefix: dict[str, list[str]] = {}
    for filename in sorted(files):
        prefix = run_prefix(Path(filename), suffix)
        if prefix:
            by_prefix.setdefault(prefix, []).append(filename)
    return by_prefix


def dialogue_prefix_map(files: set[str]) -> dict[str, list[str]]:
    by_prefix: dict[str, list[str]] = {}
    for filename in sorted(files):
        prefix = dialogue_run_prefix(Path(filename))
        if prefix:
            by_prefix.setdefault(prefix, []).append(filename)
    return by_prefix


def dialogue_path(output_root: Path, filename: str) -> Path:
    for subdir in ("dialogues", "accepted"):
        path = output_root / subdir / filename
        if path.exists():
            return path
    return output_root / "dialogues" / filename


def choose_file_triplet(record: RunAvailability, rng: random.Random) -> tuple[str, str, str]:
    dialogues = dialogue_prefix_map(record.dialogue_files)
    bias_evals = prefix_map(record.passing_bias_eval_files, "_eval")
    rar_evals = prefix_map(record.passing_rar_eval_files, "_rar_eval")

    exact_prefixes = sorted(set(dialogues) & set(bias_evals) & set(rar_evals))
    if exact_prefixes:
        prefix = rng.choice(exact_prefixes)
        return (
            rng.choice(dialogues[prefix]),
            rng.choice(bias_evals[prefix]),
            rng.choice(rar_evals[prefix]),
        )

    dialogue_eval_prefixes = sorted(set(dialogues) & set(bias_evals))
    if dialogue_eval_prefixes:
        prefix = rng.choice(dialogue_eval_prefixes)
        return (
            rng.choice(dialogues[prefix]),
            rng.choice(bias_evals[prefix]),
            choose_one(record.passing_rar_eval_files, rng),
        )

    return (
        choose_one(record.dialogue_files, rng),
        choose_one(record.passing_bias_eval_files, rng),
        choose_one(record.passing_rar_eval_files, rng),
    )


def group_available_records(
    records: dict[tuple[str, str, str, str, str], RunAvailability],
    ignored_traits: set[str],
) -> dict[ComboKey, list[RunAvailability]]:
    grouped: dict[ComboKey, list[RunAvailability]] = {}
    for record in records.values():
        if not record.available:
            continue
        if record.trait in ignored_traits:
            continue
        key = (record.trait, record.variant_a, record.variant_b, record.gt_order)
        grouped.setdefault(key, []).append(record)
    return grouped


def build_sample_rows(
    output_root: Path,
    seed: int | None,
    ignored_traits: set[str],
    samples_per_combo: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = collect_records(output_root)
    grouped = group_available_records(records, ignored_traits)

    rows: list[dict[str, Any]] = []
    for (trait, variant_a, variant_b, gt_order), candidates in sorted(grouped.items()):
        sorted_candidates = sorted(candidates, key=lambda item: item.case_id)
        sample_size = min(samples_per_combo, len(sorted_candidates))
        sampled_records = rng.sample(sorted_candidates, sample_size)

        for record in sampled_records:
            dialogue_file, bias_eval_file, rar_eval_file = choose_file_triplet(record, rng)

            rows.append(
                {
                    "trait": trait,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "gt_order": gt_order,
                    "case_id": record.case_id,
                    "dialogue_file": dialogue_file,
                    "bias_eval_file": bias_eval_file,
                    "rar_eval_file": rar_eval_file,
                    "dialogue_path": repo_display_path(dialogue_path(output_root, dialogue_file)),
                    "bias_eval_path": repo_display_path(output_root / "evals" / bias_eval_file),
                    "rar_eval_path": repo_display_path(output_root / "rar_evals" / rar_eval_file),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample one passing data point for each observed "
            "{trait, variant_a, variant_b, gt_order} combo. A data point is "
            "passing only if it has a dialogue, a passing bias eval, and a "
            "passing RaR eval according to available_data.py."
        )
    )
    parser.add_argument(
        "reference_output",
        help="Output directory tag or path, e.g. 0610_detective_v2_evidence or outputs/0610_detective_v2_evidence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--samples-per-combo",
        type=int,
        default=1,
        help="Number of passing data points to sample per combo. Must be between 1 and 4.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "CSV output path. Defaults to "
            "<reference_output>/random_passing_data_by_combo.csv."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional JSONL output path with the same sampled rows.",
    )
    parser.add_argument(
        "--include-ignored-traits",
        action="store_true",
        help=(
            "Include low-passing fallacy subdefinitions that are skipped by default: "
            + ", ".join(sorted(DEFAULT_IGNORED_TRAITS))
        ),
    )
    parser.add_argument(
        "--exclude-trait",
        action="append",
        default=[],
        help="Additional trait to exclude from sampling. Can be provided multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.reference_output)
    if not output_root.exists():
        raise SystemExit(f"Error: reference output does not exist: {output_root}")
    if args.samples_per_combo < 1 or args.samples_per_combo > 4:
        raise SystemExit("Error: --samples-per-combo must be between 1 and 4.")

    ignored_traits = set(args.exclude_trait)
    if not args.include_ignored_traits:
        ignored_traits.update(DEFAULT_IGNORED_TRAITS)

    rows = build_sample_rows(output_root, args.seed, ignored_traits, args.samples_per_combo)
    csv_path = Path(args.output_csv) if args.output_csv else output_root / "random_passing_data_by_combo.csv"
    write_csv(csv_path, rows)
    print(
        f"Wrote {len(rows)} sampled passing rows to {csv_path} "
        f"({args.samples_per_combo} per combo max)"
    )
    if ignored_traits:
        print(f"Ignored traits: {', '.join(sorted(ignored_traits))}")

    if args.output_jsonl:
        jsonl_path = Path(args.output_jsonl)
        write_jsonl(jsonl_path, rows)
        print(f"Wrote JSONL sample to {jsonl_path}")


if __name__ == "__main__":
    main()
