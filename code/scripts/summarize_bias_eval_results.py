from __future__ import annotations

import argparse

from available_data import (
    BIAS_SUMMARY_FIELDS,
    build_bias_summary_rows,
    build_failure_mode_rows,
    resolve_output_root,
    write_csv,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for the bias-eval summary view. "
            "The combined implementation lives in available_data.py."
        )
    )
    parser.add_argument("date_tag", help="Output run directory under outputs/, e.g. 0608_detective_v2.")
    parser.add_argument("--csv-name", default="bias_eval_summary.csv")
    parser.add_argument("--md-name", default="bias_eval_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.date_tag)
    if not output_root.exists():
        raise SystemExit(f"Error: reference output does not exist: {output_root}")

    rows = build_bias_summary_rows(output_root)
    csv_path = output_root / args.csv_name
    md_path = output_root / args.md_name
    write_csv(csv_path, rows, BIAS_SUMMARY_FIELDS)
    write_markdown(md_path, rows, BIAS_SUMMARY_FIELDS)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote Markdown table to {md_path}")
    print("Error modes:")
    for row in build_failure_mode_rows(rows):
        print(f"  {row['failure_mode']}: {row['count']}")


if __name__ == "__main__":
    main()
