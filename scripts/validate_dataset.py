#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from factory_common import (
    DEFAULT_GP4_WS,
    ValidationIssue,
    load_repo_contract,
    parse_single_json_object,
    read_json,
    validate_semantic_payload,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
MASTER_SCHEMA_PATH = ROOT / "schemas/master_example.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate GP4 fine-tuning JSONL rows against the local contract."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--contract-repo", type=Path, default=DEFAULT_GP4_WS)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/validation_report.json"),
    )
    args = parser.parse_args()

    input_paths = _expand_inputs(args.input)
    if not input_paths:
        print("No input files matched.")
        return 2

    contract = load_repo_contract(args.contract_repo)
    master_validator = Draft7Validator(read_json(MASTER_SCHEMA_PATH))
    issues: list[ValidationIssue] = []
    total_rows = 0

    for input_path in input_paths:
        for line_number, row in _iter_jsonl(input_path, issues):
            if row is None:
                continue
            total_rows += 1
            issues.extend(
                _validate_row(
                    row,
                    input_path=input_path,
                    line_number=line_number,
                    master_validator=master_validator,
                    contract=contract,
                )
            )

    invalid_ids = {(issue.file, issue.line) for issue in issues}
    summary = {
        "input_files": [str(path) for path in input_paths],
        "rows": total_rows,
        "valid": total_rows - len(invalid_ids),
        "invalid": len(invalid_ids),
        "issues": [issue.__dict__ for issue in issues],
    }
    write_json(args.report, summary)

    print(
        f"rows={summary['rows']} valid={summary['valid']} "
        f"invalid={summary['invalid']} report={args.report}"
    )
    for issue in issues:
        print(f"{issue.file}:{issue.line} {issue.row_id}: {issue.message}")

    if args.strict and issues:
        return 1
    return 0


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    return [path for path in paths if path.exists()]


def _iter_jsonl(
    path: Path, issues: list[ValidationIssue]
) -> list[tuple[int, dict[str, Any] | None]]:
    rows: list[tuple[int, dict[str, Any] | None]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    ValidationIssue(str(path), line_number, "<parse>", f"invalid JSONL: {exc.msg}")
                )
                rows.append((line_number, None))
                continue
            if not isinstance(decoded, dict):
                issues.append(
                    ValidationIssue(str(path), line_number, "<parse>", "row must be a JSON object")
                )
                rows.append((line_number, None))
                continue
            rows.append((line_number, decoded))
    return rows


def _validate_row(
    row: dict[str, Any],
    *,
    input_path: Path,
    line_number: int,
    master_validator: Draft7Validator,
    contract: dict[str, Any],
) -> list[ValidationIssue]:
    row_id = str(row.get("id", "<missing-id>"))
    issues: list[ValidationIssue] = []

    schema_errors = sorted(master_validator.iter_errors(row), key=lambda error: error.path)
    for error in schema_errors:
        location = ".".join(str(part) for part in error.path) or "$"
        issues.append(
            ValidationIssue(
                str(input_path),
                line_number,
                row_id,
                f"master schema error at {location}: {error.message}",
            )
        )
    if schema_errors:
        return issues

    assistant_message = row["messages"][-1]
    try:
        assistant_json = parse_single_json_object(assistant_message["content"])
    except ValueError as exc:
        return [
            ValidationIssue(str(input_path), line_number, row_id, str(exc)),
        ]

    if assistant_json != row["expected_json"]:
        issues.append(
            ValidationIssue(
                str(input_path),
                line_number,
                row_id,
                "expected_json must exactly match parsed assistant.content",
            )
        )

    semantic_issue = validate_semantic_payload(assistant_json, contract)
    if semantic_issue:
        issues.append(
            ValidationIssue(str(input_path), line_number, row_id, semantic_issue)
        )

    metadata = row["metadata"]
    if metadata["task_type"] in {"ambiguous", "hard_negative", "vision_stub"}:
        if "error" not in assistant_json:
            issues.append(
                ValidationIssue(
                    str(input_path),
                    line_number,
                    row_id,
                    f"{metadata['task_type']} rows must train a safe error payload",
                )
            )
    if metadata["requires_perception"] and "error" not in assistant_json:
        issues.append(
            ValidationIssue(
                str(input_path),
                line_number,
                row_id,
                "requires_perception rows must not produce executable Semantic IR in this wave",
            )
        )
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
