#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from factory_common import (
    find_forbidden_key,
    find_forbidden_text,
    parse_single_json_object,
    read_json,
    read_yaml,
)
from v2_taxonomy import distribution_summary, invalid_scenario_tags, quota_failures

ROOT = Path(__file__).resolve().parents[1]
MASTER_SCHEMA_PATH = ROOT / "schemas/master_example.schema.json"
REACT_SCHEMA_PATH = ROOT / "schemas/gp4_react_ir.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GP4 ReAct-IR dataset rows.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cloud-root", action="append", default=[])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-tmp", action="store_true")
    parser.add_argument("--distribution-spec", type=Path)
    parser.add_argument("--enforce-v2-distribution", action="store_true")
    args = parser.parse_args()

    policy = CloudStoragePolicy(
        cloud_roots=tuple(Path(root) for root in args.cloud_root),
        allow_tmp=args.allow_tmp,
    )
    if not args.cloud_root and not args.allow_tmp:
        print("CLOUD_ROOT is required unless --allow-tmp is used for test reports.")
        return 1
    if args.cloud_root or args.allow_tmp:
        if not is_allowed_cloud_path(args.report, policy):
            print(f"report path is not allowed by cloud storage policy: {args.report}")
            return 1

    master_validator = Draft7Validator(read_json(MASTER_SCHEMA_PATH))
    react_validator = Draft7Validator(read_json(REACT_SCHEMA_PATH))
    issues: list[dict[str, Any]] = []
    valid_rows_for_distribution: list[dict[str, Any]] = []
    total_rows = 0

    for input_path in _expand_inputs(args.input):
        for line_number, row, parse_error in _iter_jsonl(input_path):
            total_rows += 1
            if parse_error:
                issues.append(
                    {
                        "file": str(input_path),
                        "line": line_number,
                        "id": "<parse>",
                        "message": parse_error,
                    }
                )
                continue
            row_id = str(row.get("id", "<missing-id>"))
            row_issues = validate_row(
                row,
                react_validator=react_validator,
                master_validator=master_validator,
            )
            if not row_issues:
                valid_rows_for_distribution.append(row)
            for message in row_issues:
                issues.append(
                    {
                        "file": str(input_path),
                        "line": line_number,
                        "id": row_id,
                        "message": message,
                    }
                )

    distribution = distribution_summary(valid_rows_for_distribution)
    quota_issues: list[dict[str, Any]] = []
    if args.enforce_v2_distribution:
        if not args.distribution_spec:
            print("--distribution-spec is required with --enforce-v2-distribution")
            return 1
        try:
            spec = read_yaml(args.distribution_spec)
        except Exception as exc:
            print(f"invalid distribution spec: {exc}")
            return 1
        gates = spec.get("v2_distribution_gates")
        if not isinstance(gates, dict):
            print("distribution spec must contain a v2_distribution_gates mapping")
            return 1
        minimums = gates.get("scenario_tag_min_counts", {})
        if not isinstance(minimums, dict):
            print("v2_distribution_gates.scenario_tag_min_counts must be a mapping")
            return 1
        try:
            quota_issues = quota_failures(valid_rows_for_distribution, gates)
            min_total = int(gates.get("min_total_rows", 0))
        except (TypeError, ValueError) as exc:
            print(f"invalid v2_distribution_gates value: {exc}")
            return 1
        if total_rows < min_total:
            quota_issues.append(
                {
                    "tag": "__total__",
                    "actual": total_rows,
                    "minimum": min_total,
                    "message": f"dataset has {total_rows} rows but requires {min_total}",
                }
            )

    report = write_validation_report(
        args.report,
        total_rows,
        issues,
        distribution=distribution,
        quota_issues=quota_issues,
    )
    print(
        f"passed={report['passed']} rows={report['rows']} "
        f"valid={report['valid']} invalid={report['invalid']} report={args.report}"
    )
    for issue in issues:
        print(f"{issue['file']}:{issue['line']} {issue['id']}: {issue['message']}")
    return 1 if args.strict and not report["passed"] else 0


def validate_row(row: dict, *, react_validator, master_validator) -> list[str]:
    issues: list[str] = []
    master_errors = sorted(master_validator.iter_errors(row), key=lambda error: error.path)
    for error in master_errors:
        location = ".".join(str(part) for part in error.path) or "$"
        issues.append(f"master schema error at {location}: {error.message}")
    bad_tags = invalid_scenario_tags(row)
    for tag in bad_tags:
        issues.append(f"invalid scenario tag: {tag}")
    if master_errors:
        return issues

    assistant_message = row["messages"][-1]
    try:
        assistant_json = parse_single_json_object(assistant_message["content"])
    except ValueError as exc:
        issues.append(str(exc))
        return issues

    if assistant_json != row["expected_json"]:
        issues.append("expected_json must exactly match parsed assistant.content")

    issues.extend(_payload_safety_issues("expected_json", row["expected_json"]))
    react_ir = row.get("react_ir")
    if isinstance(react_ir, dict):
        react_errors = sorted(react_validator.iter_errors(react_ir), key=lambda error: error.path)
        for error in react_errors:
            location = ".".join(str(part) for part in error.path) or "$"
            issues.append(f"react_ir schema error at {location}: {error.message}")
        issues.extend(_payload_safety_issues("react_ir", react_ir))

    metadata = row["metadata"]
    requires_safe_error = (
        metadata["task_type"] in {"ambiguous", "hard_negative", "vision_stub"}
        or metadata["requires_perception"]
        or metadata["safety_class"] in {"unsafe_rejected", "perception_required"}
    )
    if requires_safe_error and not _is_safe_error(row):
        if metadata["requires_perception"]:
            issues.append("requires_perception rows must use safe_error")
        else:
            issues.append(f"{metadata['task_type']} rows must use safe_error")
    return issues


def compatibility_intent(expected_json: dict) -> str:
    intent = expected_json.get("intent")
    if isinstance(intent, str) and intent:
        return intent
    error = expected_json.get("error")
    if isinstance(error, str) and error:
        return "safe_error"
    return ""


def react_intent(react_ir: dict) -> str:
    act = react_ir.get("act", {})
    if not isinstance(act, dict):
        return ""
    intent = act.get("intent", "")
    return intent if isinstance(intent, str) else ""


def write_validation_report(
    path: Path,
    rows: int,
    issues: list[dict],
    *,
    distribution: dict[str, Any] | None = None,
    quota_issues: list[dict[str, Any]] | None = None,
) -> dict:
    invalid_rows = {(issue["file"], issue["line"]) for issue in issues}
    quota_issues = quota_issues or []
    payload = {
        "rows": rows,
        "valid": rows - len(invalid_rows),
        "invalid": len(invalid_rows),
        "passed": len(invalid_rows) == 0 and len(quota_issues) == 0,
        "issues": issues,
        "distribution": distribution or {},
        "quota_failures": quota_issues,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        paths.extend(matches or [Path(pattern)])
    return [path for path in paths if path.exists()]


def _iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any], str]]:
    rows: list[tuple[int, dict[str, Any], str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                rows.append((line_number, {}, f"invalid JSONL: {exc.msg}"))
                continue
            if not isinstance(decoded, dict):
                rows.append((line_number, {}, "row must be a JSON object"))
                continue
            rows.append((line_number, decoded, ""))
    return rows


def _payload_safety_issues(label: str, payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    forbidden_key = find_forbidden_key(payload)
    if forbidden_key:
        issues.append(f"{label} has forbidden key: {forbidden_key}")
    forbidden_text = find_forbidden_text(payload)
    if forbidden_text:
        issues.append(f"{label} has forbidden {forbidden_text}")
    return issues


def _is_safe_error(row: dict[str, Any]) -> bool:
    expected_json = row.get("expected_json", {})
    if isinstance(expected_json, dict) and compatibility_intent(expected_json) == "safe_error":
        return True
    react_ir = row.get("react_ir", {})
    if isinstance(react_ir, dict) and react_intent(react_ir) == "safe_error":
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
