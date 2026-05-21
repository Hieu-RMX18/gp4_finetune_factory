#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from benchmark_report_contract import (
    BENCHMARK_COLUMNS,
    BENCHMARK_MARKDOWN_HEADER,
    CHART_KEY_ACCEPTANCE_GATE_STATUS,
    CHART_KEY_ACTUAL_VS_THRESHOLD,
    CHART_KEY_SCENARIO_TAG_DISTRIBUTION,
    CHART_KEY_V2_QUOTA_FAILURES,
    REPORT_SCHEMA_VERSION,
)
from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import read_json, read_yaml, write_json

EXPECTED_SOURCE_BRANCH = "ws-deep-rebuild-3526"
MIN_EXPECTED_COMMIT_LENGTH = 12
EXPECTED_DRIVE_ACCOUNT_EMAIL = "johnwickiller4444@gmail.com"
ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"
PROVIDER_POLICY = ROOT / "configs/provider_policy.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a GP4 cloud workflow quality report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--html-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--input-report", action="append", default=[])
    parser.add_argument("--distribution-spec", type=Path)
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    input_reports = [Path(path) for path in args.input_report]
    if not input_reports:
        print("quality_report_blocked reason=at least one --input-report is required")
        return 1
    outputs = [args.report]
    if args.html_report:
        outputs.append(args.html_report)
    if args.markdown_report:
        outputs.append(args.markdown_report)
    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=False,
            inputs=input_reports,
            outputs=outputs,
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"quality_report_blocked reason={exc} report={args.report}")
        return 1

    quota_minimums = _quota_minimums(args.distribution_spec)
    payload = build_quality_report(
        input_reports,
        quota_minimums=quota_minimums,
        cloud_root=args.cloud_root,
        distribution_spec=args.distribution_spec,
    )
    if args.html_report:
        payload["html_report"] = str(args.html_report)
        write_html_report(args.html_report, payload)
    if args.markdown_report:
        payload["markdown_report"] = str(args.markdown_report)
        write_markdown_report(args.markdown_report, payload)
    write_json(args.report, payload)
    print(f"passed={payload['passed']} report={args.report}")
    return 0 if payload["passed"] else 1


def build_quality_report(
    paths: list[Path],
    *,
    quota_minimums: dict[str, int] | None = None,
    cloud_root: Path | None = None,
    distribution_spec: Path | None = None,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    reject_reasons: Counter[str] = Counter()
    scenario_tags: Counter[str] = Counter()
    benchmark_rows: list[dict[str, Any]] = []
    deferred_benchmark_rows: list[dict[str, Any]] = []
    passed = True
    for path in paths:
        report = read_json(path)
        source = str(path)
        reports[source] = report
        if report.get("passed") is False:
            passed = False
        for issue in report.get("issues", []):
            if isinstance(issue, dict):
                reject_reasons[str(issue.get("message", "unknown"))] += 1
        rows = _benchmark_rows(source, report, quota_minimums=quota_minimums)
        if _is_reuse_or_drive_evidence_report(source, report):
            deferred_benchmark_rows.extend(rows)
        else:
            benchmark_rows.extend(rows)
            scenario_tags.update(_scenario_tag_counts(report))
    benchmark_rows.extend(deferred_benchmark_rows)
    if any(row.get("passed") is False for row in benchmark_rows):
        passed = False
    distribution = {
        "scenario_tags": dict(sorted(scenario_tags.items())),
    }
    return {
        "passed": passed,
        "reports": reports,
        "reject_reasons": dict(reject_reasons),
        "distribution": distribution,
        "benchmark_columns": list(BENCHMARK_COLUMNS),
        "benchmark_rows": benchmark_rows,
        "charts": _chart_payload(benchmark_rows, scenario_tags),
        "provenance": _provenance_payload(
            paths=paths,
            reports=reports,
            cloud_root=cloud_root,
            distribution_spec=distribution_spec,
        ),
        "next_batch_feedback": sorted(reject_reasons),
    }


def _provenance_payload(
    *,
    paths: list[Path],
    reports: dict[str, Any],
    cloud_root: Path | None,
    distribution_spec: Path | None,
) -> dict[str, Any]:
    contract_report = _first_report_with(reports, "contract")
    contract = contract_report.get("contract", {}) if contract_report else {}
    if not isinstance(contract, dict):
        contract = {}
    contract_manifest_path = (
        cloud_root / "manifests/contract_manifest.json"
        if cloud_root is not None
        else None
    )
    contract_manifest = (
        _read_json_if_present(contract_manifest_path)
        if contract_manifest_path is not None
        else {}
    )

    local_install_path, local_install = _first_report_path_with(
        reports,
        ("ready_for_local_install", "install_action_performed"),
    )
    target_state = local_install.get("target_repo_state", {})
    if not isinstance(target_state, dict):
        target_state = {}

    package_report = _first_report_with(reports, "adapter_dir")
    adapter_dir = _adapter_dir_from_reports(package_report, local_install)

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "cloud_root": str(cloud_root) if cloud_root is not None else "",
        "input_reports": {
            str(path): _file_fingerprint(path)
            for path in paths
        },
        "dataset_spec": _file_fingerprint(distribution_spec or DATASET_SPEC),
        "provider_policy": _file_fingerprint(PROVIDER_POLICY),
        "contract": {
            "source": contract.get("source"),
            "repo_path": contract.get("repo_path"),
            "branch": contract.get("branch"),
            "head": contract.get("head"),
            "manifest": _file_fingerprint(contract_manifest_path),
            "manifest_hashes": (
                contract_manifest.get("hashes", {})
                if isinstance(contract_manifest.get("hashes"), dict)
                else {}
            ),
        },
        "local_install": {
            "manifest": _file_fingerprint(Path(local_install_path)) if local_install_path else {},
            "target_repo": local_install.get("target_repo"),
            "target_repo_expected_branch": target_state.get("expected_branch"),
            "target_repo_current_branch": target_state.get("current_branch"),
            "target_repo_head": target_state.get("head"),
            "target_repo_expected_commit": target_state.get("expected_commit"),
            "adapter_path": _adapter_path_from_local_install(local_install),
        },
        "drive_account": _drive_account_provenance(reports),
        "old_dataset_reuse": _old_dataset_reuse_provenance(reports),
        "adapter": _adapter_provenance(adapter_dir),
    }


def _first_report_with(
    reports: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    for report in reports.values():
        if isinstance(report, dict) and key in report:
            return report
    return {}


def _first_report_path_with(
    reports: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    for path, report in reports.items():
        if isinstance(report, dict) and any(key in report for key in keys):
            return path, report
    return "", {}


def _adapter_dir_from_reports(
    package_report: dict[str, Any],
    local_install: dict[str, Any],
) -> Path | None:
    if package_report.get("adapter_dir"):
        return Path(str(package_report["adapter_dir"]))
    adapter_path = _adapter_path_from_local_install(local_install)
    return Path(adapter_path) if adapter_path else None


def _adapter_path_from_local_install(local_install: dict[str, Any]) -> str:
    adapter = local_install.get("adapter", {})
    if not isinstance(adapter, dict):
        return ""
    return str(adapter.get("path") or "")


def _adapter_provenance(adapter_dir: Path | None) -> dict[str, Any]:
    if adapter_dir is None or not adapter_dir.is_dir():
        return {
            "path": str(adapter_dir) if adapter_dir is not None else "",
            "file_count": 0,
            "total_bytes": 0,
            "aggregate_sha256": "",
            "files": {},
        }
    files: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(adapter_dir).as_posix()
        fingerprint = _file_fingerprint(path)
        files[relative] = fingerprint
        total_bytes += int(fingerprint.get("bytes", 0))
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(str(fingerprint.get("sha256", "")).encode("utf-8"))
    return {
        "path": str(adapter_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest() if files else "",
        "files": files,
    }


def _file_fingerprint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    return payload


def _drive_account_provenance(reports: dict[str, Any]) -> dict[str, Any]:
    report = _first_report_with(reports, "drive_account_hint")
    readiness_report = _first_report_with(reports, "drive_account")
    readiness_drive_account = readiness_report.get("drive_account", {})
    if not isinstance(readiness_drive_account, dict):
        readiness_drive_account = {}
    email = _drive_account_email_from_report(report)
    hint_path = _drive_account_hint_path(report)
    confirmed = readiness_drive_account.get("confirmed") is True
    confirmed_email = str(readiness_drive_account.get("confirmed_email") or "").strip()
    return {
        "email": email,
        "expected": EXPECTED_DRIVE_ACCOUNT_EMAIL,
        "confirmed": confirmed,
        "confirmed_email": confirmed_email,
        "matches_expected": (
            email == EXPECTED_DRIVE_ACCOUNT_EMAIL
            and confirmed
            and confirmed_email == EXPECTED_DRIVE_ACCOUNT_EMAIL
        ),
        "hint_path": str(hint_path) if hint_path is not None else "",
        "hint_file": _file_fingerprint(hint_path) if hint_path is not None else {},
    }


def _old_dataset_reuse_provenance(reports: dict[str, Any]) -> dict[str, Any]:
    import_report = _first_report_with(reports, "old_dataset_count")
    validate_report = _first_report_with(reports, "old_rows_valid")
    plan_report = _first_report_with(reports, "new_rows_requested")
    merge_report = _first_report_with(reports, "old_rows_kept")
    fingerprints = import_report.get("old_dataset_fingerprints", [])
    if not isinstance(fingerprints, list):
        fingerprints = []
    return {
        "old_dataset_count": _int_value(import_report.get("old_dataset_count")),
        "old_dataset_fingerprints": fingerprints,
        "old_rows_valid": _int_value(validate_report.get("old_rows_valid")),
        "old_validated_merged_path": str(
            validate_report.get("old_validated_merged_path") or ""
        ),
        "target_rows": _int_value(plan_report.get("target_rows")),
        "new_rows_requested": _int_value(plan_report.get("new_rows_requested")),
        "old_rows_input": _int_value(merge_report.get("old_rows_input")),
        "new_rows_input": _int_value(merge_report.get("new_rows_input")),
        "old_rows_kept": _int_value(merge_report.get("old_rows_kept")),
        "new_rows_kept": _int_value(merge_report.get("new_rows_kept")),
        "output_rows": _int_value(merge_report.get("output_rows")),
    }


def _drive_account_email_from_report(report: dict[str, Any]) -> str:
    hint_path = _drive_account_hint_path(report)
    if hint_path is not None and hint_path.is_file():
        try:
            return hint_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    raw_hint = report.get("drive_account_hint", "")
    return str(raw_hint).strip()


def _drive_account_hint_path(report: dict[str, Any]) -> Path | None:
    raw_hint = report.get("drive_account_hint")
    if not raw_hint:
        return None
    return Path(str(raw_hint))


def _benchmark_rows(
    source: str,
    report: dict[str, Any],
    *,
    quota_minimums: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_drive_account_rows(source, report))
    rows.extend(_old_dataset_reuse_rows(source, report))
    rows.extend(_provider_rows(source, report))
    rows.extend(_eval_contract_rows(source, report))
    rows.extend(_local_install_rows(source, report))
    checks = report.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            rows.append(
                {
                    "source": source,
                    "metric": str(check.get("metric") or check.get("gate") or "unknown"),
                    "actual": check.get("actual"),
                    "operator": check.get("operator"),
                    "threshold": check.get("threshold"),
                    "passed": check.get("passed") is True,
                }
            )
    quota_minimums = quota_minimums or {}
    scenario_tags = _scenario_tag_counts(report)
    if (
        quota_minimums
        and scenario_tags
        and not _is_reuse_or_drive_evidence_report(source, report)
    ):
        for tag, minimum in sorted(quota_minimums.items()):
            actual = int(scenario_tags.get(tag, 0))
            rows.append(
                {
                    "source": source,
                    "metric": f"v2_{tag}_rows",
                    "actual": actual,
                    "operator": ">=",
                    "threshold": minimum,
                    "passed": actual >= minimum,
                }
            )
        return rows

    quota_failures = report.get("quota_failures", [])
    if isinstance(quota_failures, list):
        for failure in quota_failures:
            if not isinstance(failure, dict):
                continue
            tag = str(failure.get("tag") or "unknown")
            rows.append(
                {
                    "source": source,
                    "metric": f"v2_{tag}_rows",
                    "actual": failure.get("actual"),
                    "operator": ">=",
                    "threshold": failure.get("minimum"),
                    "passed": False,
                }
            )
    return rows


def _is_reuse_or_drive_evidence_report(source: str, report: dict[str, Any]) -> bool:
    source_name = Path(source).name
    if source_name.startswith(
        (
            "cloud-setup_",
            "import-old_",
            "validate-old-v2_",
            "plan-v2-target_",
            "merge-accepted_",
        )
    ):
        return True
    return any(
        key in report
        for key in (
            "drive_account_hint",
            "old_dataset_count",
            "old_rows_valid",
            "new_rows_requested",
            "old_rows_kept",
        )
    )


def _drive_account_rows(source: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    if "drive_account_hint" not in report:
        return []
    return [
        _row(
            source,
            "drive_account_hint",
            _drive_account_email_from_report(report),
            "==",
            EXPECTED_DRIVE_ACCOUNT_EMAIL,
        )
    ]


def _old_dataset_reuse_rows(source: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "old_dataset_count" in report:
        rows.append(_row(source, "old_dataset_count", _int_value(report.get("old_dataset_count")), ">=", 1))
    if "old_rows_valid" in report:
        rows.append(_row(source, "old_rows_valid", _int_value(report.get("old_rows_valid")), ">=", 1))
    if "new_rows_requested" in report:
        target_rows = _int_value(report.get("target_rows"))
        rows.append(
            _row(
                source,
                "new_rows_requested",
                _int_value(report.get("new_rows_requested")),
                "<=",
                target_rows,
            )
        )
    if "output_rows" in report:
        rows.append(_row(source, "accepted_300k_rows", _int_value(report.get("output_rows")), ">=", 300000))
    if "old_rows_kept" in report:
        rows.append(_row(source, "old_rows_kept", _int_value(report.get("old_rows_kept")), ">=", 1))
    if "new_rows_kept" in report:
        rows.append(_row(source, "new_rows_kept", _int_value(report.get("new_rows_kept")), ">=", 1))
    return rows


def _provider_rows(source: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    if "provider" not in report and "cloud_storage_ready" not in report:
        return []
    return [
        _row(source, "provider_available", report.get("available"), "is", True),
        _row(
            source,
            "provider_cloud_storage_ready",
            report.get("cloud_storage_ready"),
            "is",
            True,
        ),
        _row(source, "provider_free_tier", report.get("free_tier"), "is", True),
        _row(source, "provider_paid_risk", report.get("paid_risk"), "is", False),
        _row(
            source,
            "provider_account_creation_automation",
            report.get("account_creation_automation", False),
            "is",
            False,
        ),
        _row(
            source,
            "provider_quota_bypass_attempt",
            report.get("quota_bypass_attempt", False),
            "is",
            False,
        ),
        _row(
            source,
            "provider_idle_bypass_attempt",
            report.get("idle_bypass_attempt", False),
            "is",
            False,
        ),
    ]


def _eval_contract_rows(source: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    contract = report.get("contract", {})
    if not isinstance(contract, dict) or not contract:
        return []
    head = str(contract.get("head", ""))
    return [
        _row(source, "eval_contract_source", contract.get("source"), "==", "repo"),
        _row(
            source,
            "eval_contract_branch",
            contract.get("branch"),
            "==",
            EXPECTED_SOURCE_BRANCH,
        ),
        _row(source, "eval_contract_head_present", bool(head), "is", True),
    ]


def _local_install_rows(source: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    if "ready_for_local_install" not in report and "install_action_performed" not in report:
        return []
    target_state = report.get("target_repo_state", {})
    if not isinstance(target_state, dict):
        target_state = {}
    head = str(target_state.get("head_full") or target_state.get("head") or "")
    expected_commit = str(target_state.get("expected_commit", ""))
    return [
        _row(
            source,
            "ready_for_local_install",
            report.get("ready_for_local_install"),
            "is",
            True,
        ),
        _row(
            source,
            "install_action_performed",
            report.get("install_action_performed"),
            "is",
            False,
        ),
        _row(source, "target_repo_exists", target_state.get("exists"), "is", True),
        _row(
            source,
            "target_repo_branch",
            target_state.get("current_branch"),
            "==",
            target_state.get("expected_branch") or EXPECTED_SOURCE_BRANCH,
        ),
        _row(source, "target_repo_head_present", bool(head), "is", True),
        _row(source, "target_repo_commit", head, "matches", expected_commit),
        _row(
            source,
            "target_repo_expected_commit_matches",
            target_state.get("expected_commit_matches"),
            "is",
            True,
        ),
        _row(source, "target_repo_dirty", target_state.get("is_dirty"), "is", False),
    ]


def _row(
    source: str,
    metric: str,
    actual: Any,
    operator: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "source": source,
        "metric": metric,
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "passed": _compare(actual, operator, threshold),
    }


def _compare(actual: Any, operator: str, threshold: Any) -> bool:
    if operator in {"is", "=="}:
        return actual == threshold
    if operator == ">=":
        return actual >= threshold
    if operator == "<=":
        return actual <= threshold
    if operator == "matches":
        return _commit_matches(actual, threshold)
    return False


def _commit_matches(actual: Any, threshold: Any) -> bool:
    actual_commit = str(actual or "").strip().lower()
    expected_commit = str(threshold or "").strip().lower()
    if not actual_commit or len(expected_commit) < MIN_EXPECTED_COMMIT_LENGTH:
        return False
    return actual_commit.startswith(expected_commit)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _quota_minimums(spec_path: Path | None) -> dict[str, int]:
    if spec_path is None:
        return {}
    spec = read_yaml(spec_path)
    gates = spec.get("v2_distribution_gates", {})
    if not isinstance(gates, dict):
        return {}
    minimums = gates.get("scenario_tag_min_counts", {})
    if not isinstance(minimums, dict):
        return {}
    return {str(tag): int(value) for tag, value in minimums.items()}


def _scenario_tag_counts(report: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    distribution = report.get("distribution", {})
    if not isinstance(distribution, dict):
        return counts
    scenario_tags = distribution.get("scenario_tags", {})
    if not isinstance(scenario_tags, dict):
        return counts
    for tag, count in scenario_tags.items():
        if isinstance(count, bool):
            continue
        try:
            counts[str(tag)] += int(count)
        except (TypeError, ValueError):
            continue
    return counts


def _chart_payload(
    benchmark_rows: list[dict[str, Any]],
    scenario_tags: Counter[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        CHART_KEY_ACTUAL_VS_THRESHOLD: [
            {
                "label": row["metric"],
                "actual": row["actual"],
                "threshold": row["threshold"],
                "passed": row["passed"],
            }
            for row in benchmark_rows
            if _is_number(row.get("actual")) and _is_number(row.get("threshold"))
        ],
        CHART_KEY_ACCEPTANCE_GATE_STATUS: [
            {
                "label": row["metric"],
                "value": 1 if row["passed"] else 0,
            }
            for row in benchmark_rows
        ],
        CHART_KEY_V2_QUOTA_FAILURES: _quota_failure_chart_rows(benchmark_rows),
        CHART_KEY_SCENARIO_TAG_DISTRIBUTION: [
            {"label": tag, "value": count}
            for tag, count in sorted(scenario_tags.items())
        ],
    }


def _quota_failure_chart_rows(
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        metric = str(row.get("metric") or "")
        if (
            not metric.startswith("v2_")
            or not metric.endswith("_rows")
            or row.get("operator") != ">="
            or row.get("passed") is True
            or not _is_number(row.get("actual"))
            or not _is_number(row.get("threshold"))
        ):
            continue
        actual = row["actual"]
        threshold = row["threshold"]
        deficit = max(float(threshold) - float(actual), 0)
        rows.append(
            {
                "label": metric.removeprefix("v2_").removesuffix("_rows"),
                "actual": actual,
                "threshold": threshold,
                "deficit": int(deficit) if deficit.is_integer() else deficit,
                "passed": False,
            }
        )
    return rows


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def write_html_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(payload), encoding="utf-8")


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(payload), encoding="utf-8")


def render_markdown_report(payload: dict[str, Any]) -> str:
    benchmark_rows = payload.get("benchmark_rows", [])
    distribution = payload.get("distribution", {})
    reports = payload.get("reports", {})
    if not isinstance(benchmark_rows, list):
        benchmark_rows = []
    if not isinstance(distribution, dict):
        distribution = {}
    if not isinstance(reports, dict):
        reports = {}
    scenario_tags = distribution.get("scenario_tags", {})
    if not isinstance(scenario_tags, dict):
        scenario_tags = {}
    failed_metrics = [
        row for row in benchmark_rows
        if isinstance(row, dict) and row.get("passed") is not True
    ]
    next_feedback = payload.get("next_batch_feedback", [])
    if not isinstance(next_feedback, list):
        next_feedback = []

    lines = [
        "# GP4 V2 Benchmark Report",
        "",
        f"- passed: {payload.get('passed') is True}",
        f"- input reports: {len(reports)}",
        f"- benchmark rows: {len([row for row in benchmark_rows if isinstance(row, dict)])}",
        f"- failed benchmark rows: {len(failed_metrics)}",
        "",
        "## Benchmark Columns",
        "",
        BENCHMARK_MARKDOWN_HEADER,
        "| " + " | ".join("---" for _ in BENCHMARK_COLUMNS) + " |",
    ]
    for row in benchmark_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column, "")) for column in BENCHMARK_COLUMNS)
            + " |"
        )
    if not any(isinstance(row, dict) for row in benchmark_rows):
        lines.append("|  |  |  |  |  |  |")

    quota_failures = []
    charts = payload.get("charts", {})
    if isinstance(charts, dict):
        raw_quota_failures = charts.get(CHART_KEY_V2_QUOTA_FAILURES, [])
        if isinstance(raw_quota_failures, list):
            quota_failures = [
                row for row in raw_quota_failures if isinstance(row, dict)
            ]
    lines.extend(["", "## V2 Quota Failures", ""])
    if quota_failures:
        for row in quota_failures:
            lines.append(
                "- "
                f"{row.get('label', 'unknown')}: actual={row.get('actual')} "
                f"threshold={row.get('threshold')} deficit={row.get('deficit')}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Scenario Tag Distribution", ""])
    if scenario_tags:
        for tag, count in sorted(scenario_tags.items()):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Failed Metrics", ""])
    if failed_metrics:
        for row in failed_metrics:
            lines.append(
                "- "
                f"{row.get('metric', 'unknown')}: actual={row.get('actual')} "
                f"{row.get('operator', '')} threshold={row.get('threshold')}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Input Reports", ""])
    for source in sorted(reports):
        report = reports[source]
        passed = report.get("passed") if isinstance(report, dict) else "unknown"
        lines.append(f"- {source}: passed={passed}")
    if not reports:
        lines.append("- none")

    lines.extend(["", "## Provenance", ""])
    lines.extend(_markdown_provenance_lines(payload.get("provenance", {})))

    lines.extend(["", "## Next Batch Feedback", ""])
    if next_feedback:
        for item in next_feedback:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Maintenance Reference",
            "",
            "- Compare this report with the next run before changing quotas or gates.",
            "- Keep anti-hallucination and dangerous OS command rows at or above their thresholds.",
            "- Treat failed rows as generation feedback, not as acceptance evidence.",
        ]
    )
    for row in _maintenance_reference_rows(payload.get("provenance", {})):
        lines.append(f"- {row['key']}: {row['value']}")
    lines.append("")
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_provenance_lines(provenance: Any) -> list[str]:
    if not isinstance(provenance, dict):
        return ["- none"]
    contract = provenance.get("contract", {})
    if not isinstance(contract, dict):
        contract = {}
    contract_manifest = contract.get("manifest", {})
    if not isinstance(contract_manifest, dict):
        contract_manifest = {}
    local_install = provenance.get("local_install", {})
    if not isinstance(local_install, dict):
        local_install = {}
    local_manifest = local_install.get("manifest", {})
    if not isinstance(local_manifest, dict):
        local_manifest = {}
    adapter = provenance.get("adapter", {})
    if not isinstance(adapter, dict):
        adapter = {}
    dataset_spec = provenance.get("dataset_spec", {})
    if not isinstance(dataset_spec, dict):
        dataset_spec = {}
    provider_policy = provenance.get("provider_policy", {})
    if not isinstance(provider_policy, dict):
        provider_policy = {}
    drive_account = provenance.get("drive_account", {})
    if not isinstance(drive_account, dict):
        drive_account = {}
    old_dataset_reuse = provenance.get("old_dataset_reuse", {})
    if not isinstance(old_dataset_reuse, dict):
        old_dataset_reuse = {}
    lines = [
        f"- report_schema_version: {provenance.get('report_schema_version', '')}",
        f"- generated_at_utc: {provenance.get('generated_at_utc', '')}",
        f"- cloud_root: {provenance.get('cloud_root', '')}",
        f"- dataset_spec_sha256: {dataset_spec.get('sha256', '')}",
        f"- provider_policy_sha256: {provider_policy.get('sha256', '')}",
        f"- contract_branch: {contract.get('branch', '')}",
        f"- contract_head: {contract.get('head', '')}",
        f"- contract_manifest_sha256: {contract_manifest.get('sha256', '')}",
        f"- local_install_manifest_sha256: {local_manifest.get('sha256', '')}",
        f"- target_repo_head: {local_install.get('target_repo_head', '')}",
        f"- target_repo_expected_commit: {local_install.get('target_repo_expected_commit', '')}",
        f"- drive_account_email: {drive_account.get('email', '')}",
        f"- drive_account_expected: {drive_account.get('expected', '')}",
        f"- drive_account_confirmed: {drive_account.get('confirmed', False)}",
        f"- drive_account_confirmed_email: {drive_account.get('confirmed_email', '')}",
        f"- drive_account_matches: {drive_account.get('matches_expected', False)}",
        f"- old_dataset_count: {old_dataset_reuse.get('old_dataset_count', 0)}",
        f"- old_rows_valid: {old_dataset_reuse.get('old_rows_valid', 0)}",
        f"- old_rows_kept: {old_dataset_reuse.get('old_rows_kept', 0)}",
        f"- new_rows_requested: {old_dataset_reuse.get('new_rows_requested', 0)}",
        f"- new_rows_kept: {old_dataset_reuse.get('new_rows_kept', 0)}",
        f"- adapter_path: {adapter.get('path', '')}",
        f"- adapter_file_count: {adapter.get('file_count', 0)}",
        f"- adapter_total_bytes: {adapter.get('total_bytes', 0)}",
        f"- adapter_aggregate_sha256: {adapter.get('aggregate_sha256', '')}",
    ]
    return lines


def render_html_report(payload: dict[str, Any]) -> str:
    benchmark_rows = payload.get("benchmark_rows", [])
    charts = payload.get("charts", {})
    if not isinstance(benchmark_rows, list):
        benchmark_rows = []
    if not isinstance(charts, dict):
        charts = {}

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>GP4 V2 Benchmark Report</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;line-height:1.45;color:#17202a}",
            "table{border-collapse:collapse;width:100%;margin:16px 0}",
            "th,td{border:1px solid #ccd6dd;padding:8px;text-align:left}",
            "th{background:#edf2f7}",
            ".passed{color:#0f6b3f;font-weight:700}.failed{color:#9b1c1c;font-weight:700}",
            ".chart{margin:20px 0}.bar-ok{fill:#2f855a}.bar-bad{fill:#c53030}.bar-ref{fill:#718096}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>GP4 V2 Benchmark Report</h1>",
            "<h2>Benchmark Columns</h2>",
            _benchmark_table(benchmark_rows),
            "<h2>Actual vs Threshold</h2>",
            _bar_chart(
                charts.get(CHART_KEY_ACTUAL_VS_THRESHOLD, []),
                value_key="actual",
                reference_key="threshold",
            ),
            "<h2>Acceptance Gate Status</h2>",
            _bar_chart(charts.get(CHART_KEY_ACCEPTANCE_GATE_STATUS, []), value_key="value"),
            "<h2>V2 Quota Failures</h2>",
            _bar_chart(charts.get(CHART_KEY_V2_QUOTA_FAILURES, []), value_key="deficit"),
            "<h2>Scenario Tag Distribution</h2>",
            _bar_chart(charts.get(CHART_KEY_SCENARIO_TAG_DISTRIBUTION, []), value_key="value"),
            "<h2>Provenance</h2>",
            _provenance_table(payload.get("provenance", {})),
            "<h2>Maintenance Reference</h2>",
            _maintenance_reference_table(payload.get("provenance", {})),
            "</body>",
            "</html>",
        ]
    )


def _benchmark_table(rows: list[Any]) -> str:
    header = "".join(f"<th>{escape(column)}</th>" for column in BENCHMARK_COLUMNS)
    body_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cells = "".join(
            f"<td>{escape(str(row.get(column, '')))}</td>"
            for column in BENCHMARK_COLUMNS
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _provenance_table(provenance: Any) -> str:
    rows = _markdown_provenance_lines(provenance)
    body = []
    for row in rows:
        key, _, value = row.lstrip("- ").partition(": ")
        body.append(
            "<tr>"
            f"<th>{escape(key)}</th>"
            f"<td>{escape(value)}</td>"
            "</tr>"
        )
    return f"<table><tbody>{''.join(body)}</tbody></table>"


def _maintenance_reference_table(provenance: Any) -> str:
    rows = _maintenance_reference_rows(provenance)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<th>{escape(str(row['key']))}</th>"
            f"<td>{escape(str(row['value']))}</td>"
            "</tr>"
        )
    return f"<table><tbody>{''.join(body)}</tbody></table>"


def _maintenance_reference_rows(provenance: Any) -> list[dict[str, Any]]:
    if not isinstance(provenance, dict):
        provenance = {}
    local_install = provenance.get("local_install", {})
    if not isinstance(local_install, dict):
        local_install = {}
    adapter = provenance.get("adapter", {})
    if not isinstance(adapter, dict):
        adapter = {}
    drive_account = provenance.get("drive_account", {})
    if not isinstance(drive_account, dict):
        drive_account = {}
    old_dataset_reuse = provenance.get("old_dataset_reuse", {})
    if not isinstance(old_dataset_reuse, dict):
        old_dataset_reuse = {}
    return [
        {"key": "report_schema_version", "value": provenance.get("report_schema_version", "")},
        {"key": "cloud_root", "value": provenance.get("cloud_root", "")},
        {"key": "drive_account_confirmed", "value": drive_account.get("confirmed", False)},
        {"key": "drive_account_matches", "value": drive_account.get("matches_expected", False)},
        {"key": "old_dataset_count", "value": old_dataset_reuse.get("old_dataset_count", 0)},
        {"key": "old_rows_valid", "value": old_dataset_reuse.get("old_rows_valid", 0)},
        {"key": "old_rows_kept", "value": old_dataset_reuse.get("old_rows_kept", 0)},
        {"key": "new_rows_requested", "value": old_dataset_reuse.get("new_rows_requested", 0)},
        {"key": "new_rows_kept", "value": old_dataset_reuse.get("new_rows_kept", 0)},
        {
            "key": "target_repo_expected_commit",
            "value": local_install.get("target_repo_expected_commit", ""),
        },
        {"key": "adapter_aggregate_sha256", "value": adapter.get("aggregate_sha256", "")},
    ]


def _bar_chart(
    rows: Any,
    *,
    value_key: str,
    reference_key: str | None = None,
) -> str:
    if not isinstance(rows, list) or not rows:
        return '<p class="chart">No chart data available.</p>'
    numeric_rows = [
        row for row in rows
        if isinstance(row, dict) and _is_number(row.get(value_key))
    ]
    if not numeric_rows:
        return '<p class="chart">No chart data available.</p>'
    max_value = max(float(row[value_key]) for row in numeric_rows)
    if reference_key:
        refs = [
            float(row[reference_key])
            for row in numeric_rows
            if _is_number(row.get(reference_key))
        ]
        if refs:
            max_value = max(max_value, *refs)
    max_value = max(max_value, 1.0)
    width = 720
    row_height = 30
    label_width = 220
    chart_width = width - label_width - 40
    height = 30 + row_height * len(numeric_rows)
    parts = [f'<svg class="chart" width="{width}" height="{height}" role="img">']
    for index, row in enumerate(numeric_rows):
        y = 20 + index * row_height
        label = escape(str(row.get("label", "unknown")))
        value = float(row[value_key])
        bar_width = int((value / max_value) * chart_width)
        css_class = "bar-ok" if row.get("passed", True) else "bar-bad"
        parts.append(f'<text x="0" y="{y + 14}" font-size="12">{label}</text>')
        parts.append(
            f'<rect class="{css_class}" x="{label_width}" y="{y}" '
            f'width="{bar_width}" height="18"></rect>'
        )
        parts.append(
            f'<text x="{label_width + bar_width + 6}" y="{y + 14}" '
            f'font-size="12">{escape(str(row[value_key]))}</text>'
        )
        if reference_key and _is_number(row.get(reference_key)):
            ref_x = label_width + int((float(row[reference_key]) / max_value) * chart_width)
            parts.append(
                f'<line class="bar-ref" x1="{ref_x}" x2="{ref_x}" '
                f'y1="{y - 2}" y2="{y + 20}" stroke="#718096"></line>'
            )
    parts.append("</svg>")
    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
