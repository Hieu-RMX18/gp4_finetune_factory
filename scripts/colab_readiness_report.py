#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from cloud_runtime import (
    CloudPathError,
    sha256_file,
    validate_cloud_output_path,
)
from factory_common import git_value, read_yaml, write_json
from package_adapter import adapter_artifact_exists

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"
EXPECTED_DRIVE_ACCOUNT_EMAIL = "johnwickiller4444@gmail.com"
MIN_EXPECTED_COMMIT_LENGTH = 12


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a Colab/Drive preflight readiness report."
    )
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gp4-ws", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--previous-adapter", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    report_path = args.report or (
        args.cloud_root / "reports" / f"colab_readiness_{args.run_id}.json"
    )
    try:
        validate_cloud_output_path(
            report_path,
            args.cloud_root,
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"colab_readiness_blocked reason={exc} report={report_path}")
        return 1

    policy = CloudStoragePolicy(
        (args.cloud_root, args.cloud_root.parent),
        allow_tmp=args.allow_tmp,
    )
    payload = build_colab_readiness_report(
        cloud_root=args.cloud_root,
        run_id=args.run_id,
        gp4_ws=args.gp4_ws,
        old_dataset=args.old_dataset,
        previous_adapter=args.previous_adapter,
        expected_commit=args.expected_commit,
        policy=policy,
        report=report_path,
    )
    write_json(report_path, payload)
    print(f"passed={payload['passed']} report={report_path}")
    return 0 if payload["passed"] else 1


def build_colab_readiness_report(
    *,
    cloud_root: Path,
    run_id: str,
    gp4_ws: Path,
    old_dataset: Path,
    previous_adapter: Path | None = None,
    expected_commit: str | None = None,
    policy: CloudStoragePolicy,
    report: Path | None = None,
) -> dict[str, Any]:
    spec = read_yaml(DATASET_SPEC)
    project = spec.get("project", {})
    if not isinstance(project, dict):
        project = {}
    expected_branch = str(project.get("source_branch", ""))
    expected_commit = _configured_expected_commit(
        expected_commit,
        env_name=str(project.get("source_commit_env") or "GP4_WS_EXPECTED_COMMIT"),
    )
    drive_hint_path = cloud_root / "manifests/drive_account_hint.txt"
    drive_confirmation_path = cloud_root / "manifests/drive_account_confirmation.json"
    drive_email = _read_text(drive_hint_path).strip()
    drive_confirmation = _read_optional_json(drive_confirmation_path)
    confirmed_email = str(drive_confirmation.get("confirmed_email") or "").strip()
    drive_account_confirmed = (
        drive_confirmation.get("confirmed") is True
        and confirmed_email == EXPECTED_DRIVE_ACCOUNT_EMAIL
        and str(drive_confirmation.get("expected_email") or "").strip()
        == EXPECTED_DRIVE_ACCOUNT_EMAIL
    )
    old_dataset_exists = old_dataset.is_file()
    old_dataset_rows = _jsonl_rows(old_dataset) if old_dataset_exists else 0
    old_dataset_sha = sha256_file(old_dataset) if old_dataset_exists else ""
    gp4_ws_policy = CloudStoragePolicy(
        (cloud_root / "contract_snapshots",),
        allow_tmp=False,
    )
    previous_adapter_allowed = (
        is_allowed_cloud_path(previous_adapter, policy)
        if previous_adapter is not None
        else False
    )
    previous_adapter_exists = (
        previous_adapter.is_dir() if previous_adapter is not None else False
    )
    previous_adapter_artifact_exists = (
        adapter_artifact_exists(previous_adapter)
        if previous_adapter is not None
        else False
    )
    previous_run = _previous_run_reuse_state(
        cloud_root=cloud_root,
        run_id=run_id,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
    )
    gp4_ws_exists = gp4_ws.exists()
    gp4_ws_allowed = is_allowed_cloud_path(gp4_ws, gp4_ws_policy)
    gp4_ws_branch = git_value(gp4_ws, "branch", "--show-current") if gp4_ws_exists else ""
    gp4_ws_head = git_value(gp4_ws, "rev-parse", "HEAD") if gp4_ws_exists else ""
    gp4_ws_dirty = bool(git_value(gp4_ws, "status", "--short")) if gp4_ws_exists else False

    checks = [
        _check(
            "cloud_root_allowed",
            is_allowed_cloud_path(cloud_root, policy),
            str(cloud_root),
        ),
        _check(
            "drive_account_hint_matches",
            drive_email == EXPECTED_DRIVE_ACCOUNT_EMAIL,
            str(drive_hint_path),
        ),
        _check(
            "drive_account_confirmation_matches",
            drive_account_confirmed,
            str(drive_confirmation_path),
        ),
        _check("old_dataset_exists", old_dataset_exists, str(old_dataset)),
        _check(
            "old_dataset_allowed_cloud_path",
            is_allowed_cloud_path(old_dataset, policy),
            str(old_dataset),
        ),
        _check("old_dataset_has_rows", old_dataset_rows > 0, str(old_dataset)),
        _check(
            "previous_adapter_required",
            previous_adapter is not None,
            str(previous_adapter or ""),
        ),
        _check(
            "previous_adapter_allowed_cloud_path",
            previous_adapter_allowed,
            str(previous_adapter or ""),
        ),
        _check(
            "previous_adapter_exists",
            previous_adapter_exists,
            str(previous_adapter or ""),
        ),
        _check(
            "previous_adapter_artifact_exists",
            previous_adapter_artifact_exists,
            str(previous_adapter or ""),
        ),
        _check(
            "previous_reuse_same_prior_run",
            previous_run["matched"],
            previous_run["details"],
        ),
        _check("gp4_ws_exists", gp4_ws_exists, str(gp4_ws)),
        _check(
            "gp4_ws_allowed_cloud_path",
            gp4_ws_allowed,
            str(gp4_ws),
        ),
        _check(
            "gp4_ws_expected_branch",
            bool(expected_branch) and gp4_ws_branch == expected_branch,
            str(gp4_ws),
        ),
        _check("gp4_ws_has_commit", bool(gp4_ws_head), str(gp4_ws)),
        _check("gp4_ws_expected_commit_configured", bool(expected_commit), str(gp4_ws)),
        _check(
            "gp4_ws_expected_commit_matches",
            _commit_matches(gp4_ws_head, expected_commit),
            str(gp4_ws),
        ),
        _check("gp4_ws_clean", gp4_ws_dirty is False, str(gp4_ws)),
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "run_id": run_id,
        "cloud_root": str(cloud_root),
        "report": str(report) if report else "",
        "drive_account": {
            "hint_path": str(drive_hint_path),
            "confirmation_path": str(drive_confirmation_path),
            "email": drive_email,
            "confirmed_email": confirmed_email,
            "confirmed": drive_account_confirmed,
            "confirmation_method": str(drive_confirmation.get("method") or ""),
            "expected": EXPECTED_DRIVE_ACCOUNT_EMAIL,
            "matches_expected": (
                drive_email == EXPECTED_DRIVE_ACCOUNT_EMAIL
                and drive_account_confirmed
            ),
        },
        "old_dataset": {
            "path": str(old_dataset),
            "exists": old_dataset_exists,
            "allowed_cloud_path": is_allowed_cloud_path(old_dataset, policy),
            "rows": old_dataset_rows,
            "sha256": old_dataset_sha,
        },
        "previous_adapter": {
            "path": str(previous_adapter) if previous_adapter is not None else "",
            "exists": previous_adapter_exists,
            "allowed_cloud_path": previous_adapter_allowed,
            "artifact_exists": previous_adapter_artifact_exists,
        },
        "previous_run": previous_run,
        "gp4_ws": {
            "path": str(gp4_ws),
            "exists": gp4_ws_exists,
            "allowed_cloud_path": gp4_ws_allowed,
            "branch": gp4_ws_branch,
            "expected_branch": expected_branch,
            "head": gp4_ws_head,
            "expected_commit": expected_commit,
            "expected_commit_matches": _commit_matches(gp4_ws_head, expected_commit),
            "is_dirty": gp4_ws_dirty,
        },
        "install_action_performed": False,
        "checks": checks,
    }


def _previous_run_reuse_state(
    *,
    cloud_root: Path,
    run_id: str,
    old_dataset: Path,
    previous_adapter: Path | None,
) -> dict[str, Any]:
    drive_root = cloud_root.parent
    old_dataset_run_id = _previous_run_id_from_path(
        path=old_dataset,
        drive_root=drive_root,
        suffix=Path("data/validated/accepted_300k.jsonl"),
    )
    previous_adapter_run_id = (
        _previous_run_id_from_path(
            path=previous_adapter,
            drive_root=drive_root,
            suffix=Path("models/qwen25_gp4_lora"),
        )
        if previous_adapter is not None
        else ""
    )
    same_source_run = (
        bool(old_dataset_run_id)
        and bool(previous_adapter_run_id)
        and old_dataset_run_id == previous_adapter_run_id
    )
    is_prior_run = same_source_run and old_dataset_run_id != run_id
    matched = same_source_run and is_prior_run
    details = (
        f"old_dataset_run_id={old_dataset_run_id or '<unknown>'} "
        f"previous_adapter_run_id={previous_adapter_run_id or '<none>'} "
        f"current_run_id={run_id}"
    )
    return {
        "drive_root": str(drive_root),
        "old_dataset_run_id": old_dataset_run_id,
        "previous_adapter_run_id": previous_adapter_run_id,
        "same_source_run": same_source_run,
        "is_prior_run": is_prior_run,
        "matched": matched,
        "details": details,
    }


def _previous_run_id_from_path(
    *,
    path: Path | None,
    drive_root: Path,
    suffix: Path,
) -> str:
    if path is None:
        return ""
    try:
        relative = path.resolve(strict=False).relative_to(
            drive_root.resolve(strict=False)
        )
    except ValueError:
        return ""
    suffix_parts = suffix.parts
    if len(relative.parts) != len(suffix_parts) + 1:
        return ""
    if relative.parts[1:] != suffix_parts:
        return ""
    return relative.parts[0]


def _configured_expected_commit(raw_value: str | None, *, env_name: str) -> str:
    return (raw_value or os.environ.get(env_name) or "").strip()


def _commit_matches(head: str, expected_commit: str) -> bool:
    head = head.strip().lower()
    expected_commit = expected_commit.strip().lower()
    if not head or len(expected_commit) < MIN_EXPECTED_COMMIT_LENGTH:
        return False
    return head.startswith(expected_commit)


def _jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows += 1
    return rows


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_optional_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _check(check_id: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "evidence": evidence}


if __name__ == "__main__":
    raise SystemExit(main())
