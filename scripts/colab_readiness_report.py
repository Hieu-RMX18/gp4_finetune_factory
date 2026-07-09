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
from factory_common import git_has_real_changes, git_value, read_yaml, write_json
from package_adapter import adapter_artifact_exists

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"
EXPECTED_DRIVE_ACCOUNT_EMAIL = "johnwickiller4444@gmail.com"
MIN_EXPECTED_COMMIT_LENGTH = 12
ADAPTER_DIR_NAMES = ("qwen25_gp4_lora", "qwen25_gp4_lora_pilot")
LEGACY_DRIVE_ROOT_RUN_ID = "legacy-drive-root"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a Colab/Drive preflight readiness report."
    )
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gp4-ws", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path)
    parser.add_argument("--previous-adapter", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-adapter-only-reuse", action="store_true")
    parser.add_argument("--allow-mixed-prior-artifacts", action="store_true")
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
        allow_adapter_only_reuse=args.allow_adapter_only_reuse,
        allow_mixed_prior_artifacts=args.allow_mixed_prior_artifacts,
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
    old_dataset: Path | None = None,
    previous_adapter: Path | None = None,
    expected_commit: str | None = None,
    allow_adapter_only_reuse: bool = False,
    allow_mixed_prior_artifacts: bool = False,
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
    expected_drive_account_email = _expected_drive_account_email()
    drive_email = _read_text(drive_hint_path).strip()
    drive_confirmation = _read_optional_json(drive_confirmation_path)
    confirmed_email = str(drive_confirmation.get("confirmed_email") or "").strip()
    storage_owner_email = str(
        drive_confirmation.get("storage_owner_email")
        or drive_confirmation.get("expected_email")
        or drive_email
    ).strip()
    runtime_account_confirmed = str(
        drive_confirmation.get("runtime_google_account_confirmed")
        or drive_confirmation.get("runtime_account_confirmed")
        or confirmed_email
    ).strip()
    drive_account_confirmed = (
        drive_confirmation.get("confirmed") is True
        and confirmed_email == expected_drive_account_email
        and str(drive_confirmation.get("expected_email") or "").strip()
        == expected_drive_account_email
        and storage_owner_email == expected_drive_account_email
    )
    cross_account_runner = bool(
        drive_confirmation.get("cross_account_runner")
    ) or (
        bool(runtime_account_confirmed)
        and runtime_account_confirmed != expected_drive_account_email
    )
    old_dataset_exists = old_dataset.is_file() if old_dataset is not None else False
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
        allow_adapter_only_reuse=allow_adapter_only_reuse,
        allow_mixed_prior_artifacts=allow_mixed_prior_artifacts,
    )
    base_model_start = bool(previous_run["base_model_start"])
    adapter_only_reuse = allow_adapter_only_reuse and old_dataset is None and not base_model_start
    fresh_generation_start = old_dataset is None and base_model_start
    gp4_ws_exists = gp4_ws.exists()
    gp4_ws_allowed = is_allowed_cloud_path(gp4_ws, gp4_ws_policy)
    gp4_ws_branch = git_value(gp4_ws, "branch", "--show-current") if gp4_ws_exists else ""
    gp4_ws_head = git_value(gp4_ws, "rev-parse", "HEAD") if gp4_ws_exists else ""
    gp4_ws_dirty = git_has_real_changes(gp4_ws) if gp4_ws_exists else False

    checks = [
        _check(
            "cloud_root_allowed",
            is_allowed_cloud_path(cloud_root, policy),
            str(cloud_root),
        ),
        _check(
            "drive_account_hint_matches",
            drive_email == expected_drive_account_email,
            str(drive_hint_path),
        ),
        _check(
            "drive_account_confirmation_matches",
            drive_account_confirmed,
            str(drive_confirmation_path),
        ),
        _check(
            "old_dataset_exists",
            old_dataset_exists or adapter_only_reuse or fresh_generation_start,
            str(old_dataset or "<fresh-generation-from-base>"),
        ),
        _check(
            "old_dataset_allowed_cloud_path",
            (old_dataset is not None and is_allowed_cloud_path(old_dataset, policy))
            or adapter_only_reuse
            or fresh_generation_start,
            str(old_dataset or "<fresh-generation-from-base>"),
        ),
        _check(
            "old_dataset_has_rows",
            old_dataset_rows > 0 or adapter_only_reuse or fresh_generation_start,
            str(old_dataset or "<fresh-generation-from-base>"),
        ),
        _check(
            "previous_adapter_required",
            previous_adapter is not None or base_model_start,
            str(previous_adapter or "<base-model-start>"),
        ),
        _check(
            "previous_adapter_allowed_cloud_path",
            previous_adapter_allowed or base_model_start,
            str(previous_adapter or "<base-model-start>"),
        ),
        _check(
            "previous_adapter_exists",
            previous_adapter_exists or base_model_start,
            str(previous_adapter or "<base-model-start>"),
        ),
        _check(
            "previous_adapter_artifact_exists",
            previous_adapter_artifact_exists or base_model_start,
            str(previous_adapter or "<base-model-start>"),
        ),
        _check(
            "previous_reuse_same_prior_run",
            previous_run["matched"] or base_model_start,
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
            "storage_owner_email": storage_owner_email,
            "confirmed_email": confirmed_email,
            "runtime_account_confirmed": runtime_account_confirmed,
            "cross_account_runner": cross_account_runner,
            "confirmed": drive_account_confirmed,
            "confirmation_method": str(drive_confirmation.get("method") or ""),
            "expected": expected_drive_account_email,
            "matches_expected": (
                drive_email == expected_drive_account_email
                and drive_account_confirmed
            ),
        },
        "old_dataset": {
            "path": str(old_dataset) if old_dataset is not None else "",
            "exists": old_dataset_exists,
            "allowed_cloud_path": (
                is_allowed_cloud_path(old_dataset, policy)
                if old_dataset is not None
                else False
            ),
            "rows": old_dataset_rows,
            "sha256": old_dataset_sha,
            "adapter_only_reuse": adapter_only_reuse,
            "fresh_generation_from_base": fresh_generation_start,
        },
        "previous_adapter": {
            "path": str(previous_adapter) if previous_adapter is not None else "",
            "exists": previous_adapter_exists,
            "allowed_cloud_path": previous_adapter_allowed,
            "artifact_exists": previous_adapter_artifact_exists,
            "base_model_start": base_model_start,
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
    old_dataset: Path | None,
    previous_adapter: Path | None,
    allow_adapter_only_reuse: bool,
    allow_mixed_prior_artifacts: bool,
) -> dict[str, Any]:
    drive_root = cloud_root.parent
    old_dataset_run_id = _previous_run_id_from_path(
        path=old_dataset,
        drive_root=drive_root,
        suffix=Path("data/validated/accepted_300k.jsonl"),
    )
    previous_adapter_run_id = (
        _previous_adapter_run_id_from_path(
            path=previous_adapter,
            drive_root=drive_root,
        )
        if previous_adapter is not None
        else ""
    )
    adapter_only_reuse = (
        allow_adapter_only_reuse
        and not old_dataset_run_id
        and bool(previous_adapter_run_id)
    )
    base_model_start = (
        previous_adapter is None
        and not _is_current_run_path(old_dataset, cloud_root)
    )
    same_source_run = (
        bool(previous_adapter_run_id)
        and (
            (bool(old_dataset_run_id) and old_dataset_run_id == previous_adapter_run_id)
            or adapter_only_reuse
        )
    )
    mixed_prior_artifacts = (
        bool(old_dataset_run_id)
        and bool(previous_adapter_run_id)
        and old_dataset_run_id != previous_adapter_run_id
    )
    mixed_prior_artifacts_accepted = (
        allow_mixed_prior_artifacts
        and mixed_prior_artifacts
        and old_dataset_run_id != run_id
        and previous_adapter_run_id != run_id
    )
    source_run_id = previous_adapter_run_id if adapter_only_reuse else old_dataset_run_id
    is_prior_run = same_source_run and source_run_id != run_id
    matched = (
        (same_source_run and is_prior_run)
        or mixed_prior_artifacts_accepted
        or base_model_start
    )
    details = (
        f"old_dataset_run_id={old_dataset_run_id or '<unknown>'} "
        f"previous_adapter_run_id={previous_adapter_run_id or '<none>'} "
        f"current_run_id={run_id}"
    )
    return {
        "drive_root": str(drive_root),
        "old_dataset_run_id": old_dataset_run_id,
        "previous_adapter_run_id": previous_adapter_run_id,
        "adapter_only_reuse": adapter_only_reuse,
        "base_model_start": base_model_start,
        "same_source_run": same_source_run,
        "mixed_prior_artifacts": mixed_prior_artifacts,
        "mixed_prior_artifacts_allowed": allow_mixed_prior_artifacts,
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

def _previous_adapter_run_id_from_path(
    *,
    path: Path | None,
    drive_root: Path,
) -> str:
    if path is None:
        return ""
    try:
        relative = path.resolve(strict=False).relative_to(
            drive_root.resolve(strict=False)
        )
    except ValueError:
        return ""
    parts = relative.parts
    if len(parts) < 2:
        return ""
    if parts[0] == "models" and parts[1] in ADAPTER_DIR_NAMES:
        return LEGACY_DRIVE_ROOT_RUN_ID
    for index in range(1, len(parts) - 1):
        if parts[index] == "models" and parts[index + 1] in ADAPTER_DIR_NAMES:
            return parts[0]
    return ""


def _is_current_run_path(path: Path | None, cloud_root: Path) -> bool:
    if path is None:
        return False
    try:
        path.resolve(strict=False).relative_to(cloud_root.resolve(strict=False))
    except ValueError:
        return False
    return True

def _configured_expected_commit(raw_value: str | None, *, env_name: str) -> str:
    return (raw_value or os.environ.get(env_name) or "").strip()

def _expected_drive_account_email() -> str:
    return os.environ.get(
        "GP4_STORAGE_OWNER_EMAIL",
        EXPECTED_DRIVE_ACCOUNT_EMAIL,
    ).strip() or EXPECTED_DRIVE_ACCOUNT_EMAIL


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
