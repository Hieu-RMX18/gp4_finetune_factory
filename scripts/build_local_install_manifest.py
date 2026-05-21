#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import git_value, read_json, read_yaml, write_json
from package_adapter import adapter_artifact_exists

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"
DEPENDENCY_MANIFESTS = (
    ROOT / "requirements-cloud.txt",
    ROOT / "requirements-local-adapter.txt",
)
MIN_EXPECTED_COMMIT_LENGTH = 12


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build local install readiness manifest without installing."
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--target-repo", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=False,
            inputs=[args.adapter_dir, args.acceptance_report],
            outputs=[args.output],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"local_install_manifest_blocked reason={exc} output={args.output}")
        return 1

    policy = CloudStoragePolicy((args.cloud_root,), allow_tmp=args.allow_tmp)
    manifest = build_manifest(
        adapter_dir=args.adapter_dir,
        acceptance_report=args.acceptance_report,
        target_repo=args.target_repo,
        policy=policy,
        expected_commit=args.expected_commit,
    )
    write_json(args.output, manifest)
    print(
        "ready_for_local_install="
        f"{manifest['ready_for_local_install']} output={args.output}"
    )
    return 0 if manifest["ready_for_local_install"] else 1


def build_manifest(
    *,
    adapter_dir: Path,
    acceptance_report: Path,
    target_repo: Path,
    policy: CloudStoragePolicy,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    acceptance = read_json(acceptance_report)
    target_repo_state = _target_repo_state(
        target_repo,
        expected_commit=expected_commit,
    )
    target_repo_state["allowed_cloud_path"] = is_allowed_cloud_path(
        target_repo,
        policy,
    )
    blocked_reason = ""
    if acceptance.get("passed") is not True:
        blocked_reason = "acceptance gate has not passed"
    elif target_repo_state["exists"] is not True:
        blocked_reason = "target repo snapshot is missing"
    elif target_repo_state["allowed_cloud_path"] is not True:
        blocked_reason = "target repo snapshot is not allowed by cloud storage policy"
    elif target_repo_state["current_branch"] != target_repo_state["expected_branch"]:
        blocked_reason = "target repo snapshot is not on the expected branch"
    elif not target_repo_state["head"]:
        blocked_reason = "target repo snapshot has no commit"
    elif not target_repo_state["expected_commit"]:
        blocked_reason = "expected gp4_ws commit is not configured"
    elif target_repo_state["expected_commit_matches"] is not True:
        blocked_reason = "target repo snapshot is not at the expected commit"
    elif target_repo_state["is_dirty"] is True:
        blocked_reason = "target repo snapshot is dirty"
    elif not is_allowed_cloud_path(adapter_dir, policy):
        blocked_reason = "adapter path is not allowed by cloud storage policy"
    elif not adapter_artifact_exists(adapter_dir):
        blocked_reason = "adapter artifact files are missing"

    ready = blocked_reason == ""
    return {
        "passed": ready,
        "ready_for_local_install": ready,
        "blocked_reason": blocked_reason,
        "target_repo": str(target_repo),
        "target_repo_state": target_repo_state,
        "acceptance_report": str(acceptance_report),
        "dependency_manifests": _dependency_manifests(),
        "adapter": {
            "path": str(adapter_dir),
            "files": _file_manifest(adapter_dir) if adapter_dir.exists() else {},
        },
        "install_action_performed": False,
        "safety_boundary": (
            "adapter may draft Semantic IR only; robot execution remains behind "
            "validation, safety, planning, approval, and hardware gates"
        ),
    }


def _target_repo_state(
    target_repo: Path,
    *,
    expected_commit: str | None,
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
    exists = target_repo.exists()
    head = git_value(target_repo, "rev-parse", "--short", "HEAD") if exists else ""
    head_full = git_value(target_repo, "rev-parse", "HEAD") if exists else ""
    return {
        "path": str(target_repo),
        "exists": exists,
        "expected_branch": expected_branch,
        "expected_commit": expected_commit,
        "current_branch": git_value(target_repo, "branch", "--show-current") if exists else "",
        "head": head,
        "head_full": head_full,
        "expected_commit_matches": _commit_matches(head_full or head, expected_commit),
        "is_dirty": bool(git_value(target_repo, "status", "--short")) if exists else False,
    }


def _configured_expected_commit(raw_value: str | None, *, env_name: str) -> str:
    configured = (raw_value or os.environ.get(env_name) or "").strip()
    return configured


def _commit_matches(head: str, expected_commit: str) -> bool:
    head = head.strip().lower()
    expected_commit = expected_commit.strip().lower()
    if not head or len(expected_commit) < MIN_EXPECTED_COMMIT_LENGTH:
        return False
    return head.startswith(expected_commit)


def _dependency_manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for path in DEPENDENCY_MANIFESTS:
        if not path.exists():
            continue
        manifests[path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return manifests


def _file_manifest(adapter_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(adapter_dir.iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    return files


if __name__ == "__main__":
    raise SystemExit(main())
