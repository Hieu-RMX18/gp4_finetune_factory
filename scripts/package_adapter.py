#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import read_json, write_json


def adapter_artifact_exists(adapter_dir: Path) -> bool:
    if not adapter_dir.is_dir():
        return False
    has_config = adapter_dir.joinpath("adapter_config.json").is_file()
    has_weights = any(
        path.is_file() and path.stat().st_size > 0
        for pattern in ("*.safetensors", "*.bin")
        for path in adapter_dir.glob(pattern)
    )
    return has_config and has_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Package accepted GP4 LoRA adapter metadata.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--cloud-root", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    policy = CloudStoragePolicy(
        cloud_roots=tuple(Path(root) for root in args.cloud_root),
        allow_tmp=args.allow_tmp,
    )
    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root[0] if args.cloud_root else None,
            dry_run=False,
            inputs=[args.adapter_dir, args.acceptance_report],
            outputs=[args.report],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        can_write_report = False
        try:
            validate_cloud_run_paths(
                cloud_root=args.cloud_root[0] if args.cloud_root else None,
                dry_run=False,
                inputs=[],
                outputs=[args.report],
                allow_tmp=args.allow_tmp,
            )
            can_write_report = True
        except CloudPathError:
            can_write_report = False
        if can_write_report:
            write_json(args.report, {"passed": False, "blocked_reason": str(exc)})
        print(f"passed=False blocked_reason={exc} report={args.report}")
        return 1
    payload = package_adapter_metadata(
        adapter_dir=args.adapter_dir,
        acceptance_report=args.acceptance_report,
        report_path=args.report,
        policy=policy,
    )
    print(f"passed={payload['passed']} blocked_reason={payload['blocked_reason']} report={args.report}")
    return 0 if payload["passed"] else 1


def package_adapter_metadata(
    *,
    adapter_dir: Path,
    acceptance_report: Path,
    report_path: Path,
    policy: CloudStoragePolicy,
) -> dict:
    acceptance = read_json(acceptance_report)
    blocked_reason = ""
    if not acceptance.get("passed"):
        blocked_reason = "acceptance gate has not passed"
    elif not is_allowed_cloud_path(adapter_dir, policy):
        blocked_reason = "adapter path is not allowed by cloud storage policy"
    elif not adapter_dir.exists():
        blocked_reason = "adapter path does not exist"
    elif not adapter_artifact_exists(adapter_dir):
        blocked_reason = "adapter artifact files are missing"

    payload = {
        "passed": blocked_reason == "",
        "blocked_reason": blocked_reason,
        "adapter_dir": str(adapter_dir),
        "adapter_artifact_exists": adapter_artifact_exists(adapter_dir),
        "acceptance_report": str(acceptance_report),
    }
    write_json(report_path, payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
