#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from factory_common import read_json, write_json


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

    payload = {
        "passed": blocked_reason == "",
        "blocked_reason": blocked_reason,
        "adapter_dir": str(adapter_dir),
        "acceptance_report": str(acceptance_report),
    }
    write_json(report_path, payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
