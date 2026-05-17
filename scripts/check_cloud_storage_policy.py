#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from cloud_runtime import CloudPathError, validate_cloud_output_path


FORBIDDEN_LOCAL_PARTS = {
    "data/generated",
    "data/validated",
    "data/splits",
    "reports",
    "models",
    "outputs",
    "artifact_downloads",
}


@dataclass(frozen=True)
class CloudStoragePolicy:
    cloud_roots: Sequence[Path]
    allow_tmp: bool = False


def normalize_path(path: Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve(strict=False)


def is_allowed_cloud_path(path: Path, policy: CloudStoragePolicy) -> bool:
    resolved = normalize_path(path)
    if policy.allow_tmp and str(resolved).startswith("/tmp/"):
        return True
    for root in policy.cloud_roots:
        resolved_root = normalize_path(root)
        if resolved == resolved_root or resolved_root in resolved.parents:
            return True
    return False


def find_local_artifact_paths(paths: Iterable[str], policy: CloudStoragePolicy) -> list[str]:
    findings: list[str] = []
    for raw_path in paths:
        candidate = normalize_path(Path(raw_path))
        if is_allowed_cloud_path(candidate, policy):
            continue
        candidate_text = candidate.as_posix()
        if any(part in candidate_text for part in FORBIDDEN_LOCAL_PARTS):
            findings.append(raw_path)
        elif candidate_text.startswith("/home/") or "/Downloads/" in candidate_text:
            findings.append(raw_path)
    return findings


def write_policy_report(report_path: Path, findings: list[str]) -> dict:
    payload = {
        "passed": len(findings) == 0,
        "local_artifact_usage": len(findings),
        "findings": findings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check cloud-only artifact path policy.")
    parser.add_argument("--cloud-root", action="append", required=True)
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    policy = CloudStoragePolicy(
        cloud_roots=tuple(Path(root) for root in args.cloud_root),
        allow_tmp=args.allow_tmp,
    )
    try:
        validate_cloud_output_path(
            args.report,
            args.cloud_root[0] if args.cloud_root else None,
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(str(exc))
        return 1

    findings = find_local_artifact_paths(args.path, policy)
    report = write_policy_report(args.report, findings)
    print(
        f"passed={report['passed']} "
        f"local_artifact_usage={report['local_artifact_usage']} report={args.report}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
