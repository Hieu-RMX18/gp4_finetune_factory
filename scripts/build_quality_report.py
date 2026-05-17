#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a GP4 cloud workflow quality report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-report", action="append", default=[])
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    input_reports = [Path(path) for path in args.input_report]
    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=False,
            inputs=input_reports,
            outputs=[args.report],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"quality_report_blocked reason={exc} report={args.report}")
        return 1

    payload = build_quality_report(input_reports)
    write_json(args.report, payload)
    print(f"passed={payload['passed']} report={args.report}")
    return 0 if payload["passed"] else 1


def build_quality_report(paths: list[Path]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    reject_reasons: Counter[str] = Counter()
    passed = True
    for path in paths:
        report = read_json(path)
        reports[str(path)] = report
        if report.get("passed") is False:
            passed = False
        for issue in report.get("issues", []):
            if isinstance(issue, dict):
                reject_reasons[str(issue.get("message", "unknown"))] += 1
    return {
        "passed": passed,
        "reports": reports,
        "reject_reasons": dict(reject_reasons),
        "distribution": {},
        "next_batch_feedback": sorted(reject_reasons),
    }


if __name__ == "__main__":
    raise SystemExit(main())
