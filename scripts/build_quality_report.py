#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from factory_common import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a GP4 cloud workflow quality report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-report", action="append", default=[])
    args = parser.parse_args()

    payload = build_quality_report([Path(path) for path in args.input_report])
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
