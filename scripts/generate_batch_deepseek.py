#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from factory_common import read_jsonl, read_yaml, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC_PATH = ROOT / "configs/dataset_spec.yaml"
GENERATION_POLICY_PATH = ROOT / "configs/generation_policy.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GP4 ReAct-IR rows with DeepSeek.")
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-root", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    policy = CloudStoragePolicy(
        cloud_roots=tuple(Path(root) for root in args.cloud_root),
        allow_tmp=args.allow_tmp,
    )
    for path in (args.output, args.report):
        if not is_allowed_cloud_path(path, policy):
            report = _report(
                args.report,
                passed=False,
                blocked_reason="cloud storage policy blocked path",
                seed_rows=0,
                generated=0,
            )
            print(f"blocked_reason={report['blocked_reason']} report={args.report}")
            return 1

    seed_rows = read_jsonl(args.seed)
    dataset_spec = read_yaml(DATASET_SPEC_PATH)
    minimum_seed = int(dataset_spec["seed_gate"]["minimum_handwritten_seed_examples"])
    if len(seed_rows) < minimum_seed:
        report = _report(
            args.report,
            passed=False,
            blocked_reason="seed gate blocked generation",
            seed_rows=len(seed_rows),
            generated=0,
        )
        print(
            f"blocked_reason={report['blocked_reason']} "
            f"seed_rows={len(seed_rows)} minimum={minimum_seed} report={args.report}"
        )
        return 1

    generation_policy = read_yaml(GENERATION_POLICY_PATH)
    deepseek = generation_policy["deepseek"]
    if args.dry_run:
        report = _report(
            args.report,
            passed=True,
            blocked_reason="",
            seed_rows=len(seed_rows),
            generated=0,
            model=str(deepseek["model"]),
            dry_run=True,
        )
        print(f"dry_run_ok seed_rows={len(seed_rows)} model={deepseek['model']} report={args.report}")
        return 0

    api_key_env = str(deepseek["api_key_env"])
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        report = _report(
            args.report,
            passed=False,
            blocked_reason=f"{api_key_env} is not configured",
            seed_rows=len(seed_rows),
            generated=0,
        )
        print(f"blocked_reason={report['blocked_reason']} report={args.report}")
        return 1

    rows = _generate_rows(
        api_key=api_key,
        base_url=str(deepseek["base_url"]),
        model=str(deepseek["model"]),
        seed_rows=seed_rows,
        count=args.count,
        temperature=float(deepseek["temperature"]),
        max_tokens=int(deepseek["max_tokens"]),
    )
    write_jsonl(args.output, rows)
    _report(
        args.report,
        passed=True,
        blocked_reason="",
        seed_rows=len(seed_rows),
        generated=len(rows),
        model=str(deepseek["model"]),
    )
    print(f"generated={len(rows)} output={args.output} report={args.report}")
    return 0


def _report(path: Path, **values: Any) -> dict[str, Any]:
    payload = {
        "passed": bool(values.get("passed", False)),
        "blocked_reason": str(values.get("blocked_reason", "")),
        "seed_rows": int(values.get("seed_rows", 0)),
        "generated": int(values.get("generated", 0)),
        "dry_run": bool(values.get("dry_run", False)),
        "model": values.get("model"),
    }
    write_json(path, payload)
    return payload


def _generate_rows(
    *,
    api_key: str,
    base_url: str,
    model: str,
    seed_rows: list[dict[str, Any]],
    count: int,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Generate JSON-only GP4 ReAct-IR dataset rows.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"requested_rows": count, "seed_examples": seed_rows[:8]},
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    content = decoded["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    rows = parsed["examples"] if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        raise RuntimeError("DeepSeek response did not contain a row list.")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
