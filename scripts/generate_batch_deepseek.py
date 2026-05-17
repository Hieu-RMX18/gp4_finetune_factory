#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import read_jsonl, read_yaml, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC_PATH = ROOT / "configs/dataset_spec.yaml"
GENERATION_POLICY_PATH = ROOT / "configs/generation_policy.yaml"
ALLOWED_DEEPSEEK_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}


class DeepSeekProviderError(ValueError):
    pass


@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str
    model: str
    api_key_env: str
    api_key: str
    temperature: float
    max_tokens: int


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
    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root[0] if args.cloud_root else None,
            dry_run=args.dry_run,
            inputs=[args.seed],
            outputs=[args.output, args.report],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        can_write_report = False
        try:
            validate_cloud_run_paths(
                cloud_root=args.cloud_root[0] if args.cloud_root else None,
                dry_run=args.dry_run,
                inputs=[],
                outputs=[args.report],
                allow_tmp=args.allow_tmp,
            )
            can_write_report = True
        except CloudPathError:
            can_write_report = False
        if can_write_report:
            _report(
                args.report,
                passed=False,
                blocked_reason=str(exc),
                seed_rows=0,
                generated=0,
            )
        print(f"blocked_reason={exc} report={args.report}")
        return 1
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
    try:
        deepseek = resolve_deepseek_config(
            generation_policy,
            env=os.environ,
            dry_run=args.dry_run,
        )
    except DeepSeekProviderError as exc:
        report = _report(
            args.report,
            passed=False,
            blocked_reason=str(exc),
            seed_rows=len(seed_rows),
            generated=0,
        )
        print(f"blocked_reason={report['blocked_reason']} report={args.report}")
        return 1

    if args.dry_run:
        report = _report(
            args.report,
            passed=True,
            blocked_reason="",
            seed_rows=len(seed_rows),
            generated=0,
            model=deepseek.model,
            dry_run=True,
        )
        print(f"dry_run_ok seed_rows={len(seed_rows)} model={deepseek.model} report={args.report}")
        return 0

    rows = _generate_rows(
        api_key=deepseek.api_key,
        base_url=deepseek.base_url,
        model=deepseek.model,
        seed_rows=seed_rows,
        count=args.count,
        temperature=deepseek.temperature,
        max_tokens=deepseek.max_tokens,
    )
    write_jsonl(args.output, rows)
    _report(
        args.report,
        passed=True,
        blocked_reason="",
        seed_rows=len(seed_rows),
        generated=len(rows),
        model=deepseek.model,
    )
    print(f"generated={len(rows)} output={args.output} report={args.report}")
    return 0


def resolve_deepseek_config(
    generation_policy: dict[str, Any],
    *,
    env: Mapping[str, str],
    dry_run: bool,
) -> DeepSeekConfig:
    deepseek = generation_policy["deepseek"]
    api_key_env = str(deepseek.get("api_key_env", "DEEPSEEK_API_KEY"))
    api_key = env.get(api_key_env, "")
    base_url = env.get("DEEPSEEK_BASE_URL") or str(deepseek["base_url"])
    model = str(deepseek.get("model", "deepseek-v4-flash"))

    if model not in ALLOWED_DEEPSEEK_MODELS:
        raise DeepSeekProviderError(f"unsupported DeepSeek model: {model}")
    _validate_base_url(base_url, dry_run=dry_run)
    if not dry_run and not api_key:
        raise DeepSeekProviderError(f"{api_key_env} is not configured")

    return DeepSeekConfig(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key_env=api_key_env,
        api_key=api_key,
        temperature=float(deepseek["temperature"]),
        max_tokens=int(deepseek["max_tokens"]),
    )


def _validate_base_url(base_url: str, *, dry_run: bool) -> None:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if not dry_run and hostname in {"localhost", "127.0.0.1", "::1"}:
        raise DeepSeekProviderError("localhost provider URLs are forbidden in cloud non-dry-run")
    if not dry_run and hostname != "api.deepseek.com":
        raise DeepSeekProviderError(f"DeepSeek base URL must be https://api.deepseek.com: {base_url}")
    if not dry_run and parsed.scheme != "https":
        raise DeepSeekProviderError("DeepSeek base URL must use https")


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
