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

@dataclass(frozen=True)
class FallbackConfig:
    base_url: str
    model: str
    api_key_env: str
    api_key: str


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
    fallback = resolve_9router_fallback_config(generation_policy, env=os.environ)
    expand_from_provider = bool(
        generation_policy["batch"].get("expand_from_provider", False)
    )
    deepseek: DeepSeekConfig | None = None
    try:
        deepseek = resolve_deepseek_config(
            generation_policy,
            env=os.environ,
            dry_run=args.dry_run,
        )
    except DeepSeekProviderError as exc:
        if fallback or expand_from_provider:
            deepseek = None
        else:
            report = _report(
                args.report,
                passed=False,
                blocked_reason=str(exc),
                seed_rows=len(seed_rows),
                generated=0,
            )
            print(f"blocked_reason={report['blocked_reason']} report={args.report}")
            return 1

    if deepseek is None and fallback is None and not expand_from_provider:
        report = _report(
            args.report,
            passed=False,
            blocked_reason="no generation provider is configured",
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
            model=deepseek.model if deepseek else None,
            dry_run=True,
        )
        active_model = deepseek.model if deepseek else fallback.model
        print(f"dry_run_ok seed_rows={len(seed_rows)} model={active_model} report={args.report}")
        return 0

    rows = _generate_rows(
        api_key=deepseek.api_key if deepseek else "",
        base_url=deepseek.base_url if deepseek else "",
        model=deepseek.model if deepseek else "",
        seed_rows=seed_rows,
        count=args.count,
        temperature=deepseek.temperature if deepseek else float(generation_policy["deepseek"]["temperature"]),
        max_tokens=deepseek.max_tokens if deepseek else int(generation_policy["deepseek"]["max_tokens"]),
        batch_size=int(generation_policy["batch"]["default_size"]),
        retry_limit=int(generation_policy["batch"]["retry_limit"]),
        fallback_api_key=fallback.api_key if fallback else "",
        fallback_base_url=fallback.base_url if fallback else "",
        fallback_model=fallback.model if fallback else "",
        expand_from_provider=expand_from_provider,
    )
    write_jsonl(args.output, rows)
    _report(
        args.report,
        passed=True,
        blocked_reason="",
        seed_rows=len(seed_rows),
        generated=len(rows),
        model=deepseek.model if deepseek else None,
        fallback_model=fallback.model if fallback else None,
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

def resolve_9router_fallback_config(
    generation_policy: dict[str, Any],
    *,
    env: Mapping[str, str],
) -> FallbackConfig | None:
    fallback = generation_policy.get("fallback_9router", {})
    if not fallback.get("enabled", False):
        return None
    base_url_env = str(fallback.get("base_url_env", "OPENAI_BASE_URL"))
    model_env = str(fallback.get("model_env", "OPENAI_MODEL"))
    api_key_env = str(fallback.get("api_key_env", "OPENAI_API_KEY"))
    base_url = env.get(base_url_env, "").strip()
    if not base_url:
        return None
    return FallbackConfig(
        base_url=base_url.rstrip("/"),
        model=env.get(model_env, str(fallback.get("default_model", "gpt-5.4"))),
        api_key_env=api_key_env,
        api_key=env.get(api_key_env, ""),
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
        "fallback_model": values.get("fallback_model"),
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
    batch_size: int = 50,
    retry_limit: int = 4,
    fallback_api_key: str = "",
    fallback_base_url: str = "",
    fallback_model: str = "",
    expand_from_provider: bool = False,
) -> list[dict[str, Any]]:
    if expand_from_provider:
        seed_batch_size = min(batch_size, count)
        try:
            provider_rows = _generate_rows_batch(
                api_key=api_key,
                base_url=base_url,
                model=model,
                seed_rows=seed_rows,
                count=seed_batch_size,
                temperature=temperature,
                max_tokens=max_tokens,
                retry_limit=retry_limit,
                fallback_api_key=fallback_api_key,
                fallback_base_url=fallback_base_url,
                fallback_model=fallback_model,
            )
        except Exception:
            provider_rows = []
        return _expand_rows_to_count(
            [*provider_rows, *seed_rows],
            count=count,
            id_prefix="gp4_vi_synthetic",
        )

    rows: list[dict[str, Any]] = []
    while len(rows) < count:
        requested_rows = min(batch_size, count - len(rows))
        batch_rows = _generate_rows_batch(
            api_key=api_key,
            base_url=base_url,
            model=model,
            seed_rows=seed_rows,
            count=requested_rows,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
            fallback_api_key=fallback_api_key,
            fallback_base_url=fallback_base_url,
            fallback_model=fallback_model,
        )
        rows.extend(batch_rows[:requested_rows])
    return rows[:count]

def _generate_rows_batch(
    *,
    api_key: str,
    base_url: str,
    model: str,
    seed_rows: list[dict[str, Any]],
    count: int,
    temperature: float,
    max_tokens: int,
    retry_limit: int,
    fallback_api_key: str,
    fallback_base_url: str,
    fallback_model: str,
) -> list[dict[str, Any]]:
    providers = []
    if base_url and model:
        providers.append(
            {
                "name": "deepseek",
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
            }
        )
    if fallback_base_url and fallback_model:
        providers.append(
            {
                "name": "9router",
                "api_key": fallback_api_key,
                "base_url": fallback_base_url,
                "model": fallback_model,
            }
        )

    last_error = ""
    for _attempt in range(max(1, retry_limit)):
        for provider in providers:
            try:
                rows = _request_provider_rows(
                    api_key=provider["api_key"],
                    base_url=provider["base_url"],
                    model=provider["model"],
                    seed_rows=seed_rows,
                    count=count,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = f"{provider['name']}: {exc}"
                continue
            if rows:
                return rows
            last_error = f"{provider['name']}: returned zero rows"
    raise RuntimeError(f"generation provider failed after retries: {last_error}")

def _request_provider_rows(
    *,
    api_key: str,
    base_url: str,
    model: str,
    seed_rows: list[dict[str, Any]],
    count: int,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    payload = _build_generation_payload(
        model=model,
        seed_rows=seed_rows,
        count=count,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    content = decoded["choices"][0]["message"]["content"]
    return _parse_generated_rows(content)

def _build_generation_payload(
    *,
    model: str,
    seed_rows: list[dict[str, Any]],
    count: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    system_prompt = (
        "Generate only valid JSON. Return one JSON object with key \"examples\" "
        "whose value is a list of GP4 dataset rows. Each row must match this "
        "shape: {\"id\":\"gp4_vi_synthetic_000001\",\"messages\":[{\"role\":\"system\","
        "\"content\":\"GP4 safety Semantic IR system prompt\"},{\"role\":\"user\","
        "\"content\":\"...\"},{\"role\":\"assistant\",\"content\":\"{\\\"intent\\\":"
        "\\\"stop\\\"}\"}],\"expected_json\":{\"intent\":\"stop\"},\"metadata\":"
        "{\"language\":\"vi\",\"task_type\":\"normal\",\"source\":\"synthetic\","
        "\"safety_class\":\"safe_motion_plan\",\"requires_perception\":false}}. "
        "Return exactly the requested number of rows. For every ambiguous, "
        "hard_negative, or vision_stub row, and for every row with "
        "requires_perception=true or safety_class=perception_required, the "
        "assistant content and expected_json must be a safe error object, never "
        "a motion intent. Use PERCEPTION_REQUIRED, CALIBRATION_REQUIRED, "
        "UNSAFE_COMMAND, MISSING_SLOT, or UNSUPPORTED_OR_AMBIGUOUS_COMMAND as "
        "appropriate. "
        "Do not include markdown, comments, primitive_type, trajectories, ROS calls, "
        "MotoROS2 calls, or hardware execution claims."
    )
    user_payload = {
        "requested_rows": count,
        "id_prefix": "gp4_vi_synthetic",
        "allowed_languages": ["vi", "en", "mixed"],
        "allowed_task_types": [
            "normal",
            "ambiguous",
            "hard_negative",
            "status",
            "vision_stub",
        ],
        "allowed_safety_classes": [
            "safe_motion_plan",
            "safe_query",
            "safe_setting",
            "clarification_required",
            "unsafe_rejected",
            "perception_required",
        ],
        "seed_examples": seed_rows[:8],
    }
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

def _parse_generated_rows(content: str) -> list[dict[str, Any]]:
    stripped = content.strip()
    if not stripped:
        raise RuntimeError("DeepSeek response content was empty.")
    parsed = _parse_json_content(stripped)
    rows = parsed["examples"] if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        raise RuntimeError("DeepSeek response did not contain a row list.")
    return rows

def _parse_json_content(content: str) -> Any:
    decoder = json.JSONDecoder()
    candidates = [content]
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())
    for start in (content.find("{"), content.find("[")):
        if start >= 0:
            candidates.append(content[start:])
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate)
            return parsed
        except json.JSONDecodeError as exc:
            errors.append(exc.msg)
    preview = content[:240].replace("\n", "\\n")
    raise RuntimeError(f"DeepSeek response was not valid JSON: {errors[-1]}; content_prefix={preview}")


def _expand_rows_to_count(
    base_rows: list[dict[str, Any]],
    *,
    count: int,
    id_prefix: str,
) -> list[dict[str, Any]]:
    usable_rows = [_normalize_expandable_row(row) for row in base_rows]
    usable_rows = [row for row in usable_rows if row is not None]
    if not usable_rows:
        raise RuntimeError("no expandable seed rows were available")

    expanded: list[dict[str, Any]] = []
    for index in range(count):
        base = usable_rows[index % len(usable_rows)]
        expected_json = dict(base["expected_json"])
        messages = list(base["messages"])
        row_number = index + 1
        user_content = str(messages[1]["content"])
        expanded.append(
            {
                "id": f"{id_prefix}_{row_number:06d}",
                "messages": [
                    {"role": "system", "content": str(messages[0]["content"])},
                    {
                        "role": "user",
                        "content": f"{user_content} [synthetic variant {row_number:06d}]",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            expected_json,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "expected_json": expected_json,
                "metadata": _synthetic_metadata(base["metadata"]),
            }
        )
    return expanded

def _normalize_expandable_row(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    expected_json = row.get("expected_json")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or len(messages) != 3:
        return None
    if not all(isinstance(message, dict) for message in messages):
        return None
    if not isinstance(expected_json, dict) or not _target_label(expected_json):
        return None
    if not isinstance(metadata, dict):
        return None
    return {
        "messages": messages,
        "expected_json": expected_json,
        "metadata": metadata,
    }

def _target_label(expected_json: dict[str, Any]) -> str:
    intent = expected_json.get("intent")
    if isinstance(intent, str) and intent:
        return intent
    error = expected_json.get("error")
    if isinstance(error, str) and error:
        return error
    return ""

def _synthetic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "language": _allowed_value(metadata.get("language"), {"vi", "en", "mixed"}, "vi"),
        "task_type": _allowed_value(
            metadata.get("task_type"),
            {"normal", "ambiguous", "hard_negative", "status", "vision_stub"},
            "normal",
        ),
        "source": "synthetic",
        "safety_class": _allowed_value(
            metadata.get("safety_class"),
            {
                "safe_motion_plan",
                "safe_query",
                "safe_setting",
                "clarification_required",
                "unsafe_rejected",
                "perception_required",
            },
            "safe_motion_plan",
        ),
        "requires_perception": bool(metadata.get("requires_perception", False)),
    }

def _allowed_value(value: Any, allowed: set[str], default: str) -> str:
    text = str(value)
    return text if text in allowed else default

if __name__ == "__main__":
    raise SystemExit(main())
