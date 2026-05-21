#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import (
    SEMANTIC_IR_SYSTEM_PROMPT,
    read_jsonl,
    read_yaml,
    write_json,
    write_jsonl,
)
from path_config import resolve_gp4_ws


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/dataset_spec.yaml"
DEFAULT_BASE_URL = "http://localhost:20128/v1"


def _resolve_base_url(env: Mapping[str, str]) -> str:
    raw_value = env.get("OPENAI_BASE_URL", "")
    return raw_value.strip() or DEFAULT_BASE_URL


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guarded synthetic generation entrypoint for GP4 dataset batches."
    )
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4"))
    parser.add_argument("--base-url", default=_resolve_base_url(os.environ))
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--report", type=Path, default=Path("reports/generation_report.json"))
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=args.dry_run,
            inputs=[args.seed],
            outputs=[args.output, args.report],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"generation_blocked reason={exc} output={args.output} report={args.report}")
        return 1

    spec = read_yaml(SPEC_PATH)
    seed_rows = read_jsonl(args.seed)
    minimum_seed = int(spec["seed_gate"]["minimum_handwritten_seed_examples"])
    if len(seed_rows) < minimum_seed:
        print(
            "seed gate blocked generation: "
            f"seed_rows={len(seed_rows)} minimum={minimum_seed}. "
            "Add handwritten seed examples and validate them before synthetic generation."
        )
        return 1

    if not os.getenv("OPENAI_API_KEY") and not args.dry_run:
        print("OPENAI_API_KEY is not configured. Use secrets/env only; do not commit keys.")
        return 1

    if args.dry_run:
        print(
            f"dry_run_ok seed_rows={len(seed_rows)} requested={args.count} "
            f"model={args.model} output={args.output}"
        )
        return 0

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    rows = generate_examples(
        seed_rows=seed_rows,
        count=args.count,
        batch_size=args.batch_size,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        contract_repo=_resolve_contract_repo(spec, env=os.environ),
    )
    write_jsonl(args.output, rows)
    write_json(
        args.report,
        {
            "seed_rows": len(seed_rows),
            "requested": args.count,
            "generated": len(rows),
            "model": args.model,
            "base_url": _redact_base_url(args.base_url),
            "output": str(args.output),
        },
    )
    print(f"generated={len(rows)} output={args.output} report={args.report}")
    return 0


def generate_examples(
    *,
    seed_rows: list[dict[str, Any]],
    count: int,
    batch_size: int,
    base_url: str,
    model: str,
    temperature: float,
    contract_repo: Path,
) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    from factory_common import load_repo_contract, parse_single_json_object, validate_semantic_payload

    contract = load_repo_contract(contract_repo)
    generated: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(6, (count // batch_size + 1) * 4)
    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        current_count = min(batch_size, count - len(generated))
        batch_index = attempts
        response = _chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            payload={
                "model": model,
                "messages": _generation_messages(seed_rows, current_count, batch_index),
                "temperature": temperature,
                "max_tokens": 7000,
                "response_format": _response_format_schema(),
            },
        )
        batch_rows = _rows_from_response(response, batch_index)
        for row in batch_rows:
            if len(generated) >= count:
                break
            try:
                assistant_json = parse_single_json_object(row["messages"][-1]["content"])
            except (KeyError, ValueError):
                continue
            if validate_semantic_payload(assistant_json, contract) is not None:
                continue
            generated.append(row)
        time.sleep(0.2)
    if len(generated) < count:
        raise RuntimeError(
            f"generation produced only {len(generated)} valid rows after {attempts} attempts"
        )
    return _renumber_generated_rows(generated[:count])


def _resolve_contract_repo(spec: dict[str, Any], *, env: Mapping[str, str]) -> Path:
    project = spec.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("configs/dataset_spec.yaml project section must be a mapping.")
    env_name = str(project.get("source_repo_env", "GP4_WS")).strip() or "GP4_WS"
    raw_value = (env.get(env_name) or "").strip()
    if not raw_value and env_name == "GP4_WS":
        return resolve_gp4_ws(cli_value=None, env=env)
    if not raw_value:
        raise ValueError(
            f"{env_name} is required; set the environment variable to the GP4 source workspace path."
        )
    if env_name == "GP4_WS":
        return resolve_gp4_ws(cli_value=raw_value, env={})
    return Path(raw_value).expanduser().resolve(strict=False)


def _chat_completion(
    *, base_url: str, api_key: str, model: str, payload: dict[str, Any]
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        if payload.get("response_format", {}).get("type") == "json_schema":
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = {"type": "json_object"}
            return _chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                payload=fallback_payload,
            )
        raise RuntimeError(f"generation request failed: HTTP {exc.code}: {detail}") from exc


def _generation_messages(
    seed_rows: list[dict[str, Any]], count: int, batch_index: int
) -> list[dict[str, str]]:
    examples = "\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in seed_rows[:12]
    )
    return [
        {
            "role": "system",
            "content": SEMANTIC_IR_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Create {count} new diverse GP4 dataset rows for batch {batch_index}. "
                "Use Vietnamese, English, and mixed commands. Include normal, ambiguous, hard-negative, status, and vision_stub rows. "
                "Do not copy seed IDs. IDs can be temporary; the local script will renumber them. "
                "Allowed normal intents only: go_home, stop, alarm_reset, get_pose, set_speed, wait, move_relative, absolute_move_ptp, move_named_pose, absolute_move_lin, circular_move, move_joint, move_joint_delta, move_joints, io_set, draw_shape, draw_text, sequence, return_to_start only inside sequence steps. "
                "Use reference_frame base_link only. For io_set use io_address integer and io_value 0 or 1 only. For move_joint use joint_index 0..5 and joint_angle. For move_joints use joint_target with six numbers. "
                "Allowed safe errors only: MISSING_SLOT, UNSUPPORTED_OR_AMBIGUOUS_COMMAND, UNSAFE_COMMAND, PERCEPTION_REQUIRED, CALIBRATION_REQUIRED. "
                "Never include primitive_type, raw trajectories, ROS commands, MotoROS2 calls, or claims that hardware moved. "
                "Each assistant content must exactly equal JSON.stringify(expected_json) with no markdown.\n\n"
                "Seed examples:\n"
                f"{examples}"
            ),
        },
    ]


def _response_format_schema() -> dict[str, Any]:
    message_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["role", "content"],
        "properties": {
            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
            "content": {"type": "string"},
        },
    }
    row_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "messages", "expected_json", "metadata"],
        "properties": {
            "id": {"type": "string"},
            "messages": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": message_schema,
            },
            "expected_json": {"type": "object"},
            "metadata": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "language",
                    "task_type",
                    "source",
                    "safety_class",
                    "requires_perception",
                ],
                "properties": {
                    "language": {"type": "string", "enum": ["vi", "en", "mixed"]},
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "normal",
                            "ambiguous",
                            "hard_negative",
                            "status",
                            "vision_stub",
                        ],
                    },
                    "source": {"type": "string", "enum": ["synthetic"]},
                    "safety_class": {
                        "type": "string",
                        "enum": [
                            "safe_motion_plan",
                            "safe_query",
                            "safe_setting",
                            "clarification_required",
                            "unsafe_rejected",
                            "perception_required",
                        ],
                    },
                    "requires_perception": {"type": "boolean"},
                },
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "gp4_dataset_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["examples"],
                "properties": {
                    "examples": {
                        "type": "array",
                        "items": row_schema,
                    }
                },
            },
        },
    }


def _rows_from_response(response: dict[str, Any], batch_index: int) -> list[dict[str, Any]]:
    content = response["choices"][0]["message"]["content"]
    decoded = json.loads(content)
    if isinstance(decoded, list):
        rows = decoded
    elif isinstance(decoded, dict):
        rows = decoded.get("examples") or decoded.get("rows")
        if rows is None and decoded.get("error"):
            raise RuntimeError(
                f"generation batch {batch_index} returned error object: {decoded.get('error')}"
            )
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"generation batch {batch_index} returned no examples")
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"generation batch {batch_index} returned a non-object row")
        row["metadata"] = _sanitize_metadata(row.get("metadata", {}))
        row["messages"][0]["content"] = SEMANTIC_IR_SYSTEM_PROMPT
        row["messages"][-1]["content"] = json.dumps(
            row["expected_json"], ensure_ascii=False, separators=(",", ":")
        )
    return rows


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    language = metadata.get("language", "mixed")
    if language not in {"vi", "en", "mixed"}:
        language = "mixed"
    task_type = metadata.get("task_type", "normal")
    if task_type not in {"normal", "ambiguous", "hard_negative", "status", "vision_stub"}:
        task_type = "normal"
    safety_class = metadata.get("safety_class", "safe_motion_plan")
    if safety_class not in {
        "safe_motion_plan",
        "safe_query",
        "safe_setting",
        "clarification_required",
        "unsafe_rejected",
        "perception_required",
    }:
        safety_class = "safe_motion_plan"
    return {
        "language": language,
        "task_type": task_type,
        "source": "synthetic",
        "safety_class": safety_class,
        "requires_perception": bool(metadata.get("requires_perception", False)),
    }


def _renumber_generated_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        language = {"mixed": "mi"}.get(row["metadata"]["language"], row["metadata"]["language"])
        task_type = row["metadata"]["task_type"]
        row["id"] = f"gp4_{language}_{task_type}_{index:06d}"
    return rows


def _redact_base_url(base_url: str) -> str:
    return base_url.replace(os.getenv("OPENAI_API_KEY", ""), "<redacted>")


if __name__ == "__main__":
    raise SystemExit(main())
