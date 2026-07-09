from __future__ import annotations

import json
from typing import Any

from factory_common import SEMANTIC_IR_SYSTEM_PROMPT, find_forbidden_key, find_forbidden_text

IMAGE_TEXT_KEYS = {
    "image",
    "image_url",
    "image_base64",
    "pixels",
    "rgb_frame",
    "depth_frame",
}
DOWNSTREAM_TRACE_STEPS = {
    "execute_motion",
    "send_motoros2",
    "dispatch_trajectory",
}
PERCEPTION_REQUIRED_ERRORS = {
    "CALIBRATION_REQUIRED",
    "PERCEPTION_REQUIRED",
}


def _find_image_text_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in sorted(value):
            if key in IMAGE_TEXT_KEYS:
                return key
            nested_key = _find_image_text_key(value[key])
            if nested_key is not None:
                return nested_key
    if isinstance(value, list):
        for item in value:
            nested_key = _find_image_text_key(item)
            if nested_key is not None:
                return nested_key
    return None


def _requires_unresolved_perception(row: dict[str, Any]) -> bool:
    if "requires_perception" in row:
        return bool(row["requires_perception"])

    target_output = row.get("target_output", {})
    return (
        isinstance(target_output, dict)
        and target_output.get("error") in PERCEPTION_REQUIRED_ERRORS
    )


def structured_context_issues(row: dict[str, Any]) -> list[str]:
    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return ["instruction must be a non-empty string"]

    robot_state = row.get("robot_state", {})
    if not isinstance(robot_state, dict):
        return ["robot_state must be a structured object"]
    image_text_key = _find_image_text_key(robot_state)
    if image_text_key is not None:
        return [f"robot_state must contain structured facts only, not {image_text_key}"]

    scene_state = row.get("scene_state", {})
    if not isinstance(scene_state, dict):
        return ["scene_state must be a structured object"]

    image_text_key = _find_image_text_key(scene_state)
    if image_text_key is not None:
        return [f"scene_state must contain structured facts only, not {image_text_key}"]

    target_output = row.get("target_output", {})
    if isinstance(target_output, dict) and "primitive_type" in target_output:
        return [
            "target_output must be Semantic IR or safe error; primitive_type belongs downstream"
        ]
    if isinstance(target_output, dict):
        forbidden_key = find_forbidden_key(target_output)
        if forbidden_key is not None:
            return [
                "target_output must be Semantic IR or safe error; "
                f"forbidden key belongs downstream: {forbidden_key}"
            ]
        forbidden_text = find_forbidden_text(target_output)
        if forbidden_text is not None:
            return [
                "target_output must be Semantic IR or safe error; "
                f"forbidden {forbidden_text}"
            ]

    trace = row.get("react_trace", [])
    if isinstance(trace, list):
        for step in trace:
            if not isinstance(step, dict):
                continue
            name = str(step.get("step", ""))
            if name in DOWNSTREAM_TRACE_STEPS:
                return [
                    f"react_trace step {name} is downstream executor behavior, not LLM training output"
                ]

    return []


def build_messages_from_structured_row(row: dict[str, Any]) -> list[dict[str, str]]:
    issues = structured_context_issues(row)
    if issues:
        raise ValueError(issues[0])

    user_payload = {
        "instruction": row["instruction"],
        "robot_state": row.get("robot_state", {}),
        "scene_state": row.get("scene_state", {}),
        "reasoning_style": row.get("reasoning_style", "react_ir"),
        "react_trace": row.get("react_trace", []),
    }
    return [
        {"role": "system", "content": SEMANTIC_IR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                user_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                row["target_output"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def row_to_master_example(
    row: dict[str, Any],
    *,
    row_id: str,
    language: str,
    scenario_tags: list[str],
) -> dict[str, Any]:
    messages = build_messages_from_structured_row(row)
    return {
        "id": row_id,
        "instruction": row["instruction"],
        "robot_state": row.get("robot_state", {}),
        "scene_state": row.get("scene_state", {}),
        "reasoning_style": row.get("reasoning_style", "react_ir"),
        "react_trace": row.get("react_trace", []),
        "messages": messages,
        "expected_json": row["target_output"],
        "metadata": {
            "language": language,
            "task_type": "normal",
            "source_dataset": "new",
            "source": "synthetic",
            "safety_class": "safe_motion_plan",
            "requires_perception": _requires_unresolved_perception(row),
            "scenario_tags": scenario_tags,
        },
    }
