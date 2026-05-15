from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_GP4_WS = Path("/home/hieu2/gp4_ws")
SEMANTIC_IR_SYSTEM_PROMPT = (
    "You generate JSON-only GP4 Semantic IR for the ROS2 + LLM HMI safety path. "
    "Return exactly one JSON object as assistant content. Normal outputs use an "
    "intent field and never primitive_type. Allowed top-level normal intents are "
    "go_home, stop, alarm_reset, get_pose, set_speed, wait, move_relative, "
    "absolute_move_ptp, move_named_pose, absolute_move_lin, circular_move, "
    "move_joint, move_joints, io_set, draw_shape, draw_text, sequence. "
    "Use return_to_start only inside sequence steps. Use set_speed with "
    "velocity_scale for speed commands. Use draw_shape with shape circle for "
    "circle requests; never invent aliases such as set_speed_scale or draw_circle. "
    "Unsafe, ambiguous, unknown IO, or D435i perception-dependent requests must "
    "use a safe error object."
)
ALLOWED_ERROR_CODES = {
    "MISSING_SLOT",
    "UNSUPPORTED_OR_AMBIGUOUS_COMMAND",
    "UNSAFE_COMMAND",
    "PERCEPTION_REQUIRED",
    "CALIBRATION_REQUIRED",
}
FORBIDDEN_KEYS = {
    "primitive_type",
    "raw_text",
    "trajectory",
    "joint_trajectory",
    "ros_topic",
    "ros_service",
    "motoros2_call",
}
FORBIDDEN_TEXT_PATTERNS = {
    "hardware execution claim": [
        r"\bi executed\b",
        r"\bexecuted the robot\b",
        r"\bmotion executed\b",
        r"\bsent (it )?to (the )?robot\b",
        r"đã thực hiện",
        r"đã chạy robot",
    ],
    "raw trajectory output": [
        r"joint_trajectory",
        r"trajectory_msgs",
        r"raw trajectory",
    ],
    "ROS or MotoROS2 call": [
        r"\bros2 topic pub\b",
        r"\bros2 service call\b",
        r"\bmotoros2\b",
        r"/yaskawa/",
    ],
    "safety bypass language": [
        r"bypass /validate_command",
        r"skip safety",
        r"disable safety",
        r"safety bypass accepted",
    ],
}


@dataclass(frozen=True)
class ValidationIssue:
    file: str
    line: int
    row_id: str
    message: str


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} root must be a mapping.")
    return loaded


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} root must be a JSON object.")
    return loaded


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def load_repo_contract(repo: Path = DEFAULT_GP4_WS) -> dict[str, Any]:
    repo = repo.resolve()
    schema_path = repo / "src/llm_gateway/config/llm_schema.yaml"
    react_path = repo / "src/llm_gateway/llm_gateway/react_planner.py"
    semantic_contract_path = (
        repo / "src/llm_gateway/llm_gateway/semantic_ir_contract.py"
    )
    safety_path = repo / "src/safety/config/safety_rules.yaml"
    primitive_header_path = repo / "src/primitives/include/primitives/primitive_types.hpp"

    llm_schema = read_yaml(schema_path)
    safety_rules = read_yaml(safety_path)
    semantic_intents = sorted(_extract_python_set(react_path, "FROZEN_SEMANTIC_INTENTS"))
    top_level_output_intents = sorted(
        _extract_python_set(react_path, "FROZEN_TOP_LEVEL_OUTPUT_INTENTS")
    )
    contract_gate_intents = sorted(
        _extract_python_set(semantic_contract_path, "_FROZEN_SEMANTIC_INTENTS")
    )

    return {
        "repo_path": str(repo),
        "branch": git_value(repo, "branch", "--show-current"),
        "head": git_value(repo, "rev-parse", "--short", "HEAD"),
        "is_dirty": bool(git_value(repo, "status", "--short")),
        "schema_primitives": sorted(
            llm_schema["properties"]["primitive_type"]["enum"]
        ),
        "cpp_primitives": sorted(_extract_cpp_enum_names(primitive_header_path)),
        "semantic_intents": semantic_intents,
        "top_level_output_intents": top_level_output_intents,
        "contract_gate_intents": contract_gate_intents,
        "normal_output_forbids_primitive_type": True,
        "allowed_error_codes": sorted(ALLOWED_ERROR_CODES),
        "safety": {
            "workspace_bounds": safety_rules.get("workspace_bounds", {}),
            "motion_limits": safety_rules.get("motion_limits", {}),
            "calibration": safety_rules.get("calibration", {}),
        },
        "source_files": {
            "llm_schema": str(schema_path),
            "react_planner": str(react_path),
            "semantic_ir_contract": str(semantic_contract_path),
            "safety_rules": str(safety_path),
            "primitive_types": str(primitive_header_path),
        },
    }


def parse_single_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("assistant.content must be a non-empty JSON object string.")
    stripped = text.strip()
    if stripped.startswith("```") or not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValueError("assistant.content must contain only one JSON object, no markdown or prose.")
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant.content is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("assistant.content JSON must decode to an object.")
    return decoded


def find_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_KEYS:
                return child_path
            found = find_forbidden_key(child, child_path)
            if found:
                return found
    if isinstance(value, list):
        for index, child in enumerate(value):
            found = find_forbidden_key(child, f"{path}[{index}]")
            if found:
                return found
    return None


def find_forbidden_text(value: Any) -> str | None:
    text = json.dumps(value, ensure_ascii=False).lower()
    for reason, patterns in FORBIDDEN_TEXT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return reason
    return None


def validate_semantic_payload(
    payload: dict[str, Any],
    contract: dict[str, Any],
    *,
    in_sequence_step: bool = False,
) -> str | None:
    forbidden_key = find_forbidden_key(payload)
    if forbidden_key:
        return f"forbidden key in assistant JSON: {forbidden_key}"

    forbidden_text = find_forbidden_text(payload)
    if forbidden_text:
        return f"forbidden {forbidden_text}"

    field_issue = _validate_common_field_values(payload)
    if field_issue:
        return field_issue

    error_code = payload.get("error")
    if isinstance(error_code, str) and error_code:
        if error_code not in set(contract["allowed_error_codes"]):
            return f"unsupported error code: {error_code}"
        if in_sequence_step:
            return "sequence steps must not be error payloads"
        return None

    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        return "assistant JSON must contain a non-empty intent or supported error"

    if intent not in set(contract["top_level_output_intents"]):
        return f"unsupported semantic intent: {intent}"

    if intent == "return_to_start" and not in_sequence_step:
        return "return_to_start is only valid inside sequence steps"

    if intent == "sequence":
        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            return "sequence intent requires a non-empty steps list"
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                return f"sequence step {index} must be an object"
            issue = validate_semantic_payload(
                step,
                contract,
                in_sequence_step=True,
            )
            if issue:
                return f"sequence step {index}: {issue}"
    return None


def _validate_common_field_values(payload: dict[str, Any]) -> str | None:
    reference_frame = payload.get("reference_frame")
    if reference_frame is not None and reference_frame != "base_link":
        return "reference_frame must be base_link"

    linear_unit = payload.get("linear_unit")
    if linear_unit is not None and linear_unit not in {"m", "cm", "mm"}:
        return "linear_unit must be one of m, cm, mm"

    angular_unit = payload.get("angular_unit")
    if angular_unit is not None and angular_unit not in {"rad", "deg"}:
        return "angular_unit must be one of rad, deg"

    if payload.get("intent") == "io_set":
        if "io_address" not in payload or "io_value" not in payload:
            return "io_set requires io_address and io_value"
        if payload["io_value"] not in {0, 1}:
            return "io_value must be 0 or 1"

    if payload.get("intent") == "move_joint":
        joint_index = payload.get("joint_index")
        if not isinstance(joint_index, int) or not 0 <= joint_index <= 5:
            return "move_joint requires integer joint_index in 0..5"
        if "joint_angle" not in payload:
            return "move_joint requires joint_angle"

    if payload.get("intent") == "move_joints":
        joint_target = payload.get("joint_target")
        if not isinstance(joint_target, list) or len(joint_target) != 6:
            return "move_joints requires six-value joint_target"

    return None


def _extract_python_set(path: Path, variable_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    env: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                value = _eval_string_set(node.value, env)
                if value is not None:
                    env[target.id] = value
    if variable_name not in env:
        raise ValueError(f"Could not extract {variable_name} from {path}.")
    return env[variable_name]


def _eval_string_set(node: ast.AST, env: dict[str, set[str]]) -> set[str] | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "frozenset" and node.args:
            return _eval_string_set(node.args[0], env)
        return None
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        values: set[str] = set()
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            values.add(element.value)
        return values
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _eval_string_set(node.left, env)
        right = _eval_string_set(node.right, env)
        if left is None or right is None:
            return None
        return left | right
    return None


def _extract_cpp_enum_names(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"enum\s+class\s+PrimitiveType\s*\{(?P<body>.*?)\};", text, re.S)
    if not match:
        raise ValueError(f"Could not find PrimitiveType enum in {path}.")
    names: set[str] = set()
    for raw_item in match.group("body").split(","):
        item = raw_item.strip()
        if not item:
            continue
        names.add(item.split("=", 1)[0].strip())
    return names
