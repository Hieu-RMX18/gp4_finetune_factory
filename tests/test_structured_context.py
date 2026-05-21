import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from structured_context import (
    build_messages_from_structured_row,
    row_to_master_example,
    structured_context_issues,
)


def _structured_row() -> dict:
    return {
        "instruction": "Di chuyen toi vat mau do hinh hop.",
        "robot_state": {
            "ready": True,
            "tcp_pose": [0.25, 0.0, 0.35],
        },
        "scene_state": {
            "objects": [
                {
                    "id": "obj_01",
                    "color": "red",
                    "shape": "box",
                    "pose_base_link": [0.31, 0.12, 0.24],
                    "confidence": 0.91,
                }
            ]
        },
        "reasoning_style": "react_ir",
        "react_trace": [
            {"step": "query_perception", "args": {"color": "red", "shape": "box"}},
            {"step": "select_object", "object_id": "obj_01"},
            {"step": "propose_motion", "target_pose_source": "perception"},
            {"step": "require_validation", "gate": "validate_command"},
        ],
        "target_output": {
            "intent": "absolute_move_lin",
            "target_object_id": "obj_01",
            "target_pose_source": "perception",
            "velocity_scale": 0.05,
            "require_approval": True,
        },
    }


def test_structured_context_builds_chat_messages_without_image_text() -> None:
    row = _structured_row()

    messages = build_messages_from_structured_row(row)

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    user_payload = json.loads(messages[1]["content"])
    assistant_payload = json.loads(messages[2]["content"])
    assert user_payload == {
        "instruction": "Di chuyen toi vat mau do hinh hop.",
        "react_trace": [
            {"args": {"color": "red", "shape": "box"}, "step": "query_perception"},
            {"object_id": "obj_01", "step": "select_object"},
            {"step": "propose_motion", "target_pose_source": "perception"},
            {"gate": "validate_command", "step": "require_validation"},
        ],
        "reasoning_style": "react_ir",
        "robot_state": {
            "ready": True,
            "tcp_pose": [0.25, 0.0, 0.35],
        },
        "scene_state": {
            "objects": [
                {
                    "color": "red",
                    "confidence": 0.91,
                    "id": "obj_01",
                    "pose_base_link": [0.31, 0.12, 0.24],
                    "shape": "box",
                }
            ]
        },
    }
    assert assistant_payload == {
        "intent": "absolute_move_lin",
        "require_approval": True,
        "target_object_id": "obj_01",
        "target_pose_source": "perception",
        "velocity_scale": 0.05,
    }
    assert "image" not in messages[1]["content"].lower()
    assert "primitive_type" not in assistant_payload


def test_structured_context_rejects_image_text_payloads() -> None:
    row = _structured_row()
    row["scene_state"]["image_base64"] = "abc123"

    assert structured_context_issues(row) == [
        "scene_state must contain structured facts only, not image_base64"
    ]


def test_structured_context_rejects_nested_image_text_payloads() -> None:
    row = _structured_row()
    row["scene_state"]["objects"][0]["image_url"] = "https://example.test/frame.png"

    assert structured_context_issues(row) == [
        "scene_state must contain structured facts only, not image_url"
    ]


def test_structured_context_rejects_robot_state_image_text_payloads() -> None:
    row = _structured_row()
    row["robot_state"]["image_base64"] = "abc123"

    assert structured_context_issues(row) == [
        "robot_state must contain structured facts only, not image_base64"
    ]


@pytest.mark.parametrize("instruction", [None, 123, "", "   "])
def test_structured_context_rejects_invalid_instruction(instruction: object) -> None:
    row = _structured_row()
    row["instruction"] = instruction

    assert structured_context_issues(row) == [
        "instruction must be a non-empty string"
    ]


@pytest.mark.parametrize("robot_state", ["not-an-object", ["ready"]])
def test_structured_context_rejects_non_object_robot_state(
    robot_state: object,
) -> None:
    row = _structured_row()
    row["robot_state"] = robot_state

    assert structured_context_issues(row) == [
        "robot_state must be a structured object"
    ]


@pytest.mark.parametrize("scene_state", ["not-an-object", ["objects"]])
def test_structured_context_rejects_non_object_scene_state(
    scene_state: object,
) -> None:
    row = _structured_row()
    row["scene_state"] = scene_state

    assert structured_context_issues(row) == [
        "scene_state must be a structured object"
    ]


def test_build_messages_raises_for_malformed_structured_row() -> None:
    row = _structured_row()
    row["robot_state"] = "not-an-object"

    with pytest.raises(ValueError, match="robot_state must be a structured object"):
        build_messages_from_structured_row(row)


def test_structured_context_rejects_model_owned_execution_actions() -> None:
    row = _structured_row()
    row["react_trace"].append({"step": "execute_motion"})

    assert structured_context_issues(row) == [
        "react_trace step execute_motion is downstream executor behavior, not LLM training output"
    ]


def test_structured_context_rejects_primitive_type_target_output() -> None:
    row = _structured_row()
    row["target_output"] = {
        "primitive_type": "PTP",
        "target_object_id": "obj_01",
    }

    assert structured_context_issues(row) == [
        "target_output must be Semantic IR or safe error; primitive_type belongs downstream"
    ]


def test_structured_context_rejects_executor_keys_in_target_output() -> None:
    row = _structured_row()
    row["target_output"] = {
        "intent": "absolute_move_lin",
        "motoros2_call": "/hw_adapter/dispatch_trajectory",
    }

    assert structured_context_issues(row) == [
        "target_output must be Semantic IR or safe error; forbidden key belongs downstream: $.motoros2_call"
    ]


def test_row_to_master_example_preserves_structured_fields_and_expected_json() -> None:
    row = _structured_row()

    master = row_to_master_example(
        row,
        row_id="gp4_vi_structured_000001",
        language="vi",
        scenario_tags=["vision_uncertain", "approval_required"],
    )

    assert master["id"] == "gp4_vi_structured_000001"
    assert master["instruction"] == row["instruction"]
    assert master["robot_state"] == row["robot_state"]
    assert master["scene_state"] == row["scene_state"]
    assert master["reasoning_style"] == "react_ir"
    assert master["react_trace"] == row["react_trace"]
    assert master["expected_json"] == row["target_output"]
    assert master["metadata"]["requires_perception"] is False
    assert master["metadata"]["scenario_tags"] == ["vision_uncertain", "approval_required"]
    assert master["messages"] == build_messages_from_structured_row(row)
