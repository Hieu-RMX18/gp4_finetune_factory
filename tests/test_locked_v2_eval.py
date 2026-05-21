import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from locked_v2_eval import build_locked_v2_eval_rows, write_locked_v2_eval


def test_locked_v2_eval_contains_required_robot_risk_tags(tmp_path: Path) -> None:
    rows = build_locked_v2_eval_rows()
    tags = {
        tag
        for row in rows
        for tag in row["metadata"].get("scenario_tags", [])
    }

    assert {
        "singularity",
        "wrist_flip",
        "joint_wrap",
        "timeout_abort_recovery",
        "approval_required",
        "collision_limit_edge",
        "dangerous_os_command",
        "unsupported_tool_hallucination",
    } <= tags
    assert len(rows) >= 80
    assert Counter(
        tag
        for row in rows
        for tag in row["metadata"].get("scenario_tags", [])
    ) == {
        "singularity": 10,
        "wrist_flip": 10,
        "joint_wrap": 10,
        "timeout_abort_recovery": 10,
        "approval_required": 10,
        "collision_limit_edge": 10,
        "dangerous_os_command": 10,
        "unsupported_tool_hallucination": 10,
    }


def test_locked_v2_eval_labels_ascii_vietnamese_prompts_as_vi() -> None:
    rows = build_locked_v2_eval_rows()
    vi_prompts = [
        row
        for row in rows
        if "quay truc 6" in row["messages"][1]["content"]
        or "bo qua collision" in row["messages"][1]["content"]
    ]

    assert vi_prompts
    assert {row["metadata"]["language"] for row in vi_prompts} == {"vi"}


def test_locked_v2_eval_marks_unsafe_rows_and_safe_wait_rows() -> None:
    rows = build_locked_v2_eval_rows()
    unsafe_rows = [row for row in rows if "error" in row["expected_json"]]
    safe_rows = [row for row in rows if row["expected_json"].get("intent") == "wait"]

    assert unsafe_rows
    assert safe_rows
    assert {row["metadata"]["task_type"] for row in unsafe_rows} == {"hard_negative"}
    assert {row["metadata"]["safety_class"] for row in unsafe_rows} == {"unsafe_rejected"}
    assert {row["metadata"]["task_type"] for row in safe_rows} == {"normal"}
    assert {row["metadata"]["safety_class"] for row in safe_rows} == {"safe_motion_plan"}


def test_write_locked_v2_eval_writes_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "locked_v2_eval.jsonl"

    write_locked_v2_eval(output)

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) >= 80
    assert rows[0]["metadata"]["source"] == "eval"
    assert rows[0]["metadata"]["source_dataset"] == "locked_eval"


def test_locked_v2_eval_prompts_are_unique() -> None:
    rows = build_locked_v2_eval_rows()
    prompts = [row["messages"][1]["content"] for row in rows]

    assert len(prompts) == len(set(prompts)), (
        f"locked-v2 eval has {len(prompts)} rows but only {len(set(prompts))} unique prompts"
    )
