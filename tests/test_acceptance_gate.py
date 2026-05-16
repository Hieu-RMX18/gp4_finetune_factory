import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def valid_row() -> dict:
    target = {"intent": "stop"}
    react_ir = {
        "schema_version": "gp4_react_ir_v1",
        "observe": {"user_goal": "stop"},
        "reasoning_summary": "The user requests a stop command.",
        "act": {"intent": "stop"},
        "safety": {"requires_validation": True, "hardware_execution_claim": False},
    }
    return {
        "id": "gp4_en_normal_000001",
        "messages": [
            {"role": "system", "content": "GP4 ReAct-IR system prompt"},
            {"role": "user", "content": "stop"},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ],
        "expected_json": target,
        "react_ir": react_ir,
        "metadata": {
            "language": "en",
            "task_type": "normal",
            "source": "seed",
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
        },
    }


def test_validate_react_ir_dataset_accepts_valid_row(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(input_path, [valid_row()])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(input_path),
            "--report",
            str(report_path),
            "--strict",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["valid"] == 1


def test_validate_react_ir_dataset_rejects_motion_for_unresolved_vision(
    tmp_path: Path,
) -> None:
    row = valid_row()
    row["metadata"]["task_type"] = "vision_stub"
    row["metadata"]["requires_perception"] = True
    input_path = tmp_path / "rows.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(input_path, [row])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(input_path),
            "--report",
            str(report_path),
            "--strict",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "requires_perception rows must use safe_error" in result.stdout
