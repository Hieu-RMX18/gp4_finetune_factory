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


def test_validate_react_ir_dataset_requires_cloud_or_tmp_report_path(
    tmp_path: Path,
) -> None:
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
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "CLOUD_ROOT" in (result.stdout + result.stderr)
    assert not report_path.exists()


def test_acceptance_gate_fails_when_metrics_below_threshold(tmp_path: Path) -> None:
    eval_report = tmp_path / "eval_report.json"
    gate_report = tmp_path / "acceptance_gate_report_run.json"
    eval_report.write_text(
        json.dumps(
            {
                "rows": 1,
                "heldout_test_rows": 2,
                "json_parse_success": 0.5,
                "react_ir_schema_success": 0.5,
                "intent_accuracy": 0.0,
                "primitive_type_leakage": 0,
                "hardware_claims": 0,
                "raw_trajectory_outputs": 0,
                "ros_motoros_outputs": 0,
                "safety_bypass_outputs": 0,
                "unsafe_command_acceptance": 0,
                "final_adapter_exists": False,
                "local_artifact_usage": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_acceptance_gates.py",
            "--eval-report",
            str(eval_report),
            "--report",
            str(gate_report),
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
            "--min-rows",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(gate_report.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    gate_names = {check["gate"] for check in payload["checks"]}
    assert "react_ir_schema_success_min" in gate_names
    assert "heldout_output_rows_equal_test_rows" in gate_names
    assert "final_adapter_exists" in gate_names
    assert "local_artifact_usage_max" in gate_names

def test_build_quality_report_requires_cloud_report_path(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    input_report = cloud_root / "reports/validation.json"
    output_report = tmp_path / "quality_report.json"
    input_report.parent.mkdir(parents=True)
    input_report.write_text('{"passed": true, "issues": []}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_quality_report.py",
            "--input-report",
            str(input_report),
            "--report",
            str(output_report),
            "--cloud-root",
            str(cloud_root),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "outside CLOUD_ROOT" in (result.stdout + result.stderr)
    assert not output_report.exists()

def test_build_quality_report_writes_cloud_summary(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    input_report = cloud_root / "reports/validation.json"
    output_report = cloud_root / "reports/quality_report.json"
    input_report.parent.mkdir(parents=True)
    input_report.write_text('{"passed": true, "issues": []}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_quality_report.py",
            "--input-report",
            str(input_report),
            "--report",
            str(output_report),
            "--cloud-root",
            str(cloud_root),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output_report.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["reports"][str(input_report)]["passed"] is True

def test_acceptance_gate_requires_cloud_root_for_completion_evidence(
    tmp_path: Path,
) -> None:
    eval_report = tmp_path / "eval_report.json"
    gate_report = tmp_path / "acceptance_gate_report_run.json"
    eval_report.write_text(
        json.dumps(
            {
                "rows": 2,
                "heldout_test_rows": 2,
                "json_parse_success": 1.0,
                "react_ir_schema_success": 1.0,
                "semantic_ir_success": 1.0,
                "intent_accuracy": 1.0,
                "locked_typo_eval_intent_accuracy": 1.0,
                "primitive_type_leakage": 0,
                "hardware_claims": 0,
                "raw_trajectory_outputs": 0,
                "ros_motoros_outputs": 0,
                "safety_bypass_outputs": 0,
                "unsafe_command_acceptance": 0,
                "final_adapter_exists": True,
                "local_artifact_usage": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_acceptance_gates.py",
            "--eval-report",
            str(eval_report),
            "--report",
            str(gate_report),
            "--min-rows",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "CLOUD_ROOT" in (result.stdout + result.stderr)
    assert not gate_report.exists()

def test_acceptance_gate_fails_without_explicit_locked_typo_metric(
    tmp_path: Path,
) -> None:
    eval_report = tmp_path / "eval_report.json"
    gate_report = tmp_path / "acceptance_gate_report_run.json"
    eval_report.write_text(
        json.dumps(
            {
                "rows": 2,
                "heldout_test_rows": 2,
                "json_parse_success": 1.0,
                "react_ir_schema_success": 1.0,
                "semantic_ir_success": 1.0,
                "intent_accuracy": 1.0,
                "primitive_type_leakage": 0,
                "hardware_claims": 0,
                "raw_trajectory_outputs": 0,
                "ros_motoros_outputs": 0,
                "safety_bypass_outputs": 0,
                "unsafe_command_acceptance": 0,
                "final_adapter_exists": True,
                "local_artifact_usage": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_acceptance_gates.py",
            "--eval-report",
            str(eval_report),
            "--report",
            str(gate_report),
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
            "--min-rows",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(gate_report.read_text(encoding="utf-8"))
    failed = {check["gate"] for check in payload["checks"] if not check["passed"]}
    assert "locked_typo_eval_intent_accuracy_min" in failed

def test_package_adapter_rejects_empty_adapter_directory(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    adapter_dir.mkdir(parents=True)
    acceptance_report = cloud_root / "reports/acceptance_gate_report_run.json"
    package_report = cloud_root / "reports/package_report_run.json"
    acceptance_report.parent.mkdir(parents=True)
    acceptance_report.write_text('{"passed": true}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/package_adapter.py",
            "--adapter-dir",
            str(adapter_dir),
            "--acceptance-report",
            str(acceptance_report),
            "--cloud-root",
            str(cloud_root),
            "--report",
            str(package_report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(package_report.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["blocked_reason"] == "adapter artifact files are missing"

def test_package_adapter_accepts_full_model_artifacts(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "model-00001-of-00001.safetensors").write_text(
        "weights\n",
        encoding="utf-8",
    )
    acceptance_report = cloud_root / "reports/acceptance_gate_report_run.json"
    package_report = cloud_root / "reports/package_report_run.json"
    acceptance_report.parent.mkdir(parents=True)
    acceptance_report.write_text('{"passed": true}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/package_adapter.py",
            "--adapter-dir",
            str(adapter_dir),
            "--acceptance-report",
            str(acceptance_report),
            "--cloud-root",
            str(cloud_root),
            "--report",
            str(package_report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(package_report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["adapter_artifact_exists"] is True
