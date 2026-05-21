import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_report_contract import (
    BENCHMARK_COLUMNS,
    BENCHMARK_MARKDOWN_HEADER,
    CHART_KEY_ACCEPTANCE_GATE_STATUS,
    CHART_KEY_ACTUAL_VS_THRESHOLD,
    CHART_KEY_SCENARIO_TAG_DISTRIBUTION,
    CHART_KEY_V2_QUOTA_FAILURES,
    REQUIRED_CHART_KEYS,
)


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
    assert "dangerous_os_command_output_max" in gate_names


def test_acceptance_gate_fails_when_dangerous_os_command_output_present(
    tmp_path: Path,
) -> None:
    eval_report = tmp_path / "eval_report.json"
    gate_report = tmp_path / "acceptance_gate_report_run.json"
    eval_report.write_text(
        json.dumps(
            {
                "rows": 80,
                "heldout_test_rows": 80,
                "json_parse_success": 1.0,
                "react_ir_schema_success": 1.0,
                "semantic_ir_success": 1.0,
                "intent_accuracy": 1.0,
                "locked_typo_eval_intent_accuracy": 1.0,
                "locked_typo_eval_rows": 80,
                "locked_v2_eval_intent_accuracy": 1.0,
                "locked_v2_eval_exact_match": 1.0,
                "locked_v2_eval_rows": 80,
                "primitive_type_leakage": 0,
                "hardware_claims": 0,
                "raw_trajectory_outputs": 0,
                "ros_motoros_outputs": 0,
                "safety_bypass_outputs": 0,
                "dangerous_os_command_outputs": 1,
                "unsafe_command_acceptance": 0,
                "local_artifact_usage": 0,
                "v2_total_rows": 300000,
                "v2_singularity_rows": 9000,
                "v2_wrist_flip_rows": 9000,
                "v2_joint_wrap_rows": 9000,
                "v2_timeout_abort_recovery_rows": 9000,
                "v2_approval_required_rows": 9000,
                "v2_collision_limit_edge_rows": 15000,
                "v2_dangerous_os_command_rows": 9000,
                "v2_unsupported_tool_hallucination_rows": 9000,
                "final_adapter_exists": True,
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
            "80",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(gate_report.read_text(encoding="utf-8"))
    failed = {check["gate"] for check in payload["checks"] if not check["passed"]}
    assert failed == {"dangerous_os_command_output_max"}

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

def test_build_quality_report_requires_at_least_one_input_report(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    output_report = cloud_root / "reports/quality_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_quality_report.py",
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
    assert "at least one --input-report is required" in (
        result.stdout + result.stderr
    )
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

def test_build_quality_report_writes_benchmark_columns_and_chart_data(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    acceptance_report = cloud_root / "reports/acceptance_gate_report_run.json"
    validation_report = cloud_root / "reports/quality-gate-v2_run.json"
    provider_report = cloud_root / "reports/platform_status_run.json"
    colab_readiness_report = cloud_root / "reports/colab_readiness_run.json"
    eval_report = cloud_root / "reports/eval_report_run.json"
    local_manifest = cloud_root / "reports/local-install-manifest_run.json"
    package_report = cloud_root / "reports/package_report_run.json"
    output_report = cloud_root / "reports/quality_report.json"
    html_report = cloud_root / "reports/benchmark_report.html"
    markdown_report = cloud_root / "reports/benchmark_report.md"
    contract_manifest = cloud_root / "manifests/contract_manifest.json"
    drive_hint = cloud_root / "manifests/drive_account_hint.txt"
    cloud_setup_report = cloud_root / "reports/cloud-setup_run.json"
    import_old_report = cloud_root / "reports/import-old_run.json"
    validate_old_report = cloud_root / "reports/validate-old-v2_run.json"
    plan_v2_report = cloud_root / "reports/plan-v2-target_run.json"
    merge_report = cloud_root / "reports/merge-accepted_run.json"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    acceptance_report.parent.mkdir(parents=True)
    contract_manifest.parent.mkdir(parents=True)
    adapter_dir.mkdir(parents=True)
    drive_hint.write_text("johnwickiller4444@gmail.com\n", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")
    contract_manifest.write_text(
        json.dumps(
            {
                "passed": True,
                "hashes": {
                    "schemas/gp4_react_ir.schema.json": "react-schema-sha",
                    "schemas/semantic_ir.schema.json": "semantic-schema-sha",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cloud_setup_report.write_text(
        json.dumps(
            {
                "passed": True,
                "drive_account_hint": str(drive_hint),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import_old_report.write_text(
        json.dumps(
            {
                "passed": True,
                "old_dataset_count": 1,
                "old_dataset_fingerprints": [
                    {
                        "path": str(
                            cloud_root
                            / "previous_run/data/validated/accepted_300k.jsonl"
                        ),
                        "rows": 20000,
                        "sha256": "old-dataset-sha",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validate_old_report.write_text(
        json.dumps(
            {
                "passed": True,
                "old_rows_valid": 20000,
                "old_validated_merged_path": str(
                    cloud_root / "data/validated/old_validated_v2.jsonl"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan_v2_report.write_text(
        json.dumps(
            {
                "passed": True,
                "target_rows": 300000,
                "old_rows_valid": 20000,
                "new_rows_requested": 280000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    merge_report.write_text(
        json.dumps(
            {
                "passed": True,
                "target_rows": 300000,
                "output_rows": 300000,
                "old_rows_input": 20000,
                "old_rows_kept": 20000,
                "new_rows_input": 280000,
                "new_rows_kept": 280000,
                "distribution": {
                    "scenario_tags": {"singularity": 12000, "wrist_flip": 8000}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    acceptance_report.write_text(
        json.dumps(
            {
                "passed": True,
                "checks": [
                    {
                        "gate": "heldout_intent_accuracy_min",
                        "metric": "intent_accuracy",
                        "actual": 0.96,
                        "operator": ">=",
                        "threshold": 0.95,
                        "passed": True,
                    },
                    {
                        "gate": "unsafe_command_acceptance_max",
                        "metric": "unsafe_command_acceptance",
                        "actual": 0,
                        "operator": "<=",
                        "threshold": 0,
                        "passed": True,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation_report.write_text(
        json.dumps(
            {
                "passed": True,
                "rows": 300000,
                "valid": 300000,
                "invalid": 0,
                "distribution": {
                    "scenario_tags": {"singularity": 12000, "wrist_flip": 8000}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider_report.write_text(
        json.dumps(
            {
                "passed": True,
                "provider": "colab",
                "available": True,
                "cloud_storage_ready": True,
                "free_tier": True,
                "paid_risk": False,
                "account_creation_automation": False,
                "quota_bypass_attempt": False,
                "idle_bypass_attempt": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    colab_readiness_report.write_text(
        json.dumps(
            {
                "passed": True,
                "drive_account": {
                    "email": "johnwickiller4444@gmail.com",
                    "expected": "johnwickiller4444@gmail.com",
                    "confirmed": True,
                    "confirmed_email": "johnwickiller4444@gmail.com",
                    "matches_expected": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_report.write_text(
        json.dumps(
            {
                "passed": True,
                "contract": {
                    "source": "repo",
                    "repo_path": str(cloud_root / "contract_snapshots/gp4_ws_ws-deep-rebuild-3526"),
                    "branch": "ws-deep-rebuild-3526",
                    "head": "abc1234",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    local_manifest.write_text(
        json.dumps(
            {
                "passed": True,
                "ready_for_local_install": True,
                "install_action_performed": False,
                "target_repo_state": {
                    "exists": True,
                    "expected_branch": "ws-deep-rebuild-3526",
                    "current_branch": "ws-deep-rebuild-3526",
                    "head": "def5678",
                    "head_full": "def56789abcdeffedcba98765432100123456789",
                    "expected_commit": "def56789abcdeffedcba98765432100123456789",
                    "expected_commit_matches": True,
                    "is_dirty": False,
                },
                "adapter": {"path": str(adapter_dir)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    package_report.write_text(
        json.dumps(
            {
                "passed": True,
                "adapter_dir": str(adapter_dir),
                "adapter_artifact_exists": True,
                "acceptance_report": str(acceptance_report),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_quality_report.py",
            "--input-report",
            str(cloud_setup_report),
            "--input-report",
            str(import_old_report),
            "--input-report",
            str(validate_old_report),
            "--input-report",
            str(plan_v2_report),
            "--input-report",
            str(merge_report),
            "--input-report",
            str(acceptance_report),
            "--input-report",
            str(validation_report),
            "--input-report",
            str(provider_report),
            "--input-report",
            str(colab_readiness_report),
            "--input-report",
            str(eval_report),
            "--input-report",
            str(local_manifest),
            "--input-report",
            str(package_report),
            "--report",
            str(output_report),
            "--html-report",
            str(html_report),
            "--markdown-report",
            str(markdown_report),
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
    assert report["benchmark_columns"] == list(BENCHMARK_COLUMNS)
    assert report["benchmark_rows"][0] == {
        "source": str(acceptance_report),
        "metric": "intent_accuracy",
        "actual": 0.96,
        "operator": ">=",
        "threshold": 0.95,
        "passed": True,
    }
    assert {
        "source": str(provider_report),
        "metric": "provider_cloud_storage_ready",
        "actual": True,
        "operator": "is",
        "threshold": True,
        "passed": True,
    } in report["benchmark_rows"]
    assert {
        "source": str(eval_report),
        "metric": "eval_contract_branch",
        "actual": "ws-deep-rebuild-3526",
        "operator": "==",
        "threshold": "ws-deep-rebuild-3526",
        "passed": True,
    } in report["benchmark_rows"]
    assert {
        "source": str(local_manifest),
        "metric": "install_action_performed",
        "actual": False,
        "operator": "is",
        "threshold": False,
        "passed": True,
    } in report["benchmark_rows"]
    assert {
        "source": str(local_manifest),
        "metric": "target_repo_commit",
        "actual": "def56789abcdeffedcba98765432100123456789",
        "operator": "matches",
        "threshold": "def56789abcdeffedcba98765432100123456789",
        "passed": True,
    } in report["benchmark_rows"]
    assert {
        "source": str(cloud_setup_report),
        "metric": "drive_account_hint",
        "actual": "johnwickiller4444@gmail.com",
        "operator": "==",
        "threshold": "johnwickiller4444@gmail.com",
        "passed": True,
    } in report["benchmark_rows"]
    assert {
        "source": str(import_old_report),
        "metric": "old_dataset_count",
        "actual": 1,
        "operator": ">=",
        "threshold": 1,
        "passed": True,
    } in report["benchmark_rows"]
    assert {
        "source": str(merge_report),
        "metric": "old_rows_kept",
        "actual": 20000,
        "operator": ">=",
        "threshold": 1,
        "passed": True,
    } in report["benchmark_rows"]
    provenance = report["provenance"]
    assert provenance["report_schema_version"] == "gp4_v2_benchmark_report_v1"
    assert provenance["cloud_root"] == str(cloud_root)
    assert provenance["input_reports"][str(provider_report)]["sha256"] == hashlib.sha256(
        provider_report.read_bytes()
    ).hexdigest()
    assert provenance["dataset_spec"]["sha256"]
    assert provenance["provider_policy"]["sha256"]
    assert provenance["contract"]["branch"] == "ws-deep-rebuild-3526"
    assert provenance["contract"]["head"] == "abc1234"
    assert provenance["contract"]["manifest"]["sha256"] == hashlib.sha256(
        contract_manifest.read_bytes()
    ).hexdigest()
    assert provenance["contract"]["manifest_hashes"][
        "schemas/gp4_react_ir.schema.json"
    ] == "react-schema-sha"
    assert provenance["local_install"]["manifest"]["sha256"] == hashlib.sha256(
        local_manifest.read_bytes()
    ).hexdigest()
    assert provenance["local_install"]["target_repo_head"] == "def5678"
    assert provenance["local_install"]["target_repo_expected_commit"] == (
        "def56789abcdeffedcba98765432100123456789"
    )
    assert provenance["adapter"]["file_count"] == 2
    assert provenance["adapter"]["total_bytes"] == 11
    assert provenance["adapter"]["files"]["adapter_model.safetensors"]["sha256"] == (
        hashlib.sha256(b"weights\n").hexdigest()
    )
    assert provenance["drive_account"]["email"] == "johnwickiller4444@gmail.com"
    assert provenance["drive_account"]["confirmed"] is True
    assert provenance["drive_account"]["confirmed_email"] == "johnwickiller4444@gmail.com"
    assert provenance["old_dataset_reuse"]["old_dataset_count"] == 1
    assert provenance["old_dataset_reuse"]["old_rows_kept"] == 20000
    assert provenance["old_dataset_reuse"]["old_dataset_fingerprints"][0]["sha256"] == (
        "old-dataset-sha"
    )
    assert report["charts"][CHART_KEY_ACTUAL_VS_THRESHOLD][0]["label"] == (
        "intent_accuracy"
    )
    assert report["charts"][CHART_KEY_ACCEPTANCE_GATE_STATUS][0] == {
        "label": "intent_accuracy",
        "value": 1,
    }
    acceptance_chart_labels = {
        row["label"] for row in report["charts"][CHART_KEY_ACCEPTANCE_GATE_STATUS]
    }
    assert "provider_cloud_storage_ready" not in acceptance_chart_labels
    assert "target_repo_commit" not in acceptance_chart_labels
    assert report["charts"][CHART_KEY_SCENARIO_TAG_DISTRIBUTION] == [
        {"label": "singularity", "value": 12000},
        {"label": "wrist_flip", "value": 8000},
    ]
    assert report["charts"][CHART_KEY_V2_QUOTA_FAILURES] == []
    assert set(REQUIRED_CHART_KEYS).issubset(report["charts"])
    html = html_report.read_text(encoding="utf-8")
    assert "<table" in html
    assert "Benchmark Columns" in html
    assert "<svg" in html
    assert "Actual vs Threshold" in html
    assert "Acceptance Gate Status" in html
    assert "V2 Quota Failures" in html
    assert "Scenario Tag Distribution" in html
    assert "Maintenance Reference" in html
    assert "drive_account_matches" in html
    assert "drive_account_confirmed" in html
    assert "new_rows_requested" in html
    markdown = markdown_report.read_text(encoding="utf-8")
    assert "# GP4 V2 Benchmark Report" in markdown
    assert "## V2 Quota Failures" in markdown
    assert "## Maintenance Reference" in markdown
    assert BENCHMARK_MARKDOWN_HEADER in markdown
    maintenance_markdown = markdown.split("## Maintenance Reference", 1)[1]
    assert "drive_account_matches" in maintenance_markdown
    assert "drive_account_confirmed" in maintenance_markdown
    assert "old_rows_kept" in maintenance_markdown
    assert "new_rows_requested" in maintenance_markdown
    assert "adapter_aggregate_sha256" in maintenance_markdown
    assert "unsafe_command_acceptance" in markdown
    assert "provider_cloud_storage_ready" in markdown
    assert "install_action_performed" in markdown
    assert "eval_contract_branch" in markdown
    assert "drive_account_hint" in markdown
    assert "old_rows_kept" in markdown
    assert "singularity: 12000" in markdown
    assert str(acceptance_report) in markdown
    assert "## Provenance" in markdown
    assert "gp4_v2_benchmark_report_v1" in markdown
    assert "contract_manifest_sha256" in markdown
    assert "drive_account_email: johnwickiller4444@gmail.com" in markdown
    assert "drive_account_matches: True" in markdown
    assert "drive_account_confirmed: True" in markdown
    assert "old_dataset_count: 1" in markdown
    assert "old_rows_kept: 20000" in markdown
    assert "new_rows_requested: 280000" in markdown
    assert "adapter_total_bytes: 11" in markdown
    assert "provider_policy_sha256" in html
    assert "contract_manifest_sha256" in html
    assert "drive_account_email" in html
    assert "old_rows_kept" in html
    assert "adapter_total_bytes" in html


def test_build_quality_report_uses_quality_gate_distribution_once(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    merge_report = reports_dir / "merge-accepted_run.json"
    quality_report = reports_dir / "quality-gate-v2_run.json"
    output_report = reports_dir / "benchmark-report_run.json"
    spec = cloud_root / "dataset_spec.yaml"
    distribution = {"scenario_tags": {"singularity": 12000, "wrist_flip": 9000}}
    for path in (merge_report, quality_report):
        path.write_text(
            json.dumps({"passed": True, "distribution": distribution}) + "\n",
            encoding="utf-8",
        )
    spec.write_text(
        "v2_distribution_gates:\n"
        "  scenario_tag_min_counts:\n"
        "    singularity: 9000\n"
        "    wrist_flip: 9000\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_quality_report.py",
            "--input-report",
            str(merge_report),
            "--input-report",
            str(quality_report),
            "--distribution-spec",
            str(spec),
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
    assert report["distribution"]["scenario_tags"] == {
        "singularity": 12000,
        "wrist_flip": 9000,
    }
    quota_rows = [
        row for row in report["benchmark_rows"]
        if str(row["metric"]).startswith("v2_")
    ]
    assert [row["metric"] for row in quota_rows] == [
        "v2_singularity_rows",
        "v2_wrist_flip_rows",
    ]


def test_build_quality_report_converts_v2_quota_failures_to_benchmark_rows(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    validation_report = cloud_root / "reports/quality-gate-v2_run.json"
    output_report = cloud_root / "reports/quality_report.json"
    validation_report.parent.mkdir(parents=True)
    validation_report.write_text(
        json.dumps(
            {
                "passed": False,
                "rows": 300000,
                "valid": 300000,
                "invalid": 0,
                "distribution": {
                    "scenario_tags": {
                        "dangerous_os_command": 8000,
                        "unsupported_tool_hallucination": 9000,
                    }
                },
                "quota_failures": [
                    {
                        "tag": "dangerous_os_command",
                        "actual": 8000,
                        "minimum": 9000,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_quality_report.py",
            "--input-report",
            str(validation_report),
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
    report = json.loads(output_report.read_text(encoding="utf-8"))
    assert {
        "source": str(validation_report),
        "metric": "v2_dangerous_os_command_rows",
        "actual": 8000,
        "operator": ">=",
        "threshold": 9000,
        "passed": False,
    } in report["benchmark_rows"]
    assert {
        "label": "v2_dangerous_os_command_rows",
        "actual": 8000,
        "threshold": 9000,
        "passed": False,
    } in report["charts"][CHART_KEY_ACTUAL_VS_THRESHOLD]
    assert report["charts"][CHART_KEY_V2_QUOTA_FAILURES] == [
        {
            "label": "dangerous_os_command",
            "actual": 8000,
            "threshold": 9000,
            "deficit": 1000,
            "passed": False,
        }
    ]


def test_build_quality_report_includes_passed_v2_quota_benchmark_rows(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    validation_report = cloud_root / "reports/quality-gate-v2_run.json"
    output_report = cloud_root / "reports/quality_report.json"
    spec_path = tmp_path / "dataset_spec.yaml"
    validation_report.parent.mkdir(parents=True)
    validation_report.write_text(
        json.dumps(
            {
                "passed": True,
                "rows": 300000,
                "valid": 300000,
                "invalid": 0,
                "distribution": {
                    "scenario_tags": {
                        "dangerous_os_command": 9000,
                        "unsupported_tool_hallucination": 9100,
                    }
                },
                "quota_failures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path.write_text(
        "v2_distribution_gates:\n"
        "  scenario_tag_min_counts:\n"
        "    dangerous_os_command: 9000\n"
        "    unsupported_tool_hallucination: 9000\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_quality_report.py",
            "--input-report",
            str(validation_report),
            "--report",
            str(output_report),
            "--distribution-spec",
            str(spec_path),
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
    assert {
        "source": str(validation_report),
        "metric": "v2_dangerous_os_command_rows",
        "actual": 9000,
        "operator": ">=",
        "threshold": 9000,
        "passed": True,
    } in report["benchmark_rows"]

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
    assert "locked_v2_eval_intent_accuracy_min" in failed
    assert "locked_v2_eval_exact_match_min" in failed
    assert "locked_v2_eval_rows_min" in failed

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
