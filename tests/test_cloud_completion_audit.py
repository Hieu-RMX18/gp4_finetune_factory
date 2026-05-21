import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_cloud_completion import REQUIRED_PHASES, audit_completion
from benchmark_report_contract import (
    BENCHMARK_COLUMNS,
    BENCHMARK_MARKDOWN_HEADER,
    CHART_KEY_ACCEPTANCE_GATE_STATUS,
    CHART_KEY_ACTUAL_VS_THRESHOLD,
    CHART_KEY_SCENARIO_TAG_DISTRIBUTION,
    CHART_KEY_V2_QUOTA_FAILURES,
)

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def complete_acceptance_checks() -> list[dict[str, object]]:
    return [
        {"gate": "heldout_rows_min", "actual": 3000, "passed": True},
        {"gate": "json_parse_success_min", "actual": 0.995, "passed": True},
        {"gate": "heldout_intent_accuracy_min", "actual": 0.96, "passed": True},
        {"gate": "semantic_ir_schema_success_min", "actual": 0.99, "passed": True},
        {"gate": "react_ir_schema_success_min", "actual": 0.99, "passed": True},
        {"gate": "primitive_type_leakage_normal_max", "actual": 0, "passed": True},
        {"gate": "hardware_execution_claim_max", "actual": 0, "passed": True},
        {"gate": "raw_trajectory_output_max", "actual": 0, "passed": True},
        {"gate": "ros_motoros_call_output_max", "actual": 0, "passed": True},
        {"gate": "safety_bypass_output_max", "actual": 0, "passed": True},
        {"gate": "dangerous_os_command_output_max", "actual": 0, "passed": True},
        {"gate": "unsafe_command_acceptance_max", "actual": 0, "passed": True},
        {"gate": "local_artifact_usage_max", "actual": 0, "passed": True},
        {
            "gate": "locked_typo_eval_intent_accuracy_min",
            "actual": 0.96,
            "passed": True,
        },
        {"gate": "locked_typo_eval_rows_min", "actual": 50, "passed": True},
        {
            "gate": "locked_v2_eval_intent_accuracy_min",
            "actual": 0.96,
            "passed": True,
        },
        {
            "gate": "locked_v2_eval_exact_match_min",
            "actual": 0.96,
            "passed": True,
        },
        {"gate": "locked_v2_eval_rows_min", "actual": 80, "passed": True},
        {"gate": "v2_total_rows_min", "actual": 300000, "passed": True},
        {"gate": "v2_singularity_rows_min", "actual": 9000, "passed": True},
        {"gate": "v2_wrist_flip_rows_min", "actual": 9000, "passed": True},
        {"gate": "v2_joint_wrap_rows_min", "actual": 9000, "passed": True},
        {
            "gate": "v2_timeout_abort_recovery_rows_min",
            "actual": 9000,
            "passed": True,
        },
        {"gate": "v2_approval_required_rows_min", "actual": 9000, "passed": True},
        {"gate": "v2_collision_limit_edge_rows_min", "actual": 15000, "passed": True},
        {
            "gate": "v2_dangerous_os_command_rows_min",
            "actual": 9000,
            "passed": True,
        },
        {
            "gate": "v2_unsupported_tool_hallucination_rows_min",
            "actual": 9000,
            "passed": True,
        },
        {"gate": "heldout_output_rows_equal_test_rows", "actual": 3000, "passed": True},
        {"gate": "final_adapter_exists", "actual": True, "passed": True},
    ]

def write_cloud_setup_evidence(cloud_root: Path, run_id: str) -> None:
    source_plan = cloud_root / "manifests" / "source_plan.md"
    source_plan.parent.mkdir(parents=True)
    source_plan.write_text("# GP4 cloud fine-tune source plan\n", encoding="utf-8")
    drive_hint = cloud_root / "manifests" / "drive_account_hint.txt"
    drive_hint.write_text("johnwickiller4444@gmail.com\n", encoding="utf-8")
    source_plan_sha = hashlib.sha256(source_plan.read_bytes()).hexdigest()
    write_json(
        cloud_root / "reports" / f"cloud-setup_{run_id}.json",
        {
            "passed": True,
            "source_plan": str(source_plan),
            "source_plan_sha256": source_plan_sha,
            "drive_account_hint": str(drive_hint),
        },
    )

def write_colab_readiness_evidence(cloud_root: Path, run_id: str) -> None:
    gp4_ws_path = cloud_root / "contract_snapshots/gp4_ws_ws-deep-rebuild-3526"
    gp4_ws_path.mkdir(parents=True, exist_ok=True)
    previous_root = cloud_root.parent / "previous_run"
    old_dataset = previous_root / "data/validated/accepted_300k.jsonl"
    old_dataset.parent.mkdir(parents=True, exist_ok=True)
    old_dataset.write_text('{"id":"old-1"}\n', encoding="utf-8")
    previous_adapter = previous_root / "models/qwen25_gp4_lora"
    previous_adapter.mkdir(parents=True, exist_ok=True)
    (previous_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (previous_adapter / "adapter_model.safetensors").write_text(
        "weights\n",
        encoding="utf-8",
    )
    write_json(
        cloud_root / "reports" / f"colab_readiness_{run_id}.json",
        {
            "passed": True,
            "run_id": run_id,
            "cloud_root": str(cloud_root),
            "drive_account": {
                "email": "johnwickiller4444@gmail.com",
                "expected": "johnwickiller4444@gmail.com",
                "confirmed": True,
                "confirmed_email": "johnwickiller4444@gmail.com",
                "matches_expected": True,
            },
            "gp4_ws": {
                "path": str(gp4_ws_path),
                "exists": True,
                "allowed_cloud_path": True,
                "branch": "ws-deep-rebuild-3526",
                "expected_branch": "ws-deep-rebuild-3526",
                "head": "abc1234def56",
                "expected_commit": "abc1234def56",
                "expected_commit_matches": True,
                "is_dirty": False,
            },
            "old_dataset": {
                "path": str(old_dataset),
                "exists": True,
                "rows": 1,
                "allowed_cloud_path": True,
            },
            "previous_adapter": {
                "path": str(previous_adapter),
                "exists": True,
                "allowed_cloud_path": True,
                "artifact_exists": True,
            },
            "previous_run": {
                "drive_root": str(cloud_root.parent),
                "old_dataset_run_id": "previous_run",
                "previous_adapter_run_id": "previous_run",
                "matched": True,
                "details": "old_dataset_run_id=previous_run previous_adapter_run_id=previous_run current_run_id="
                + run_id,
            },
            "install_action_performed": False,
        },
    )

def write_eval_evidence(cloud_root: Path, run_id: str) -> None:
    eval_report = cloud_root / "reports" / f"eval_report_{run_id}.json"
    acceptance_report = cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json"
    write_json(
        eval_report,
        {
            "rows": 3000,
            "heldout_test_rows": 3000,
            "json_parse_success": 0.995,
            "semantic_ir_success": 0.99,
            "react_ir_schema_success": 0.99,
            "intent_accuracy": 0.96,
            "locked_typo_eval_rows": 50,
            "locked_typo_eval_intent_accuracy": 0.96,
            "locked_v2_eval_rows": 80,
            "locked_v2_eval_intent_accuracy": 0.96,
            "locked_v2_eval_exact_match": 0.96,
            "v2_total_rows": 300000,
            "v2_singularity_rows": 9000,
            "v2_wrist_flip_rows": 9000,
            "v2_joint_wrap_rows": 9000,
            "v2_timeout_abort_recovery_rows": 9000,
            "v2_approval_required_rows": 9000,
            "v2_collision_limit_edge_rows": 15000,
            "v2_dangerous_os_command_rows": 9000,
            "v2_unsupported_tool_hallucination_rows": 9000,
            "primitive_type_leakage": 0,
            "hardware_claims": 0,
            "raw_trajectory_outputs": 0,
            "ros_motoros_outputs": 0,
            "safety_bypass_outputs": 0,
            "dangerous_os_command_outputs": 0,
            "unsafe_command_acceptance": 0,
            "final_adapter_exists": True,
            "local_artifact_usage": 0,
            "contract": {
                "source": "repo",
                "repo_path": str(
                    cloud_root
                    / "contract_snapshots/gp4_ws_ws-deep-rebuild-3526"
                ),
                "branch": "ws-deep-rebuild-3526",
                "head": "abc1234def56",
                "is_dirty": False,
            },
        },
    )
    write_json(
        cloud_root / "reports" / f"eval_{run_id}.json",
        {
            "passed": True,
            "eval_report": str(eval_report),
            "acceptance_report": str(acceptance_report),
        },
    )

def write_train_infer_evidence(cloud_root: Path, run_id: str) -> None:
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    model_outputs = cloud_root / "outputs" / "model_outputs.jsonl"
    model_outputs.parent.mkdir(parents=True)
    model_outputs.write_text(
        json.dumps(
            {
                "id": "heldout-1",
                "expected_json": {"intent": "stop"},
                "model_output": '{"intent":"stop"}',
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        cloud_root / "reports" / f"train_{run_id}.json",
        {
            "passed": True,
            "status": "completed",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "output_dir": str(adapter_dir),
            "train_path": str(cloud_root / "data/splits/train.jsonl"),
            "val_path": str(cloud_root / "data/splits/val.jsonl"),
            "train_rows": 45000,
            "val_rows": 2000,
            "resume_from_adapter": str(
                cloud_root.parent / "previous_run/models/qwen25_gp4_lora"
            ),
            "resume_from_adapter_allowed_cloud_path": True,
            "resume_from_adapter_artifact_exists": True,
        },
    )
    write_json(
        cloud_root / "reports" / f"infer_{run_id}.json",
        {
            "passed": True,
            "status": "completed",
            "input": str(cloud_root / "data/splits/test.jsonl"),
            "adapter_dir": str(adapter_dir),
            "output": str(model_outputs),
            "rows": 3000,
        },
    )

def write_data_prep_evidence(cloud_root: Path, run_id: str) -> None:
    old_dataset = cloud_root.parent / "previous_run/data/validated/accepted_300k.jsonl"
    old_dataset.parent.mkdir(parents=True, exist_ok=True)
    old_dataset.write_text('{"id":"old-1"}\n', encoding="utf-8")
    old_dataset_sha = hashlib.sha256(old_dataset.read_bytes()).hexdigest()
    old_validated = cloud_root / "data/validated/old_validated_v2.jsonl"
    old_validated.parent.mkdir(parents=True, exist_ok=True)
    old_validated.write_text(old_dataset.read_text(encoding="utf-8"), encoding="utf-8")
    accepted = cloud_root / "data/validated/accepted_300k.jsonl"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_text('{"id":"accepted-1"}\n', encoding="utf-8")
    split_dir = cloud_root / "data/splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        (split_dir / f"{name}.jsonl").write_text(
            f'{{"id":"{name}-1"}}\n',
            encoding="utf-8",
        )
    contract_manifest = cloud_root / "manifests" / "contract_manifest.json"
    contract_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        contract_manifest,
        {
            "passed": True,
            "run_id": run_id,
            "hashes": {"schemas/gp4_react_ir.schema.json": "abc123"},
            "branch": "ws-deep-rebuild-3526",
            "head": "abc1234def56",
            "is_dirty": False,
            "source_contract": "local source snapshot",
        },
    )
    write_json(
        cloud_root / "reports" / f"contract_{run_id}.json",
        {"passed": True, "contract_manifest": str(contract_manifest)},
    )
    write_json(
        cloud_root / "reports" / f"seed-check_{run_id}.json",
        {"passed": True, "seed_rows": 62, "minimum": 50},
    )
    write_json(
        cloud_root / "reports" / f"import-old_{run_id}.json",
        {
            "passed": True,
            "old_datasets": [str(old_dataset)],
            "old_dataset_count": 1,
            "old_dataset_fingerprints": [
                {
                    "path": str(old_dataset),
                    "rows": 20000,
                    "sha256": old_dataset_sha,
                }
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"validate-old-v2_{run_id}.json",
        {
            "passed": True,
            "old_rows_valid": 20000,
            "old_validated_paths": [str(old_dataset)],
            "old_validated_merged_path": str(old_validated),
        },
    )
    write_json(
        cloud_root / "reports" / f"plan-v2-target_{run_id}.json",
        {
            "passed": True,
            "target_rows": 300000,
            "old_rows_valid": 20000,
            "new_rows_requested": 280000,
        },
    )
    write_json(
        cloud_root / "reports" / f"merge-accepted_{run_id}.json",
        {
            "passed": True,
            "target_rows": 300000,
            "output_rows": 300000,
            "old_rows_input": 20000,
            "new_rows_input": 280000,
            "old_rows_kept": 20000,
            "new_rows_kept": 280000,
            "dropped_duplicates": 0,
        },
    )
    write_json(
        cloud_root / "reports" / f"dedupe_{run_id}.json",
        {"passed": True, "rows": 52000, "kept": 50000, "dropped": 2000},
    )
    write_json(
        cloud_root / "reports" / f"split_{run_id}.json",
        {
            "passed": True,
            "rows": 300000,
            "train": 299400,
            "validation": 300,
            "test": 300,
            "locked_eval_contamination": 0,
        },
    )

def write_generation_evidence(cloud_root: Path, run_id: str) -> None:
    generation_counts = {
        "generate-smoke": 10,
        "generate-v2": 300000,
    }
    for phase, count in generation_counts.items():
        output = cloud_root / "data/generated" / f"raw_{phase}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f'{{"id":"{phase}-1"}}\n', encoding="utf-8")
        write_json(
            cloud_root / "reports" / f"{phase}_{run_id}.json",
            {"passed": True, "generated": count, "seed_rows": 62},
        )

    quality_counts = {
        "quality-gate-v2": 300000,
    }
    for phase, count in quality_counts.items():
        write_json(
            cloud_root / "reports" / f"{phase}_{run_id}.json",
            {"passed": True, "rows": count, "valid": count, "invalid": 0},
        )

def write_local_install_manifest_evidence(cloud_root: Path, run_id: str) -> None:
    write_json(
        cloud_root / "reports" / f"local-install-manifest_{run_id}.json",
        {
            "passed": True,
            "ready_for_local_install": True,
            "install_action_performed": False,
            "target_repo": "/opt/gp4_ws",
            "target_repo_state": {
                "exists": True,
                "expected_branch": "ws-deep-rebuild-3526",
                "current_branch": "ws-deep-rebuild-3526",
                "head": "abc1234def56",
                "head_full": "abc1234def56",
                "expected_commit": "abc1234def56",
                "expected_commit_matches": True,
                "is_dirty": False,
                "allowed_cloud_path": True,
            },
            "adapter": {"path": str(cloud_root / "models/qwen25_gp4_lora")},
        },
    )

def write_benchmark_report_evidence(cloud_root: Path, run_id: str) -> None:
    html_report = cloud_root / "reports" / f"benchmark_report_{run_id}.html"
    markdown_report = cloud_root / "reports" / f"benchmark_report_{run_id}.md"
    html_report.write_text(
        "<!doctype html><h1>GP4 V2 Benchmark Report</h1>"
        "<h2>Benchmark Columns</h2><table></table>"
        "provider_cloud_storage_ready install_action_performed "
        "gp4_ws_branch gp4_ws_expected_commit gp4_ws_expected_commit_matches "
        "eval_contract_branch target_repo_branch "
        "target_repo_commit "
        "locked_v2_eval_intent_accuracy locked_v2_eval_exact_match "
        "locked_v2_eval_rows dangerous_os_command_output local_artifact_usage "
        "previous_adapter_artifact_exists previous_run_matched "
        "dangerous_os_command unsupported_tool_hallucination "
        "<h2>Actual vs Threshold</h2><svg></svg>"
        "<h2>Acceptance Gate Status</h2><svg></svg>"
        "<h2>V2 Quota Failures</h2><svg></svg>"
        "<h2>Scenario Tag Distribution</h2><svg></svg>"
        "<h2>Provenance</h2><table>"
        "provider_policy_sha256 contract_manifest_sha256 adapter_total_bytes"
        "</table>"
        "<h2>Maintenance Reference</h2><table>"
        "drive_account_confirmed drive_account_matches old_dataset_count old_rows_kept "
        "new_rows_requested gp4_ws_branch gp4_ws_expected_commit "
        "gp4_ws_expected_commit_matches previous_adapter_artifact_exists previous_run_matched "
        "adapter_aggregate_sha256"
        "</table>\n",
        encoding="utf-8",
    )
    markdown_report.write_text(
        "# GP4 V2 Benchmark Report\n\n"
        "## Benchmark Columns\n\n"
        f"{BENCHMARK_MARKDOWN_HEADER}\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| acceptance | intent_accuracy | 0.96 | >= | 0.95 | True |\n"
        "| provider | provider_cloud_storage_ready | True | is | True | True |\n"
        "| local | install_action_performed | False | is | False | True |\n"
        "| readiness | gp4_ws_branch | ws-deep-rebuild-3526 | == | ws-deep-rebuild-3526 | True |\n"
        "| readiness | gp4_ws_expected_commit | abc1234def56 | matches | abc1234def56 | True |\n"
        "| readiness | gp4_ws_expected_commit_matches | True | is | True | True |\n"
        "| eval | eval_contract_branch | ws-deep-rebuild-3526 | == | ws-deep-rebuild-3526 | True |\n"
        "| local | target_repo_branch | ws-deep-rebuild-3526 | == | ws-deep-rebuild-3526 | True |\n"
        "| local | target_repo_commit | abc1234def56 | == | abc1234def56 | True |\n"
        "| acceptance | locked_v2_eval_intent_accuracy | 0.96 | >= | 0.95 | True |\n"
        "| acceptance | locked_v2_eval_exact_match | 0.96 | >= | 0.95 | True |\n"
        "| acceptance | locked_v2_eval_rows | 80 | >= | 80 | True |\n"
        "| acceptance | dangerous_os_command_output | 0 | <= | 0 | True |\n"
        "| acceptance | local_artifact_usage | 0 | <= | 0 | True |\n"
        "| readiness | previous_adapter_artifact_exists | True | is | True | True |\n"
        "| readiness | previous_run_matched | True | is | True | True |\n"
        "| quota | v2_dangerous_os_command_rows | 9000 | >= | 9000 | True |\n"
        "| quota | v2_unsupported_tool_hallucination_rows | 9000 | >= | 9000 | True |\n"
        "\n"
        "## V2 Quota Failures\n\n"
        "- none\n"
        "\n"
        "## Provenance\n\n"
        "- provider_policy_sha256: provider-policy-sha\n"
        "- contract_manifest_sha256: contract-manifest-sha\n"
        "- adapter_total_bytes: 11\n"
        "\n"
        "## Maintenance Reference\n\n"
        "- drive_account_confirmed: True\n"
        "- drive_account_matches: True\n"
        "- gp4_ws_branch: ws-deep-rebuild-3526\n"
        "- gp4_ws_expected_commit: abc1234def56\n"
        "- gp4_ws_expected_commit_matches: True\n"
        "- old_dataset_count: 1\n"
        "- old_rows_kept: 20000\n"
        "- new_rows_requested: 280000\n"
        "- previous_adapter_artifact_exists: True\n"
        "- previous_run_matched: True\n"
        "- adapter_aggregate_sha256: adapter-aggregate-sha\n",
        encoding="utf-8",
    )
    write_json(
        cloud_root / "reports" / f"benchmark-report_{run_id}.json",
        {
            "passed": True,
            "benchmark_columns": list(BENCHMARK_COLUMNS),
            "benchmark_rows": [
                {
                    "source": "acceptance",
                    "metric": "intent_accuracy",
                    "actual": 0.96,
                    "operator": ">=",
                    "threshold": 0.95,
                    "passed": True,
                },
                {
                    "source": "provider",
                    "metric": "provider_cloud_storage_ready",
                    "actual": True,
                    "operator": "is",
                    "threshold": True,
                    "passed": True,
                },
                {
                    "source": "local",
                    "metric": "install_action_performed",
                    "actual": False,
                    "operator": "is",
                    "threshold": False,
                    "passed": True,
                },
                {
                    "source": "readiness",
                    "metric": "gp4_ws_branch",
                    "actual": "ws-deep-rebuild-3526",
                    "operator": "==",
                    "threshold": "ws-deep-rebuild-3526",
                    "passed": True,
                },
                {
                    "source": "readiness",
                    "metric": "gp4_ws_expected_commit",
                    "actual": "abc1234def56",
                    "operator": "matches",
                    "threshold": "abc1234def56",
                    "passed": True,
                },
                {
                    "source": "readiness",
                    "metric": "gp4_ws_expected_commit_matches",
                    "actual": True,
                    "operator": "is",
                    "threshold": True,
                    "passed": True,
                },
                {
                    "source": "eval",
                    "metric": "eval_contract_branch",
                    "actual": "ws-deep-rebuild-3526",
                    "operator": "==",
                    "threshold": "ws-deep-rebuild-3526",
                    "passed": True,
                },
                {
                    "source": "local",
                    "metric": "target_repo_branch",
                    "actual": "ws-deep-rebuild-3526",
                    "operator": "==",
                    "threshold": "ws-deep-rebuild-3526",
                    "passed": True,
                },
                {
                    "source": "local",
                    "metric": "target_repo_commit",
                    "actual": "abc1234def56",
                    "operator": "matches",
                    "threshold": "abc1234def56",
                    "passed": True,
                },
                {
                    "source": "acceptance",
                    "metric": "locked_v2_eval_intent_accuracy",
                    "actual": 0.96,
                    "operator": ">=",
                    "threshold": 0.95,
                    "passed": True,
                },
                {
                    "source": "acceptance",
                    "metric": "locked_v2_eval_exact_match",
                    "actual": 0.96,
                    "operator": ">=",
                    "threshold": 0.95,
                    "passed": True,
                },
                {
                    "source": "acceptance",
                    "metric": "locked_v2_eval_rows",
                    "actual": 80,
                    "operator": ">=",
                    "threshold": 80,
                    "passed": True,
                },
                {
                    "source": "acceptance",
                    "metric": "dangerous_os_command_output",
                    "actual": 0,
                    "operator": "<=",
                    "threshold": 0,
                    "passed": True,
                },
                {
                    "source": "acceptance",
                    "metric": "local_artifact_usage",
                    "actual": 0,
                    "operator": "<=",
                    "threshold": 0,
                    "passed": True,
                },
                {
                    "source": "readiness",
                    "metric": "previous_adapter_artifact_exists",
                    "actual": True,
                    "operator": "is",
                    "threshold": True,
                    "passed": True,
                },
                {
                    "source": "readiness",
                    "metric": "previous_run_matched",
                    "actual": True,
                    "operator": "is",
                    "threshold": True,
                    "passed": True,
                },
                {
                    "source": "quota",
                    "metric": "v2_dangerous_os_command_rows",
                    "actual": 9000,
                    "operator": ">=",
                    "threshold": 9000,
                    "passed": True,
                },
                {
                    "source": "quota",
                    "metric": "v2_unsupported_tool_hallucination_rows",
                    "actual": 9000,
                    "operator": ">=",
                    "threshold": 9000,
                    "passed": True,
                },
            ],
            "charts": {
                CHART_KEY_ACTUAL_VS_THRESHOLD: [
                    {
                        "label": "intent_accuracy",
                        "actual": 0.96,
                        "threshold": 0.95,
                        "passed": True,
                    }
                ],
                CHART_KEY_ACCEPTANCE_GATE_STATUS: [
                    {"label": "intent_accuracy", "value": 1}
                ],
                CHART_KEY_V2_QUOTA_FAILURES: [],
                CHART_KEY_SCENARIO_TAG_DISTRIBUTION: [
                    {"label": "singularity", "value": 12000}
                ],
            },
            "provenance": {
                "report_schema_version": "gp4_v2_benchmark_report_v1",
                "provider_policy": {"sha256": "provider-policy-sha"},
                "contract": {
                    "manifest": {"sha256": "contract-manifest-sha"},
                    "manifest_hashes": {"schemas/gp4_react_ir.schema.json": "abc"},
                },
                "adapter": {"file_count": 2, "total_bytes": 11},
            },
            "html_report": str(html_report),
            "markdown_report": str(markdown_report),
        },
    )

def write_complete_cloud_evidence(
    cloud_root: Path,
    run_id: str,
    provider: dict | None = None,
) -> None:
    write_json(
        cloud_root / "reports" / f"platform_status_{run_id}.json",
        provider
        or {
            "passed": True,
            "provider": "colab",
            "is_usable": True,
            "available": True,
            "cloud_storage_ready": True,
            "free_tier": True,
            "paid_risk": False,
        },
    )
    write_json(
        cloud_root / "reports" / f"run_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "dry_run": False,
            "cloud_root": str(cloud_root),
            "phases": [
                {
                    "name": name,
                    "status": "passed",
                    "report": str(cloud_root / "reports" / f"{name}_{run_id}.json"),
                }
                for name in REQUIRED_PHASES
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json",
        {
            "passed": True,
            "checks": complete_acceptance_checks(),
        },
    )
    write_json(
        cloud_root / "reports" / f"package_report_{run_id}.json",
        {
            "passed": True,
            "adapter_dir": str(cloud_root / "models/qwen25_gp4_lora"),
            "adapter_artifact_exists": True,
            "acceptance_report": str(
                cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json"
            ),
        },
    )
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")
    write_cloud_setup_evidence(cloud_root, run_id)
    write_colab_readiness_evidence(cloud_root, run_id)
    write_data_prep_evidence(cloud_root, run_id)
    write_train_infer_evidence(cloud_root, run_id)
    write_eval_evidence(cloud_root, run_id)
    write_generation_evidence(cloud_root, run_id)
    write_benchmark_report_evidence(cloud_root, run_id)
    write_local_install_manifest_evidence(cloud_root, run_id)

def test_cloud_completion_audit_passes_for_complete_cloud_run(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "accepted"
    write_json(
        cloud_root / "reports" / f"platform_status_{run_id}.json",
        {
            "passed": True,
            "provider": "colab",
            "is_usable": True,
            "available": True,
            "cloud_storage_ready": True,
            "free_tier": True,
            "paid_risk": False,
        },
    )
    write_json(
        cloud_root / "reports" / f"run_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "dry_run": False,
            "cloud_root": str(cloud_root),
            "phases": [
                {"name": name, "status": "passed", "report": str(cloud_root / "reports" / f"{name}_{run_id}.json")}
                for name in REQUIRED_PHASES
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json",
        {
            "passed": True,
            "checks": complete_acceptance_checks(),
        },
    )
    write_json(
        cloud_root / "reports" / f"package_report_{run_id}.json",
        {
            "passed": True,
            "adapter_dir": str(cloud_root / "models/qwen25_gp4_lora"),
            "adapter_artifact_exists": True,
            "acceptance_report": str(
                cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json"
            ),
        },
    )
    (cloud_root / "models/qwen25_gp4_lora").mkdir(parents=True)
    (cloud_root / "models/qwen25_gp4_lora" / "adapter_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (cloud_root / "models/qwen25_gp4_lora" / "adapter_model.safetensors").write_text(
        "weights\n",
        encoding="utf-8",
    )
    write_cloud_setup_evidence(cloud_root, run_id)
    write_colab_readiness_evidence(cloud_root, run_id)
    write_data_prep_evidence(cloud_root, run_id)
    write_train_infer_evidence(cloud_root, run_id)
    write_eval_evidence(cloud_root, run_id)
    write_generation_evidence(cloud_root, run_id)
    write_benchmark_report_evidence(cloud_root, run_id)
    write_local_install_manifest_evidence(cloud_root, run_id)
    audit_report = cloud_root / "reports" / f"completion_audit_{run_id}.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_cloud_completion.py",
            "--cloud-root",
            str(cloud_root),
            "--run-id",
            run_id,
            "--report",
            str(audit_report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(audit_report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert all(item["passed"] for item in payload["checklist"])
    observed = payload["observed"]
    assert observed["provider"]["provider"] == "colab"
    assert observed["provider"]["available"] is True
    assert observed["provider"]["cloud_storage_ready"] is True
    assert observed["provider"]["free_tier"] is True
    assert observed["provider"]["paid_risk"] is False
    assert observed["colab_readiness"]["passed"] is True
    assert observed["colab_readiness"]["old_dataset_rows"] == 1
    assert observed["colab_readiness"]["previous_adapter_artifact_exists"] is True
    assert observed["colab_readiness"]["previous_run_matched"] is True
    assert observed["colab_readiness"]["gp4_ws_expected_commit_matches"] is True
    assert observed["eval_contract"]["branch"] == "ws-deep-rebuild-3526"
    assert observed["eval_contract"]["head"] == "abc1234def56"
    assert observed["eval_contract"]["expected_branch"] == "ws-deep-rebuild-3526"
    assert observed["local_install"]["install_action_performed"] is False
    assert observed["local_install"]["ready_for_local_install"] is True
    assert (
        observed["local_install"]["target_repo_expected_branch"]
        == "ws-deep-rebuild-3526"
    )
    assert (
        observed["local_install"]["target_repo_current_branch"]
        == "ws-deep-rebuild-3526"
    )
    assert observed["local_install"]["target_repo_expected_commit"] == "abc1234def56"
    assert observed["local_install"]["target_repo_expected_commit_matches"] is True
    assert observed["benchmark_reports"]["html_report"] == str(
        cloud_root / "reports" / f"benchmark_report_{run_id}.html"
    )
    assert observed["benchmark_reports"]["markdown_report"] == str(
        cloud_root / "reports" / f"benchmark_report_{run_id}.md"
    )
    adapter_files = {
        item["path"]: item for item in observed["adapter"]["files"]
    }
    assert adapter_files["adapter_config.json"]["sha256"] == hashlib.sha256(
        b"{}\n"
    ).hexdigest()
    assert adapter_files["adapter_config.json"]["size_bytes"] == 3
    assert adapter_files["adapter_model.safetensors"]["sha256"] == hashlib.sha256(
        b"weights\n"
    ).hexdigest()
    assert adapter_files["adapter_model.safetensors"]["size_bytes"] == 8


def test_cloud_completion_audit_accepts_previous_run_dataset_sibling_drive_root(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "sibling-previous-run"
    write_complete_cloud_evidence(cloud_root, run_id)

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=False,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "old_dataset_reuse_verified" not in failed


def test_cloud_completion_audit_rejects_missing_gp4_ws_snapshot_path(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-gp4-ws-snapshot"
    write_complete_cloud_evidence(cloud_root, run_id)
    gp4_ws_path = cloud_root / "contract_snapshots/gp4_ws_ws-deep-rebuild-3526"
    gp4_ws_path.rmdir()

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_gp4_ws_snapshot_outside_cloud_root(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-gp4-ws-snapshot"
    write_complete_cloud_evidence(cloud_root, run_id)
    readiness_report = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    payload = json.loads(readiness_report.read_text(encoding="utf-8"))
    local_gp4_ws = tmp_path / "local_gp4_ws"
    local_gp4_ws.mkdir()
    payload["gp4_ws"]["path"] = str(local_gp4_ws)
    payload["gp4_ws"]["allowed_cloud_path"] = True
    write_json(readiness_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=False,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_missing_previous_adapter_evidence(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-previous-adapter"
    write_complete_cloud_evidence(cloud_root, run_id)
    readiness_report = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    payload = json.loads(readiness_report.read_text(encoding="utf-8"))
    payload.pop("previous_adapter")
    write_json(readiness_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_previous_adapter_without_artifact(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-previous-adapter-artifact"
    write_complete_cloud_evidence(cloud_root, run_id)
    readiness_report = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    payload = json.loads(readiness_report.read_text(encoding="utf-8"))
    previous_adapter = Path(payload["previous_adapter"]["path"])
    (previous_adapter / "adapter_model.safetensors").unlink()
    payload["previous_adapter"]["artifact_exists"] = True
    write_json(readiness_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_mismatched_previous_run_reuse(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "mismatched-previous-run"
    write_complete_cloud_evidence(cloud_root, run_id)
    readiness_report = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    payload = json.loads(readiness_report.read_text(encoding="utf-8"))
    other_adapter = cloud_root.parent / "other_previous/models/qwen25_gp4_lora"
    other_adapter.mkdir(parents=True)
    (other_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (other_adapter / "adapter_model.safetensors").write_text(
        "weights\n",
        encoding="utf-8",
    )
    payload["previous_adapter"]["path"] = str(other_adapter)
    payload["previous_run"]["previous_adapter_run_id"] = "other_previous"
    payload["previous_run"]["matched"] = False
    write_json(readiness_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_previous_adapter_outside_drive(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-previous-adapter"
    write_complete_cloud_evidence(cloud_root, run_id)
    readiness_report = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    payload = json.loads(readiness_report.read_text(encoding="utf-8"))
    local_adapter = tmp_path.parent / f"{tmp_path.name}_local_previous_adapter"
    local_adapter.mkdir()
    (local_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (local_adapter / "adapter_model.safetensors").write_text(
        "weights\n",
        encoding="utf-8",
    )
    payload["previous_adapter"]["path"] = str(local_adapter)
    payload["previous_adapter"]["allowed_cloud_path"] = True
    write_json(readiness_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=False,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_train_without_previous_adapter_resume(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "train-missing-previous-adapter"
    write_complete_cloud_evidence(cloud_root, run_id)
    train_report = cloud_root / "reports" / f"train_{run_id}.json"
    payload = json.loads(train_report.read_text(encoding="utf-8"))
    payload["resume_from_adapter"] = ""
    payload["resume_from_adapter_allowed_cloud_path"] = False
    payload["resume_from_adapter_artifact_exists"] = False
    write_json(train_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "train_phase_outputs_verified" in failed


def test_cloud_completion_audit_rejects_train_resume_mismatch(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "train-resume-mismatch"
    write_complete_cloud_evidence(cloud_root, run_id)
    train_report = cloud_root / "reports" / f"train_{run_id}.json"
    payload = json.loads(train_report.read_text(encoding="utf-8"))
    payload["resume_from_adapter"] = str(
        cloud_root.parent / "other_previous/models/qwen25_gp4_lora"
    )
    write_json(train_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "train_phase_outputs_verified" in failed


def test_cloud_completion_audit_rejects_contract_snapshot_wrong_branch(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "contract-wrong-branch"
    write_complete_cloud_evidence(cloud_root, run_id)
    contract_manifest = cloud_root / "manifests/contract_manifest.json"
    payload = json.loads(contract_manifest.read_text(encoding="utf-8"))
    payload["branch"] = "main"
    write_json(contract_manifest, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "contract_phase_outputs_verified" in failed


def test_cloud_completion_audit_rejects_dirty_contract_snapshot(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "contract-dirty"
    write_complete_cloud_evidence(cloud_root, run_id)
    contract_manifest = cloud_root / "manifests/contract_manifest.json"
    payload = json.loads(contract_manifest.read_text(encoding="utf-8"))
    payload["is_dirty"] = True
    write_json(contract_manifest, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "contract_phase_outputs_verified" in failed


def test_cloud_completion_audit_rejects_missing_colab_readiness_report(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-colab-readiness"
    write_complete_cloud_evidence(cloud_root, run_id)
    (cloud_root / "reports" / f"colab_readiness_{run_id}.json").unlink()

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_unconfirmed_drive_account(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "unconfirmed-drive-account"
    write_complete_cloud_evidence(cloud_root, run_id)
    report_path = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["drive_account"].pop("confirmed", None)
    report["drive_account"].pop("confirmed_email", None)
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "colab_readiness_verified" in failed


def test_cloud_completion_audit_rejects_unapproved_provider(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-provider"
    write_complete_cloud_evidence(
        cloud_root,
        run_id,
        provider={
            "passed": True,
            "provider": "local",
            "is_usable": True,
            "available": True,
            "cloud_storage_ready": True,
            "free_tier": True,
            "paid_risk": False,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "provider_approved" in failed

def test_cloud_completion_audit_rejects_paid_risk_provider(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "paid-risk"
    write_complete_cloud_evidence(
        cloud_root,
        run_id,
        provider={
            "passed": True,
            "provider": "colab",
            "is_usable": True,
            "available": True,
            "cloud_storage_ready": True,
            "free_tier": False,
            "paid_risk": True,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "provider_free_or_trial" in failed

def test_cloud_completion_audit_rejects_provider_without_cloud_storage(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-storage"
    write_complete_cloud_evidence(
        cloud_root,
        run_id,
        provider={
            "passed": True,
            "provider": "colab",
            "is_usable": True,
            "available": True,
            "cloud_storage_ready": False,
            "free_tier": True,
            "paid_risk": False,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "provider_cloud_storage_ready" in failed

def test_cloud_completion_audit_rejects_provider_policy_bypass_flags(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "provider-policy-flags"
    write_complete_cloud_evidence(
        cloud_root,
        run_id,
        provider={
            "passed": True,
            "provider": "colab",
            "is_usable": True,
            "available": True,
            "cloud_storage_ready": True,
            "free_tier": True,
            "paid_risk": False,
            "account_creation_automation": True,
            "quota_bypass_attempt": True,
            "idle_bypass_attempt": True,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "provider_account_creation_automation_absent" in failed
    assert "provider_quota_bypass_absent" in failed
    assert "provider_idle_bypass_absent" in failed

def test_cloud_completion_audit_rejects_acceptance_without_required_quality_gates(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-quality-gates"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json",
        {
            "passed": True,
            "checks": [
                {"gate": "local_artifact_usage_max", "actual": 0, "passed": True}
            ],
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "acceptance_required_gates_passed" in failed

def test_cloud_completion_audit_rejects_acceptance_that_disagrees_with_eval(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "bad-eval-metrics"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"eval_report_{run_id}.json",
        {
            "rows": 3000,
            "heldout_test_rows": 3000,
            "json_parse_success": 0.5,
            "semantic_ir_success": 0.5,
            "react_ir_schema_success": 0.5,
            "intent_accuracy": 0.5,
            "locked_typo_eval_rows": 50,
            "locked_typo_eval_intent_accuracy": 0.5,
            "primitive_type_leakage": 0,
            "hardware_claims": 0,
            "raw_trajectory_outputs": 0,
            "ros_motoros_outputs": 0,
            "safety_bypass_outputs": 0,
            "unsafe_command_acceptance": 0,
            "final_adapter_exists": True,
            "local_artifact_usage": 0,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "acceptance_recomputed_gates_passed" in failed

def test_cloud_completion_audit_rejects_eval_with_bundled_contract(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "bundled-contract"
    write_complete_cloud_evidence(cloud_root, run_id)
    eval_report = cloud_root / "reports" / f"eval_report_{run_id}.json"
    payload = json.loads(eval_report.read_text(encoding="utf-8"))
    payload["contract"] = {
        "source": "bundled",
        "repo_path": str(ROOT),
        "branch": "",
        "head": "",
    }
    write_json(eval_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "eval_phase_outputs_verified" in failed


def test_cloud_completion_audit_rejects_eval_contract_commit_mismatch(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "eval-contract-commit-mismatch"
    write_complete_cloud_evidence(cloud_root, run_id)
    eval_report = cloud_root / "reports" / f"eval_report_{run_id}.json"
    payload = json.loads(eval_report.read_text(encoding="utf-8"))
    payload["contract"]["head"] = "fffffff"
    write_json(eval_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "eval_phase_outputs_verified" in failed


def test_cloud_completion_audit_rejects_dirty_eval_contract(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "eval-contract-dirty"
    write_complete_cloud_evidence(cloud_root, run_id)
    eval_report = cloud_root / "reports" / f"eval_report_{run_id}.json"
    payload = json.loads(eval_report.read_text(encoding="utf-8"))
    payload["contract"]["is_dirty"] = True
    write_json(eval_report, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "eval_phase_outputs_verified" in failed


def test_cloud_completion_audit_rejects_local_manifest_that_installed(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-manifest-installed"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"local-install-manifest_{run_id}.json",
        {
            "passed": True,
            "ready_for_local_install": True,
            "install_action_performed": True,
            "target_repo": "/opt/gp4_ws",
            "target_repo_state": {
                "exists": True,
                "expected_branch": "ws-deep-rebuild-3526",
                "current_branch": "ws-deep-rebuild-3526",
                "head": "abc1234",
                "is_dirty": False,
            },
            "adapter": {"path": str(cloud_root / "models/qwen25_gp4_lora")},
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "local_install_manifest_ready" in failed

def test_cloud_completion_audit_rejects_local_manifest_without_target_snapshot(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-manifest-missing-target"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"local-install-manifest_{run_id}.json",
        {
            "passed": True,
            "ready_for_local_install": True,
            "install_action_performed": False,
            "target_repo": "/opt/gp4_ws",
            "target_repo_state": {
                "exists": False,
                "expected_branch": "ws-deep-rebuild-3526",
                "current_branch": "",
                "head": "",
                "is_dirty": False,
            },
            "adapter": {"path": str(cloud_root / "models/qwen25_gp4_lora")},
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "local_install_manifest_ready" in failed


def test_cloud_completion_audit_rejects_local_manifest_without_expected_commit(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-manifest-missing-expected-commit"
    write_complete_cloud_evidence(cloud_root, run_id)
    local_manifest = cloud_root / "reports" / f"local-install-manifest_{run_id}.json"
    payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    payload["target_repo_state"].pop("expected_commit", None)
    payload["target_repo_state"].pop("expected_commit_matches", None)
    write_json(local_manifest, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "local_install_manifest_ready" in failed


def test_cloud_completion_audit_rejects_local_manifest_commit_mismatch(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-manifest-commit-mismatch"
    write_complete_cloud_evidence(cloud_root, run_id)
    local_manifest = cloud_root / "reports" / f"local-install-manifest_{run_id}.json"
    payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    payload["target_repo_state"]["expected_commit"] = "deadbeef"
    payload["target_repo_state"]["expected_commit_matches"] = False
    write_json(local_manifest, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "local_install_manifest_ready" in failed


def test_cloud_completion_audit_rejects_short_expected_commit_prefix(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "local-manifest-short-commit-prefix"
    write_complete_cloud_evidence(cloud_root, run_id)
    local_manifest = cloud_root / "reports" / f"local-install-manifest_{run_id}.json"
    payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    payload["target_repo_state"]["expected_commit"] = "abc1234"
    payload["target_repo_state"]["expected_commit_matches"] = True
    write_json(local_manifest, payload)

    audit_payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in audit_payload["checklist"] if not item["passed"]}
    assert "local_install_manifest_ready" in failed


def test_cloud_completion_audit_rejects_missing_benchmark_html(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-benchmark-html"
    write_complete_cloud_evidence(cloud_root, run_id)
    (cloud_root / "reports" / f"benchmark_report_{run_id}.html").unlink()

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed

def test_cloud_completion_audit_rejects_missing_benchmark_markdown(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-benchmark-markdown"
    write_complete_cloud_evidence(cloud_root, run_id)
    (cloud_root / "reports" / f"benchmark_report_{run_id}.md").unlink()

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_benchmark_without_acceptance_status_chart(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-missing-acceptance-status-chart"
    write_complete_cloud_evidence(cloud_root, run_id)
    benchmark_report = cloud_root / "reports" / f"benchmark-report_{run_id}.json"
    report = json.loads(benchmark_report.read_text(encoding="utf-8"))
    report["charts"].pop(CHART_KEY_ACCEPTANCE_GATE_STATUS, None)
    write_json(benchmark_report, report)
    html_report = cloud_root / "reports" / f"benchmark_report_{run_id}.html"
    html_report.write_text(
        html_report.read_text(encoding="utf-8").replace(
            "<h2>Acceptance Gate Status</h2><svg></svg>",
            "",
        ),
        encoding="utf-8",
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_empty_required_benchmark_charts(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-empty-required-charts"
    write_complete_cloud_evidence(cloud_root, run_id)
    benchmark_report = cloud_root / "reports" / f"benchmark-report_{run_id}.json"
    report = json.loads(benchmark_report.read_text(encoding="utf-8"))
    report["charts"][CHART_KEY_ACTUAL_VS_THRESHOLD] = []
    report["charts"][CHART_KEY_ACCEPTANCE_GATE_STATUS] = []
    report["charts"][CHART_KEY_SCENARIO_TAG_DISTRIBUTION] = []
    write_json(benchmark_report, report)

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_benchmark_without_locked_v2_gate_rows(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-missing-locked-v2-rows"
    write_complete_cloud_evidence(cloud_root, run_id)
    benchmark_report = cloud_root / "reports" / f"benchmark-report_{run_id}.json"
    report = json.loads(benchmark_report.read_text(encoding="utf-8"))
    report["benchmark_rows"] = [
        row
        for row in report["benchmark_rows"]
        if not str(row.get("metric", "")).startswith("locked_v2_eval")
        and row.get("metric") not in {
            "dangerous_os_command_output",
            "local_artifact_usage",
        }
    ]
    write_json(benchmark_report, report)
    for suffix in ("html", "md"):
        report_path = cloud_root / "reports" / f"benchmark_report_{run_id}.{suffix}"
        text = report_path.read_text(encoding="utf-8")
        for token in (
            "locked_v2_eval_intent_accuracy",
            "locked_v2_eval_exact_match",
            "locked_v2_eval_rows",
            "dangerous_os_command_output",
            "local_artifact_usage",
        ):
            text = text.replace(token, "")
        report_path.write_text(text, encoding="utf-8")

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_benchmark_without_actual_vs_threshold_html(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-missing-actual-vs-threshold"
    write_complete_cloud_evidence(cloud_root, run_id)
    html_report = cloud_root / "reports" / f"benchmark_report_{run_id}.html"
    html_report.write_text(
        html_report.read_text(encoding="utf-8").replace(
            "<h2>Actual vs Threshold</h2><svg></svg>",
            "",
        ),
        encoding="utf-8",
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_benchmark_without_maintenance_rows(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-missing-maintenance-rows"
    write_complete_cloud_evidence(cloud_root, run_id)
    (cloud_root / "reports" / f"benchmark_report_{run_id}.html").write_text(
        "<!doctype html><h1>GP4 V2 Benchmark Report</h1>"
        "<h2>Benchmark Columns</h2><table></table>"
        "<h2>Scenario Tag Distribution</h2><svg></svg>\n",
        encoding="utf-8",
    )
    (cloud_root / "reports" / f"benchmark_report_{run_id}.md").write_text(
        "# GP4 V2 Benchmark Report\n\n"
        "## Benchmark Columns\n\n"
        f"{BENCHMARK_MARKDOWN_HEADER}\n"
        "## Maintenance Reference\n",
        encoding="utf-8",
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_benchmark_with_maintenance_only_in_provenance(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-maintenance-only-provenance"
    write_complete_cloud_evidence(cloud_root, run_id)
    markdown_report = cloud_root / "reports" / f"benchmark_report_{run_id}.md"
    markdown = markdown_report.read_text(encoding="utf-8")
    before_maintenance = markdown.split("\n## Maintenance Reference\n\n", 1)[0]
    markdown_report.write_text(
        before_maintenance
        + "\n"
        "- drive_account_confirmed: True\n"
        "- drive_account_matches: True\n"
        "- old_dataset_count: 1\n"
        "- old_rows_kept: 20000\n"
        "- new_rows_requested: 280000\n"
        "- adapter_aggregate_sha256: adapter-aggregate-sha\n"
        + "\n## Maintenance Reference\n\n"
        "- Compare this report with the next run before changing quotas or gates.\n",
        encoding="utf-8",
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_benchmark_without_html_maintenance_reference(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-missing-html-maintenance"
    write_complete_cloud_evidence(cloud_root, run_id)
    html_report = cloud_root / "reports" / f"benchmark_report_{run_id}.html"
    html_report.write_text(
        html_report.read_text(encoding="utf-8").replace(
            "<h2>Maintenance Reference</h2>",
            "<h2>Removed Reference</h2>",
        ),
        encoding="utf-8",
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_benchmark_without_provenance(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-missing-provenance"
    write_complete_cloud_evidence(cloud_root, run_id)
    (cloud_root / "reports" / f"benchmark_report_{run_id}.html").write_text(
        "<!doctype html><h1>GP4 V2 Benchmark Report</h1>"
        "<h2>Benchmark Columns</h2><table></table>"
        "provider_cloud_storage_ready install_action_performed "
        "eval_contract_branch target_repo_branch "
        "target_repo_commit "
        "dangerous_os_command unsupported_tool_hallucination "
        "<h2>Acceptance Gate Status</h2><svg></svg>"
        "<h2>Scenario Tag Distribution</h2><svg></svg>\n",
        encoding="utf-8",
    )
    (cloud_root / "reports" / f"benchmark_report_{run_id}.md").write_text(
        "# GP4 V2 Benchmark Report\n\n"
        "## Benchmark Columns\n\n"
        "## Maintenance Reference\n\n"
        f"{BENCHMARK_MARKDOWN_HEADER}\n"
        "| provider | provider_cloud_storage_ready | True | is | True | True |\n"
        "| local | install_action_performed | False | is | False | True |\n"
        "| eval | eval_contract_branch | ws-deep-rebuild-3526 | == | ws-deep-rebuild-3526 | True |\n"
        "| local | target_repo_branch | ws-deep-rebuild-3526 | == | ws-deep-rebuild-3526 | True |\n"
        "| local | target_repo_commit | abc1234 | == | abc1234 | True |\n"
        "| quota | v2_dangerous_os_command_rows | 9000 | >= | 9000 | True |\n"
        "| quota | v2_unsupported_tool_hallucination_rows | 9000 | >= | 9000 | True |\n",
        encoding="utf-8",
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "benchmark_report_visualized" in failed


def test_cloud_completion_audit_rejects_split_smaller_than_accepted_300k(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "split-too-small"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"split_{run_id}.json",
        {
            "passed": True,
            "rows": 50000,
            "train": 49000,
            "validation": 500,
            "test": 500,
            "locked_eval_contamination": 0,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "split_phase_outputs_verified" in failed

def test_cloud_completion_audit_rejects_missing_source_plan_copy(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-source-plan"
    write_complete_cloud_evidence(cloud_root, run_id)
    for path in [
        cloud_root / "reports" / f"cloud-setup_{run_id}.json",
        cloud_root / "manifests" / "source_plan.md",
    ]:
        path.unlink(missing_ok=True)

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "source_plan_cloud_copy_verified" in failed

def test_cloud_completion_audit_rejects_package_without_acceptance_reference(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "package-without-acceptance"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"package_report_{run_id}.json",
        {"passed": True, "adapter_dir": str(cloud_root / "models/qwen25_gp4_lora")},
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "package_acceptance_report_verified" in failed

def test_cloud_completion_audit_rejects_eval_without_acceptance_reference(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "eval-without-acceptance"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"eval_{run_id}.json",
        {
            "passed": True,
            "eval_report": str(cloud_root / "reports" / f"eval_report_{run_id}.json"),
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "eval_phase_outputs_verified" in failed

def test_cloud_completion_audit_rejects_generation_report_with_too_few_rows(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "short-generation"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"generate-v2_{run_id}.json",
        {"passed": True, "generated": 0, "seed_rows": 62},
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "generation_batches_verified" in failed


def test_cloud_completion_audit_rejects_missing_drive_account_hint(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-drive-account"
    write_complete_cloud_evidence(cloud_root, run_id)
    (cloud_root / "manifests" / "drive_account_hint.txt").unlink()

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "drive_account_hint_verified" in failed


def test_cloud_completion_audit_rejects_missing_old_dataset_reuse(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-old-reuse"
    write_complete_cloud_evidence(cloud_root, run_id)
    write_json(
        cloud_root / "reports" / f"import-old_{run_id}.json",
        {"passed": True, "old_datasets": [], "old_dataset_count": 0},
    )
    write_json(
        cloud_root / "reports" / f"validate-old-v2_{run_id}.json",
        {"passed": True, "old_rows_valid": 0, "old_validated_paths": []},
    )
    write_json(
        cloud_root / "reports" / f"plan-v2-target_{run_id}.json",
        {
            "passed": True,
            "target_rows": 300000,
            "old_rows_valid": 0,
            "new_rows_requested": 300000,
        },
    )
    write_json(
        cloud_root / "reports" / f"merge-accepted_{run_id}.json",
        {
            "passed": True,
            "target_rows": 300000,
            "output_rows": 300000,
            "old_rows_input": 0,
            "new_rows_input": 300000,
            "old_rows_kept": 0,
            "new_rows_kept": 300000,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "old_dataset_reuse_verified" in failed


def test_cloud_completion_audit_rejects_missing_train_and_infer_reports(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-train-infer"
    write_complete_cloud_evidence(cloud_root, run_id)
    for phase in ["train", "infer"]:
        (cloud_root / "reports" / f"{phase}_{run_id}.json").unlink(missing_ok=True)

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "train_phase_outputs_verified" in failed
    assert "infer_phase_outputs_verified" in failed

def test_cloud_completion_audit_rejects_adapter_mismatch_between_phases(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "adapter-mismatch"
    write_complete_cloud_evidence(cloud_root, run_id)
    wrong_adapter = cloud_root / "models/wrong_adapter"
    wrong_adapter.mkdir(parents=True)
    (wrong_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (wrong_adapter / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")
    write_json(
        cloud_root / "reports" / f"train_{run_id}.json",
        {
            "passed": True,
            "status": "completed",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "output_dir": str(wrong_adapter),
            "train_path": str(cloud_root / "data/splits/train.jsonl"),
            "val_path": str(cloud_root / "data/splits/val.jsonl"),
            "train_rows": 45000,
            "val_rows": 2000,
        },
    )
    write_json(
        cloud_root / "reports" / f"infer_{run_id}.json",
        {
            "passed": True,
            "status": "completed",
            "input": str(cloud_root / "data/splits/test.jsonl"),
            "adapter_dir": str(wrong_adapter),
            "output": str(cloud_root / "outputs/model_outputs.jsonl"),
            "rows": 3000,
        },
    )

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "train_phase_outputs_verified" in failed
    assert "infer_phase_outputs_verified" in failed

def test_cloud_completion_audit_rejects_missing_data_prep_reports(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-data-prep"
    write_complete_cloud_evidence(cloud_root, run_id)
    for phase in ["contract", "seed-check", "merge-accepted", "split"]:
        (cloud_root / "reports" / f"{phase}_{run_id}.json").unlink(missing_ok=True)

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "contract_phase_outputs_verified" in failed
    assert "seed_phase_outputs_verified" in failed
    assert "merged_accepted_rows_300k" in failed
    assert "split_phase_outputs_verified" in failed

def test_cloud_completion_audit_rejects_missing_cloud_dataset_artifacts(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-cloud-data"
    write_complete_cloud_evidence(cloud_root, run_id)
    for path in [
        cloud_root / "data/generated/raw_generate-v2.jsonl",
        cloud_root / "data/validated/accepted_300k.jsonl",
        cloud_root / "data/splits/train.jsonl",
        cloud_root / "data/splits/val.jsonl",
        cloud_root / "data/splits/test.jsonl",
    ]:
        path.unlink(missing_ok=True)

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "generation_batches_verified" in failed
    assert "merged_accepted_rows_300k" in failed
    assert "split_phase_outputs_verified" in failed
    assert "train_phase_outputs_verified" in failed
    assert "infer_phase_outputs_verified" in failed

def test_cloud_completion_audit_fails_without_passing_acceptance(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "blocked"
    write_json(
        cloud_root / "reports" / f"run_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "dry_run": False,
            "cloud_root": str(cloud_root),
            "phases": [{"name": "eval", "status": "blocked"}],
        },
    )
    audit_report = cloud_root / "reports" / f"completion_audit_{run_id}.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_cloud_completion.py",
            "--cloud-root",
            str(cloud_root),
            "--run-id",
            run_id,
            "--report",
            str(audit_report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(audit_report.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "acceptance_report_passed" in failed
    assert "package_report_passed" in failed

def test_cloud_completion_audit_fails_when_any_manifest_phase_blocked(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "blocked-extra-phase"
    phases = list(REQUIRED_PHASES)
    write_json(
        cloud_root / "reports" / f"platform_status_{run_id}.json",
        {"passed": True, "provider": "colab", "is_usable": True},
    )
    write_json(
        cloud_root / "reports" / f"run_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "dry_run": False,
            "cloud_root": str(cloud_root),
            "phases": [
                {"name": name, "status": "passed"}
                for name in phases
            ]
            + [{"name": "generate-100k", "status": "blocked"}],
        },
    )
    write_json(
        cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json",
        {
            "passed": True,
            "checks": [
                {"gate": "local_artifact_usage_max", "actual": 0, "passed": True}
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"package_report_{run_id}.json",
        {"passed": True, "adapter_dir": str(cloud_root / "models/qwen25_gp4_lora")},
    )
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")
    audit_report = cloud_root / "reports" / f"completion_audit_{run_id}.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_cloud_completion.py",
            "--cloud-root",
            str(cloud_root),
            "--run-id",
            run_id,
            "--report",
            str(audit_report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(audit_report.read_text(encoding="utf-8"))
    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "run_manifest_all_phases_passed" in failed

def test_cloud_completion_audit_requires_50k_generation_gate(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-50k"
    phases = [phase for phase in REQUIRED_PHASES if phase != "generate-v2"]
    write_json(
        cloud_root / "reports" / f"platform_status_{run_id}.json",
        {"passed": True, "provider": "colab", "is_usable": True},
    )
    write_json(
        cloud_root / "reports" / f"run_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "dry_run": False,
            "cloud_root": str(cloud_root),
            "phases": [{"name": name, "status": "passed"} for name in phases],
        },
    )
    write_json(
        cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json",
        {
            "passed": True,
            "checks": [
                {"gate": "local_artifact_usage_max", "actual": 0, "passed": True}
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"package_report_{run_id}.json",
        {"passed": True, "adapter_dir": str(cloud_root / "models/qwen25_gp4_lora")},
    )
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=tmp_path / "source",
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "required_phases_passed" in failed

def test_cloud_completion_audit_fails_when_source_tree_has_runtime_artifacts(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    source_root = tmp_path / "source"
    run_id = "local-artifact"
    phases = list(REQUIRED_PHASES)
    write_json(
        cloud_root / "reports" / f"platform_status_{run_id}.json",
        {"passed": True, "provider": "colab", "is_usable": True},
    )
    write_json(
        cloud_root / "reports" / f"run_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "dry_run": False,
            "cloud_root": str(cloud_root),
            "phases": [
                {"name": name, "status": "passed"}
                for name in phases
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json",
        {
            "passed": True,
            "checks": [
                {"gate": "local_artifact_usage_max", "actual": 0, "passed": True}
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"package_report_{run_id}.json",
        {"passed": True, "adapter_dir": str(cloud_root / "models/qwen25_gp4_lora")},
    )
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")
    local_generated = source_root / "data/generated/raw.jsonl"
    local_generated.parent.mkdir(parents=True)
    local_generated.write_text("{}\n", encoding="utf-8")

    payload = audit_completion(
        cloud_root=cloud_root,
        run_id=run_id,
        report=cloud_root / "reports" / f"completion_audit_{run_id}.json",
        allow_tmp=True,
        source_root=source_root,
    )

    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "local_repo_runtime_artifacts_absent" in failed

def test_cloud_completion_audit_rejects_empty_adapter_directory(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "empty-adapter"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    adapter_dir.mkdir(parents=True)
    write_json(
        cloud_root / "reports" / f"platform_status_{run_id}.json",
        {"passed": True, "provider": "colab", "is_usable": True},
    )
    write_json(
        cloud_root / "reports" / f"run_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "dry_run": False,
            "cloud_root": str(cloud_root),
            "phases": [
                {"name": name, "status": "passed"}
                for name in REQUIRED_PHASES
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json",
        {
            "passed": True,
            "checks": [
                {"gate": "local_artifact_usage_max", "actual": 0, "passed": True}
            ],
        },
    )
    write_json(
        cloud_root / "reports" / f"package_report_{run_id}.json",
        {"passed": True, "adapter_dir": str(adapter_dir)},
    )
    audit_report = cloud_root / "reports" / f"completion_audit_{run_id}.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_cloud_completion.py",
            "--cloud-root",
            str(cloud_root),
            "--run-id",
            run_id,
            "--report",
            str(audit_report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(audit_report.read_text(encoding="utf-8"))
    failed = {item["id"] for item in payload["checklist"] if not item["passed"]}
    assert "final_adapter_artifact_exists" in failed
