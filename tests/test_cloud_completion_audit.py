import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_cloud_completion import audit_completion

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
        {"gate": "unsafe_command_acceptance_max", "actual": 0, "passed": True},
        {"gate": "local_artifact_usage_max", "actual": 0, "passed": True},
        {
            "gate": "locked_typo_eval_intent_accuracy_min",
            "actual": 0.96,
            "passed": True,
        },
        {"gate": "locked_typo_eval_rows_min", "actual": 50, "passed": True},
        {"gate": "heldout_output_rows_equal_test_rows", "actual": 3000, "passed": True},
        {"gate": "final_adapter_exists", "actual": True, "passed": True},
    ]

def write_cloud_setup_evidence(cloud_root: Path, run_id: str) -> None:
    source_plan = cloud_root / "manifests" / "source_plan.md"
    source_plan.parent.mkdir(parents=True)
    source_plan.write_text("# GP4 cloud fine-tune source plan\n", encoding="utf-8")
    source_plan_sha = hashlib.sha256(source_plan.read_bytes()).hexdigest()
    write_json(
        cloud_root / "reports" / f"cloud-setup_{run_id}.json",
        {
            "passed": True,
            "source_plan": str(source_plan),
            "source_plan_sha256": source_plan_sha,
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
    accepted = cloud_root / "data/validated/accepted.jsonl"
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
        cloud_root / "reports" / f"dedupe_{run_id}.json",
        {"passed": True, "rows": 52000, "kept": 50000, "dropped": 2000},
    )
    write_json(
        cloud_root / "reports" / f"split_{run_id}.json",
        {
            "passed": True,
            "rows": 50000,
            "train": 44000,
            "validation": 3000,
            "test": 3000,
            "locked_eval_contamination": 0,
        },
    )

def write_generation_evidence(cloud_root: Path, run_id: str) -> None:
    generation_counts = {
        "generate-smoke": 10,
        "generate-1k": 1000,
        "generate-30k": 30000,
        "generate-50k": 50000,
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
        "quality-gate-1k": 1000,
        "quality-gate-30k": 30000,
        "quality-gate-50k": 50000,
    }
    for phase, count in quality_counts.items():
        write_json(
            cloud_root / "reports" / f"{phase}_{run_id}.json",
            {"passed": True, "rows": count, "valid": count, "invalid": 0},
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
                for name in [
                    "cloud-setup",
                    "provider-probe",
                    "contract",
                    "seed-check",
                    "generate-smoke",
                    "generate-1k",
                    "quality-gate-1k",
                    "generate-30k",
                    "quality-gate-30k",
                    "generate-50k",
                    "quality-gate-50k",
                    "dedupe",
                    "split",
                    "train",
                    "infer",
                    "eval",
                    "package",
                ]
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
    write_data_prep_evidence(cloud_root, run_id)
    write_train_infer_evidence(cloud_root, run_id)
    write_eval_evidence(cloud_root, run_id)
    write_generation_evidence(cloud_root, run_id)

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
                for name in [
                    "cloud-setup",
                    "provider-probe",
                    "contract",
                    "seed-check",
                    "generate-smoke",
                    "generate-1k",
                    "quality-gate-1k",
                    "generate-30k",
                    "quality-gate-30k",
                    "generate-50k",
                    "quality-gate-50k",
                    "dedupe",
                    "split",
                    "train",
                    "infer",
                    "eval",
                    "package",
                ]
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
    write_data_prep_evidence(cloud_root, run_id)
    write_train_infer_evidence(cloud_root, run_id)
    write_eval_evidence(cloud_root, run_id)
    write_generation_evidence(cloud_root, run_id)
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
        cloud_root / "reports" / f"generate-50k_{run_id}.json",
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
    for phase in ["contract", "seed-check", "dedupe", "split"]:
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
    assert "dedupe_phase_outputs_verified" in failed
    assert "split_phase_outputs_verified" in failed

def test_cloud_completion_audit_rejects_missing_cloud_dataset_artifacts(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "missing-cloud-data"
    write_complete_cloud_evidence(cloud_root, run_id)
    for path in [
        cloud_root / "data/generated/raw_generate-50k.jsonl",
        cloud_root / "data/validated/accepted.jsonl",
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
    assert "dedupe_phase_outputs_verified" in failed
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
    phases = [
        "cloud-setup",
        "provider-probe",
        "contract",
        "seed-check",
        "generate-smoke",
        "generate-1k",
        "quality-gate-1k",
        "generate-30k",
        "quality-gate-30k",
        "generate-50k",
        "quality-gate-50k",
        "dedupe",
        "split",
        "train",
        "infer",
        "eval",
        "package",
    ]
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
    phases = [
        "cloud-setup",
        "provider-probe",
        "contract",
        "seed-check",
        "generate-smoke",
        "generate-1k",
        "quality-gate-1k",
        "generate-30k",
        "quality-gate-30k",
        "dedupe",
        "split",
        "train",
        "infer",
        "eval",
        "package",
    ]
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
    phases = [
        "cloud-setup",
        "provider-probe",
        "contract",
        "seed-check",
        "generate-smoke",
        "generate-1k",
        "quality-gate-1k",
        "generate-30k",
        "quality-gate-30k",
        "dedupe",
        "split",
        "train",
        "infer",
        "eval",
        "package",
    ]
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
                for name in [
                    "cloud-setup",
                    "provider-probe",
                    "contract",
                    "seed-check",
                    "generate-smoke",
                    "generate-1k",
                    "quality-gate-1k",
                    "generate-30k",
                    "quality-gate-30k",
                    "generate-50k",
                    "quality-gate-50k",
                    "dedupe",
                    "split",
                    "train",
                    "infer",
                    "eval",
                    "package",
                ]
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
