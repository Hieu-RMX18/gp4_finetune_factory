#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from check_acceptance_gates import evaluate_gates
from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from cloud_runtime import CloudPathError, sha256_file, validate_cloud_run_paths
from factory_common import read_json, read_yaml, write_json
from package_adapter import adapter_artifact_exists

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_POLICY = ROOT / "configs/provider_policy.yaml"
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"
LOCAL_RUNTIME_ARTIFACT_DIRS = (
    Path("artifact_downloads"),
    Path("data/generated"),
    Path("data/validated"),
    Path("data/splits"),
    Path("models"),
    Path("outputs"),
    Path("reports"),
)

REQUIRED_PHASES = (
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
)

REQUIRED_ACCEPTANCE_GATES = (
    "heldout_rows_min",
    "json_parse_success_min",
    "heldout_intent_accuracy_min",
    "semantic_ir_schema_success_min",
    "react_ir_schema_success_min",
    "primitive_type_leakage_normal_max",
    "hardware_execution_claim_max",
    "raw_trajectory_output_max",
    "ros_motoros_call_output_max",
    "safety_bypass_output_max",
    "unsafe_command_acceptance_max",
    "local_artifact_usage_max",
    "locked_typo_eval_intent_accuracy_min",
    "locked_typo_eval_rows_min",
    "heldout_output_rows_equal_test_rows",
    "final_adapter_exists",
)

GENERATION_PHASE_MINIMUMS = {
    "generate-smoke": 10,
    "generate-1k": 1000,
    "generate-30k": 30000,
    "generate-50k": 50000,
}

QUALITY_GATE_MINIMUMS = {
    "quality-gate-1k": 1000,
    "quality-gate-30k": 30000,
    "quality-gate-50k": 50000,
}

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit cloud GP4 fine-tune completion evidence."
    )
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=False,
            inputs=[],
            outputs=[args.report],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"completion_audit_blocked reason={exc} report={args.report}")
        return 1

    payload = audit_completion(
        cloud_root=args.cloud_root,
        run_id=args.run_id,
        report=args.report,
        allow_tmp=args.allow_tmp,
    )
    write_json(args.report, payload)
    print(f"passed={payload['passed']} report={args.report}")
    for item in payload["checklist"]:
        if not item["passed"]:
            print(f"{item['id']}: {item['message']}")
    return 0 if payload["passed"] else 1

def audit_completion(
    *,
    cloud_root: Path,
    run_id: str,
    report: Path,
    allow_tmp: bool,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    policy = CloudStoragePolicy(cloud_roots=(cloud_root,), allow_tmp=allow_tmp)
    reports_dir = cloud_root / "reports"
    manifest_path = reports_dir / f"run_manifest_{run_id}.json"
    provider_path = reports_dir / f"platform_status_{run_id}.json"
    eval_report_path = reports_dir / f"eval_report_{run_id}.json"
    acceptance_path = reports_dir / f"acceptance_gate_report_{run_id}.json"
    package_path = reports_dir / f"package_report_{run_id}.json"

    manifest = _read_optional_json(manifest_path)
    provider = _read_optional_json(provider_path)
    eval_report = _read_optional_json(eval_report_path)
    acceptance = _read_optional_json(acceptance_path)
    package = _read_optional_json(package_path)
    provider_policy = _read_optional_yaml(PROVIDER_POLICY)
    dataset_spec = _read_optional_yaml(DATASET_SPEC)
    local_artifacts = _local_runtime_artifacts(source_root)
    adapter_dir = Path(
        str(package.get("adapter_dir") or cloud_root / "models/qwen25_gp4_lora")
    )

    checklist = [
        _check(
            "audit_report_cloud_path",
            is_allowed_cloud_path(report, policy),
            f"audit report path must be under CLOUD_ROOT: {report}",
            str(report),
        ),
        _check(
            "provider_status_passed",
            bool(provider.get("passed")) and bool(provider.get("is_usable", True)),
            f"provider status must pass: {provider_path}",
            str(provider_path),
        ),
        _check(
            "provider_cloud_storage_ready",
            provider.get("available") is True and provider.get("cloud_storage_ready") is True,
            "provider must be available with cloud storage mounted",
            str(provider_path),
        ),
        _check(
            "provider_approved",
            _provider_approved(provider, provider_policy),
            f"provider must be approved by policy: {provider.get('provider')}",
            str(provider_path),
        ),
        _check(
            "provider_free_or_trial",
            _provider_free_or_trial(provider, provider_policy),
            "provider must be free/trial with no paid-risk flag",
            str(provider_path),
        ),
        _check(
            "run_manifest_exists",
            bool(manifest),
            f"run manifest is missing or invalid: {manifest_path}",
            str(manifest_path),
        ),
        _check(
            "run_manifest_matches_run_id",
            manifest.get("run_id") == run_id,
            f"run manifest run_id must match {run_id}",
            str(manifest_path),
        ),
        _check(
            "run_manifest_not_dry_run",
            manifest.get("dry_run") is False,
            "completion requires a non-dry-run cloud manifest",
            str(manifest_path),
        ),
        _check(
            "run_manifest_cloud_root_matches",
            Path(str(manifest.get("cloud_root", ""))).resolve(strict=False)
            == cloud_root.resolve(strict=False),
            f"manifest cloud_root must match {cloud_root}",
            str(manifest_path),
        ),
        _check(
            "required_phases_passed",
            _required_phases_passed(manifest),
            "all required workflow phases must be present with status=passed",
            str(manifest_path),
        ),
        _check(
            "run_manifest_all_phases_passed",
            _all_manifest_phases_passed(manifest),
            "every phase in the run manifest must have status=passed",
            str(manifest_path),
        ),
        _check(
            "manifest_report_paths_cloud_only",
            _manifest_report_paths_cloud_only(manifest, policy),
            "all manifest report paths must be under CLOUD_ROOT",
            str(manifest_path),
        ),
        _check(
            "generation_batches_verified",
            _generation_batches_verified(manifest, policy, cloud_root),
            "generation and quality-gate reports must meet required row counts",
            str(manifest_path),
        ),
        _check(
            "source_plan_cloud_copy_verified",
            _source_plan_cloud_copy_verified(manifest, policy),
            "cloud-setup report must reference an existing source_plan.md under "
            "CLOUD_ROOT with a matching sha256",
            str(manifest_path),
        ),
        _check(
            "contract_phase_outputs_verified",
            _contract_phase_outputs_verified(manifest, policy),
            "contract phase report must pass and reference an existing cloud "
            "contract manifest with hashes",
            str(manifest_path),
        ),
        _check(
            "seed_phase_outputs_verified",
            _seed_phase_outputs_verified(manifest, policy, dataset_spec),
            "seed-check report must pass and meet the configured seed minimum",
            str(manifest_path),
        ),
        _check(
            "dedupe_phase_outputs_verified",
            _dedupe_phase_outputs_verified(manifest, policy, cloud_root),
            "dedupe report must pass with balanced row, kept, and dropped counts",
            str(manifest_path),
        ),
        _check(
            "split_phase_outputs_verified",
            _split_phase_outputs_verified(manifest, policy, cloud_root),
            "split report must pass with nonzero train, validation, and test "
            "rows and no locked eval contamination",
            str(manifest_path),
        ),
        _check(
            "train_phase_outputs_verified",
            _train_phase_outputs_verified(manifest, policy, adapter_dir),
            "train phase report must pass and reference cloud train, val, and "
            "adapter output paths",
            str(manifest_path),
        ),
        _check(
            "infer_phase_outputs_verified",
            _infer_phase_outputs_verified(manifest, policy, adapter_dir),
            "infer phase report must pass and reference cloud input, adapter, "
            "and output paths",
            str(manifest_path),
        ),
        _check(
            "eval_phase_outputs_verified",
            _eval_phase_outputs_verified(
                manifest,
                eval_report_path,
                acceptance_path,
                policy,
            ),
            "eval phase report must reference the expected cloud eval and "
            "acceptance reports",
            str(manifest_path),
        ),
        _check(
            "acceptance_report_passed",
            acceptance.get("passed") is True,
            f"acceptance report must pass: {acceptance_path}",
            str(acceptance_path),
        ),
        _check(
            "acceptance_required_gates_passed",
            _acceptance_required_gates_passed(acceptance),
            "acceptance report must contain every required strict quality gate "
            "with passed=true",
            str(acceptance_path),
        ),
        _check(
            "acceptance_recomputed_gates_passed",
            _acceptance_recomputed_gates_passed(
                acceptance,
                eval_report,
                dataset_spec,
            ),
            "acceptance gates recomputed from eval metrics must pass and match "
            "the acceptance report",
            str(acceptance_path),
        ),
        _check(
            "local_artifact_usage_zero",
            _local_artifact_usage_zero(acceptance),
            "acceptance local_artifact_usage_max gate must pass with actual=0",
            str(acceptance_path),
        ),
        _check(
            "local_repo_runtime_artifacts_absent",
            not local_artifacts,
            "local source tree contains runtime artifacts: "
            + ", ".join(local_artifacts[:10]),
            str(source_root),
        ),
        _check(
            "package_report_passed",
            package.get("passed") is True,
            f"package report must pass: {package_path}",
            str(package_path),
        ),
        _check(
            "package_acceptance_report_verified",
            _package_acceptance_report_verified(package, acceptance_path, policy),
            "package report must reference the expected cloud acceptance report",
            str(package_path),
        ),
        _check(
            "final_adapter_cloud_path",
            is_allowed_cloud_path(adapter_dir, policy),
            f"final adapter path must be under CLOUD_ROOT: {adapter_dir}",
            str(adapter_dir),
        ),
        _check(
            "final_adapter_exists",
            adapter_dir.exists(),
            f"final adapter directory must exist: {adapter_dir}",
            str(adapter_dir),
        ),
        _check(
            "final_adapter_artifact_exists",
            adapter_artifact_exists(adapter_dir),
            f"final adapter artifact files must exist: {adapter_dir}",
            str(adapter_dir),
        ),
    ]
    return {
        "passed": all(item["passed"] for item in checklist),
        "run_id": run_id,
        "cloud_root": str(cloud_root),
        "checklist": checklist,
    }

def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}

def _read_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_yaml(path)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}

def _provider_approved(
    provider: dict[str, Any],
    provider_policy: dict[str, Any],
) -> bool:
    approved = provider_policy.get("providers", {}).get("order", [])
    if not isinstance(approved, list):
        return False
    return str(provider.get("provider", "")) in {str(name) for name in approved}

def _provider_free_or_trial(
    provider: dict[str, Any],
    provider_policy: dict[str, Any],
) -> bool:
    policy = provider_policy.get("providers", {})
    if not isinstance(policy, dict):
        return False
    if policy.get("require_free_or_trial") is True and provider.get("free_tier") is not True:
        return False
    if policy.get("forbid_paid_fallback") is True and provider.get("paid_risk") is not False:
        return False
    return True

def _required_phases_passed(manifest: dict[str, Any]) -> bool:
    phases = {
        str(phase.get("name")): str(phase.get("status"))
        for phase in manifest.get("phases", [])
        if isinstance(phase, dict)
    }
    return all(phases.get(phase) == "passed" for phase in REQUIRED_PHASES)

def _all_manifest_phases_passed(manifest: dict[str, Any]) -> bool:
    phases = manifest.get("phases", [])
    if not isinstance(phases, list) or not phases:
        return False
    return all(
        isinstance(phase, dict) and phase.get("status") == "passed"
        for phase in phases
    )

def _manifest_report_paths_cloud_only(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
) -> bool:
    phases = manifest.get("phases", [])
    if not isinstance(phases, list):
        return False
    for phase in phases:
        if not isinstance(phase, dict):
            return False
        report = phase.get("report")
        if report and not is_allowed_cloud_path(Path(str(report)), policy):
            return False
    return True

def _generation_batches_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    cloud_root: Path,
) -> bool:
    for phase, minimum in GENERATION_PHASE_MINIMUMS.items():
        report = _phase_report(manifest, phase, policy)
        if report is None or report.get("passed") is not True:
            return False
        if _int_value(report.get("generated")) < minimum:
            return False
        if not _cloud_file_exists(
            cloud_root / "data/generated" / f"raw_{phase}.jsonl",
            policy,
        ):
            return False

    for phase, minimum in QUALITY_GATE_MINIMUMS.items():
        report = _phase_report(manifest, phase, policy)
        if report is None or report.get("passed") is not True:
            return False
        if _int_value(report.get("rows")) < minimum:
            return False
        if _int_value(report.get("valid")) < minimum:
            return False
        if _int_value(report.get("invalid")) != 0:
            return False
    return True

def _phase_report(
    manifest: dict[str, Any],
    phase_name: str,
    policy: CloudStoragePolicy,
) -> dict[str, Any] | None:
    report_path = _phase_report_path(manifest, phase_name)
    if report_path is None or not is_allowed_cloud_path(report_path, policy):
        return None
    return _read_optional_json(report_path)

def _source_plan_cloud_copy_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
) -> bool:
    setup_report_path = _phase_report_path(manifest, "cloud-setup")
    if setup_report_path is None or not is_allowed_cloud_path(setup_report_path, policy):
        return False
    setup_report = _read_optional_json(setup_report_path)
    source_plan = setup_report.get("source_plan")
    expected_sha = setup_report.get("source_plan_sha256")
    if not source_plan or not expected_sha:
        return False
    source_plan_path = Path(str(source_plan))
    if not is_allowed_cloud_path(source_plan_path, policy) or not source_plan_path.exists():
        return False
    return sha256_file(source_plan_path) == str(expected_sha)

def _contract_phase_outputs_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
) -> bool:
    report = _phase_report(manifest, "contract", policy)
    if report is None or report.get("passed") is not True:
        return False
    contract_manifest = report.get("contract_manifest")
    if not contract_manifest:
        return False
    contract_path = Path(str(contract_manifest))
    if not is_allowed_cloud_path(contract_path, policy) or not contract_path.exists():
        return False
    contract = _read_optional_json(contract_path)
    return contract.get("passed") is True and isinstance(contract.get("hashes"), dict)

def _seed_phase_outputs_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    dataset_spec: dict[str, Any],
) -> bool:
    report = _phase_report(manifest, "seed-check", policy)
    if report is None or report.get("passed") is not True:
        return False
    seed_gate = dataset_spec.get("seed_gate", {})
    if not isinstance(seed_gate, dict):
        return False
    required_minimum = _int_value(seed_gate.get("minimum_handwritten_seed_examples"))
    if required_minimum <= 0:
        return False
    return (
        _int_value(report.get("seed_rows")) >= required_minimum
        and _int_value(report.get("minimum")) == required_minimum
    )

def _dedupe_phase_outputs_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    cloud_root: Path,
) -> bool:
    report = _phase_report(manifest, "dedupe", policy)
    if report is None or report.get("passed") is not True:
        return False
    rows = _int_value(report.get("rows"))
    kept = _int_value(report.get("kept"))
    dropped = _int_value(report.get("dropped"))
    return (
        rows > 0
        and kept > 0
        and dropped >= 0
        and rows == kept + dropped
        and _cloud_file_exists(cloud_root / "data/validated/accepted.jsonl", policy)
    )

def _split_phase_outputs_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    cloud_root: Path,
) -> bool:
    report = _phase_report(manifest, "split", policy)
    if report is None or report.get("passed") is not True:
        return False
    rows = _int_value(report.get("rows"))
    train = _int_value(report.get("train"))
    validation = _int_value(report.get("validation"))
    test = _int_value(report.get("test"))
    return (
        rows > 0
        and train > 0
        and validation > 0
        and test > 0
        and rows == train + validation + test
        and _int_value(report.get("locked_eval_contamination")) == 0
        and _cloud_file_exists(cloud_root / "data/splits/train.jsonl", policy)
        and _cloud_file_exists(cloud_root / "data/splits/val.jsonl", policy)
        and _cloud_file_exists(cloud_root / "data/splits/test.jsonl", policy)
    )

def _train_phase_outputs_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    expected_adapter_dir: Path,
) -> bool:
    report = _phase_report(manifest, "train", policy)
    if report is None or report.get("passed") is not True:
        return False
    for key in ["train_path", "val_path", "output_dir"]:
        value = report.get(key)
        if not value or not is_allowed_cloud_path(Path(str(value)), policy):
            return False
    if not _cloud_file_exists(Path(str(report["train_path"])), policy):
        return False
    if not _cloud_file_exists(Path(str(report["val_path"])), policy):
        return False
    output_dir = Path(str(report["output_dir"]))
    if not _same_path(output_dir, expected_adapter_dir):
        return False
    return (
        _int_value(report.get("train_rows")) > 0
        and _int_value(report.get("val_rows")) > 0
    )

def _infer_phase_outputs_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    expected_adapter_dir: Path,
) -> bool:
    report = _phase_report(manifest, "infer", policy)
    if report is None or report.get("passed") is not True:
        return False
    for key in ["input", "adapter_dir", "output"]:
        value = report.get(key)
        if not value or not is_allowed_cloud_path(Path(str(value)), policy):
            return False
    input_path = Path(str(report["input"]))
    if not _cloud_file_exists(input_path, policy):
        return False
    adapter_dir = Path(str(report["adapter_dir"]))
    if not adapter_dir.exists():
        return False
    if not _same_path(adapter_dir, expected_adapter_dir):
        return False
    output_path = Path(str(report["output"]))
    return _int_value(report.get("rows")) > 0 and _cloud_file_exists(output_path, policy)

def _cloud_file_exists(path: Path, policy: CloudStoragePolicy) -> bool:
    try:
        return (
            is_allowed_cloud_path(path, policy)
            and path.is_file()
            and path.stat().st_size > 0
        )
    except OSError:
        return False

def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)

def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1

def _phase_report_path(manifest: dict[str, Any], phase_name: str) -> Path | None:
    phases = manifest.get("phases", [])
    if not isinstance(phases, list):
        return None
    for phase in phases:
        if not isinstance(phase, dict) or phase.get("name") != phase_name:
            continue
        report = phase.get("report")
        return Path(str(report)) if report else None
    return None

def _eval_phase_outputs_verified(
    manifest: dict[str, Any],
    eval_report_path: Path,
    acceptance_path: Path,
    policy: CloudStoragePolicy,
) -> bool:
    phase_report_path = _phase_report_path(manifest, "eval")
    if phase_report_path is None or not is_allowed_cloud_path(phase_report_path, policy):
        return False
    phase_report = _read_optional_json(phase_report_path)
    if phase_report.get("passed") is not True:
        return False
    phase_eval_report = phase_report.get("eval_report")
    phase_acceptance_report = phase_report.get("acceptance_report")
    if not phase_eval_report or not phase_acceptance_report:
        return False
    phase_eval_path = Path(str(phase_eval_report))
    phase_acceptance_path = Path(str(phase_acceptance_report))
    if not is_allowed_cloud_path(phase_eval_path, policy):
        return False
    if not is_allowed_cloud_path(phase_acceptance_path, policy):
        return False
    return (
        phase_eval_path.resolve(strict=False) == eval_report_path.resolve(strict=False)
        and phase_acceptance_path.resolve(strict=False)
        == acceptance_path.resolve(strict=False)
        and eval_report_path.exists()
    )

def _package_acceptance_report_verified(
    package: dict[str, Any],
    acceptance_path: Path,
    policy: CloudStoragePolicy,
) -> bool:
    package_acceptance = package.get("acceptance_report")
    if not package_acceptance:
        return False
    package_acceptance_path = Path(str(package_acceptance))
    if not is_allowed_cloud_path(package_acceptance_path, policy):
        return False
    return package_acceptance_path.resolve(strict=False) == acceptance_path.resolve(strict=False)

def _local_artifact_usage_zero(acceptance: dict[str, Any]) -> bool:
    checks = acceptance.get("checks", [])
    if not isinstance(checks, list):
        return False
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("gate") != "local_artifact_usage_max":
            continue
        return check.get("passed") is True and check.get("actual") == 0
    return False

def _acceptance_required_gates_passed(acceptance: dict[str, Any]) -> bool:
    checks = acceptance.get("checks", [])
    if not isinstance(checks, list):
        return False
    gates = {
        str(check.get("gate")): check
        for check in checks
        if isinstance(check, dict)
    }
    return all(
        isinstance(gates.get(gate), dict) and gates[gate].get("passed") is True
        for gate in REQUIRED_ACCEPTANCE_GATES
    )

def _acceptance_recomputed_gates_passed(
    acceptance: dict[str, Any],
    eval_report: dict[str, Any],
    dataset_spec: dict[str, Any],
) -> bool:
    if not eval_report or not dataset_spec:
        return False
    heldout_rows = _int_value(eval_report.get("heldout_test_rows"))
    recomputed = evaluate_gates(
        spec=dataset_spec,
        eval_report=eval_report,
        min_rows=heldout_rows if heldout_rows > 0 else 1,
    )
    if recomputed.get("passed") is not True:
        return False
    reported_checks = _checks_by_gate(acceptance)
    recomputed_checks = _checks_by_gate(recomputed)
    for gate in REQUIRED_ACCEPTANCE_GATES:
        reported = reported_checks.get(gate)
        expected = recomputed_checks.get(gate)
        if not isinstance(reported, dict) or not isinstance(expected, dict):
            return False
        if reported.get("passed") is not True or expected.get("passed") is not True:
            return False
        if reported.get("actual") != expected.get("actual"):
            return False
    return True

def _checks_by_gate(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return {}
    return {
        str(check.get("gate")): check
        for check in checks
        if isinstance(check, dict)
    }

def _local_runtime_artifacts(source_root: Path) -> list[str]:
    findings: list[str] = []
    for relative_dir in LOCAL_RUNTIME_ARTIFACT_DIRS:
        directory = source_root / relative_dir
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                findings.append(path.relative_to(source_root).as_posix())
    return sorted(findings)

def _check(check_id: str, passed: bool, message: str, evidence: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "message": "" if passed else message,
        "evidence": evidence,
    }

if __name__ == "__main__":
    raise SystemExit(main())
