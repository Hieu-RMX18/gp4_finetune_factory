#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmark_report_contract import (
    BENCHMARK_COLUMNS,
    CHART_KEY_ACCEPTANCE_GATE_STATUS,
    CHART_KEY_ACTUAL_VS_THRESHOLD,
    CHART_KEY_SCENARIO_TAG_DISTRIBUTION,
    REQUIRED_BENCHMARK_ROW_TOKENS,
    REQUIRED_CHART_KEYS,
    REQUIRED_HTML_TOKENS,
    REQUIRED_MAINTENANCE_REFERENCE_TOKENS,
    REQUIRED_MARKDOWN_TOKENS,
    REQUIRED_PROVENANCE_TOKENS,
)
from check_acceptance_gates import evaluate_gates
from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from cloud_runtime import CloudPathError, sha256_file, validate_cloud_run_paths
from factory_common import read_json, read_yaml, write_json
from package_adapter import adapter_artifact_exists

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_POLICY = ROOT / "configs/provider_policy.yaml"
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"
EXPECTED_DRIVE_ACCOUNT_EMAIL = "johnwickiller4444@gmail.com"
LOCAL_RUNTIME_ARTIFACT_DIRS = (
    Path("artifact_downloads"),
    Path("data/generated"),
    Path("data/validated"),
    Path("data/splits"),
    Path("models"),
    Path("outputs"),
    Path("reports"),
)
MIN_EXPECTED_COMMIT_LENGTH = 12

REQUIRED_PHASES = (
    "cloud-setup",
    "provider-probe",
    "contract",
    "seed-check",
    "import-old",
    "validate-old-v2",
    "plan-v2-target",
    "generate-smoke",
    "generate-v2",
    "validate-new-v2",
    "merge-accepted",
    "quality-gate-v2",
    "split",
    "train",
    "infer",
    "eval",
    "local-install-manifest",
    "benchmark-report",
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
    "dangerous_os_command_output_max",
    "unsafe_command_acceptance_max",
    "local_artifact_usage_max",
    "locked_typo_eval_intent_accuracy_min",
    "locked_typo_eval_rows_min",
    "locked_v2_eval_intent_accuracy_min",
    "locked_v2_eval_exact_match_min",
    "locked_v2_eval_rows_min",
    "v2_total_rows_min",
    "v2_singularity_rows_min",
    "v2_wrist_flip_rows_min",
    "v2_joint_wrap_rows_min",
    "v2_timeout_abort_recovery_rows_min",
    "v2_approval_required_rows_min",
    "v2_collision_limit_edge_rows_min",
    "v2_dangerous_os_command_rows_min",
    "v2_unsupported_tool_hallucination_rows_min",
    "heldout_output_rows_equal_test_rows",
    "final_adapter_exists",
)

GENERATION_PHASE_MINIMUMS = {
    "generate-smoke": 10,
    "generate-v2": 1,
}

QUALITY_GATE_MINIMUMS = {
    "quality-gate-v2": 300000,
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
            "provider_account_creation_automation_absent",
            _provider_policy_flag_absent(
                provider,
                provider_policy,
                "forbid_account_creation_automation",
                "account_creation_automation",
            ),
            "provider policy forbids account creation automation",
            str(provider_path),
        ),
        _check(
            "provider_quota_bypass_absent",
            _provider_policy_flag_absent(
                provider,
                provider_policy,
                "forbid_quota_bypass",
                "quota_bypass_attempt",
            ),
            "provider policy forbids quota bypass attempts",
            str(provider_path),
        ),
        _check(
            "provider_idle_bypass_absent",
            _provider_policy_flag_absent(
                provider,
                provider_policy,
                "forbid_idle_bypass",
                "idle_bypass_attempt",
            ),
            "provider policy forbids idle bypass attempts",
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
            "drive_account_hint_verified",
            _drive_account_hint_verified(cloud_root, policy),
            "cloud run must record the expected Google Drive account hint",
            str(cloud_root / "manifests/drive_account_hint.txt"),
        ),
        _check(
            "colab_readiness_verified",
            _colab_readiness_verified(cloud_root, run_id, policy),
            "Colab preflight must prove confirmed Drive account, old dataset, and gp4_ws pin readiness",
            str(cloud_root / "reports" / f"colab_readiness_{run_id}.json"),
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
            "merged_accepted_rows_300k",
            _merge_accepted_rows_verified(manifest, policy, cloud_root),
            "merge-accepted report must pass with output_rows >= 300000",
            str(manifest_path),
        ),
        _check(
            "old_dataset_reuse_verified",
            _old_dataset_reuse_verified(manifest, policy, cloud_root),
            "v2 run must record and keep rows from a previous accepted dataset",
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
            _train_phase_outputs_verified(
                manifest,
                policy,
                adapter_dir,
                cloud_root,
                run_id,
            ),
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
            "benchmark_report_visualized",
            _benchmark_report_visualized(manifest, policy),
            "benchmark-report phase must pass and reference an existing HTML "
            "report with benchmark table and charts",
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
            "local_install_manifest_ready",
            _local_install_manifest_ready(manifest, policy),
            "local-install-manifest phase must pass and record "
            "install_action_performed=false",
            str(manifest_path),
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
    observed = _observed_summary(
        cloud_root=cloud_root,
        manifest=manifest,
        provider_path=provider_path,
        provider=provider,
        eval_report_path=eval_report_path,
        eval_report=eval_report,
        package_path=package_path,
        package=package,
        adapter_dir=adapter_dir,
        dataset_spec=dataset_spec,
    )
    return {
        "passed": all(item["passed"] for item in checklist),
        "run_id": run_id,
        "cloud_root": str(cloud_root),
        "observed": observed,
        "checklist": checklist,
    }

def _observed_summary(
    *,
    cloud_root: Path,
    manifest: dict[str, Any],
    provider_path: Path,
    provider: dict[str, Any],
    eval_report_path: Path,
    eval_report: dict[str, Any],
    package_path: Path,
    package: dict[str, Any],
    adapter_dir: Path,
    dataset_spec: dict[str, Any],
) -> dict[str, Any]:
    expected_branch = _expected_source_branch(dataset_spec)
    contract = eval_report.get("contract", {})
    if not isinstance(contract, dict):
        contract = {}
    local_install_path = _phase_report_path(manifest, "local-install-manifest")
    local_install = (
        _read_optional_json(local_install_path)
        if local_install_path is not None
        else {}
    )
    target_state = local_install.get("target_repo_state", {})
    if not isinstance(target_state, dict):
        target_state = {}
    adapter = local_install.get("adapter", {})
    if not isinstance(adapter, dict):
        adapter = {}
    benchmark_path = _phase_report_path(manifest, "benchmark-report")
    benchmark = (
        _read_optional_json(benchmark_path)
        if benchmark_path is not None
        else {}
    )
    import_old = _phase_report(manifest, "import-old", CloudStoragePolicy((cloud_root,), allow_tmp=True)) or {}
    validate_old = _phase_report(manifest, "validate-old-v2", CloudStoragePolicy((cloud_root,), allow_tmp=True)) or {}
    plan_v2 = _phase_report(manifest, "plan-v2-target", CloudStoragePolicy((cloud_root,), allow_tmp=True)) or {}
    merge_accepted = _phase_report(manifest, "merge-accepted", CloudStoragePolicy((cloud_root,), allow_tmp=True)) or {}
    return {
        "drive_account": {
            "hint_path": str(cloud_root / "manifests/drive_account_hint.txt"),
            "expected": EXPECTED_DRIVE_ACCOUNT_EMAIL,
            "verified": _drive_account_hint_verified(
                cloud_root,
                CloudStoragePolicy((cloud_root,), allow_tmp=True),
            ),
        },
        "colab_readiness": _observed_colab_readiness(cloud_root, manifest),
        "old_dataset_reuse": {
            "old_dataset_count": import_old.get("old_dataset_count"),
            "old_rows_valid": validate_old.get("old_rows_valid"),
            "new_rows_requested": plan_v2.get("new_rows_requested"),
            "old_rows_kept": merge_accepted.get("old_rows_kept"),
            "old_dataset_fingerprints": import_old.get("old_dataset_fingerprints", []),
        },
        "provider": {
            "report_path": str(provider_path),
            "provider": provider.get("provider"),
            "available": provider.get("available"),
            "cloud_storage_ready": provider.get("cloud_storage_ready"),
            "is_usable": provider.get("is_usable"),
            "free_tier": provider.get("free_tier"),
            "paid_risk": provider.get("paid_risk"),
        },
        "eval_contract": {
            "report_path": str(eval_report_path),
            "source": contract.get("source"),
            "repo_path": contract.get("repo_path"),
            "branch": contract.get("branch"),
            "head": contract.get("head"),
            "expected_branch": expected_branch,
        },
        "local_install": {
            "report_path": str(local_install_path) if local_install_path else "",
            "install_action_performed": local_install.get("install_action_performed"),
            "ready_for_local_install": local_install.get("ready_for_local_install"),
            "target_repo": local_install.get("target_repo"),
            "target_repo_exists": target_state.get("exists"),
            "target_repo_expected_branch": target_state.get("expected_branch"),
            "target_repo_current_branch": target_state.get("current_branch"),
            "target_repo_head": target_state.get("head"),
            "target_repo_expected_commit": target_state.get("expected_commit"),
            "target_repo_expected_commit_matches": target_state.get("expected_commit_matches"),
            "target_repo_is_dirty": target_state.get("is_dirty"),
            "adapter_path": adapter.get("path"),
        },
        "benchmark_reports": {
            "phase_report_path": str(benchmark_path) if benchmark_path else "",
            "html_report": benchmark.get("html_report"),
            "markdown_report": benchmark.get("markdown_report"),
        },
        "adapter": {
            "package_report_path": str(package_path),
            "adapter_dir": str(adapter_dir),
            "artifact_exists": package.get("adapter_artifact_exists"),
            "files": _adapter_file_summary(adapter_dir, cloud_root),
        },
    }

def _expected_source_branch(dataset_spec: dict[str, Any]) -> str:
    project = dataset_spec.get("project", {})
    if not isinstance(project, dict):
        return ""
    return str(project.get("source_branch", ""))

def _adapter_file_summary(adapter_dir: Path, cloud_root: Path) -> list[dict[str, Any]]:
    if not adapter_dir.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            relative_path = path.relative_to(adapter_dir).as_posix()
            size_bytes = path.stat().st_size
        except OSError:
            continue
        try:
            cloud_relative_path = path.relative_to(cloud_root).as_posix()
        except ValueError:
            cloud_relative_path = ""
        summaries.append(
            {
                "path": relative_path,
                "cloud_relative_path": cloud_relative_path,
                "size_bytes": size_bytes,
                "sha256": sha256_file(path),
            }
        )
    return summaries

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

def _provider_policy_flag_absent(
    provider: dict[str, Any],
    provider_policy: dict[str, Any],
    policy_key: str,
    provider_key: str,
) -> bool:
    policy = provider_policy.get("providers", {})
    if not isinstance(policy, dict):
        return False
    if policy.get(policy_key) is not True:
        return True
    return provider.get(provider_key) is not True

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


def _drive_account_hint_verified(
    cloud_root: Path,
    policy: CloudStoragePolicy,
) -> bool:
    hint_path = cloud_root / "manifests/drive_account_hint.txt"
    if not _cloud_file_exists(hint_path, policy):
        return False
    try:
        hint = hint_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return hint == EXPECTED_DRIVE_ACCOUNT_EMAIL


def _colab_readiness_verified(
    cloud_root: Path,
    run_id: str,
    policy: CloudStoragePolicy,
) -> bool:
    report_path = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    if not _cloud_file_exists(report_path, policy):
        return False
    report = _read_optional_json(report_path)
    old_dataset_policy = CloudStoragePolicy(
        (cloud_root, cloud_root.parent),
        allow_tmp=policy.allow_tmp,
    )
    gp4_ws_policy = CloudStoragePolicy(
        (cloud_root / "contract_snapshots",),
        allow_tmp=False,
    )
    drive_account = report.get("drive_account", {})
    old_dataset = report.get("old_dataset", {})
    previous_adapter = report.get("previous_adapter", {})
    previous_run = report.get("previous_run", {})
    gp4_ws = report.get("gp4_ws", {})
    if not all(
        isinstance(item, dict)
        for item in (
            drive_account,
            old_dataset,
            previous_adapter,
            previous_run,
            gp4_ws,
        )
    ):
        return False
    old_dataset_path = Path(str(old_dataset.get("path") or ""))
    previous_adapter_path = Path(str(previous_adapter.get("path") or ""))
    old_dataset_run_id = str(previous_run.get("old_dataset_run_id") or "")
    previous_adapter_run_id = str(previous_run.get("previous_adapter_run_id") or "")
    gp4_ws_path = Path(str(gp4_ws.get("path") or ""))
    return (
        report.get("passed") is True
        and drive_account.get("matches_expected") is True
        and drive_account.get("email") == EXPECTED_DRIVE_ACCOUNT_EMAIL
        and drive_account.get("confirmed") is True
        and drive_account.get("confirmed_email") == EXPECTED_DRIVE_ACCOUNT_EMAIL
        and old_dataset.get("exists") is True
        and old_dataset.get("allowed_cloud_path") is True
        and _int_value(old_dataset.get("rows")) > 0
        and _cloud_file_exists(old_dataset_path, old_dataset_policy)
        and previous_adapter.get("exists") is True
        and previous_adapter.get("allowed_cloud_path") is True
        and previous_adapter.get("artifact_exists") is True
        and is_allowed_cloud_path(previous_adapter_path, old_dataset_policy)
        and previous_adapter_path.exists()
        and adapter_artifact_exists(previous_adapter_path)
        and previous_run.get("matched") is True
        and bool(old_dataset_run_id)
        and old_dataset_run_id == previous_adapter_run_id
        and old_dataset_run_id != run_id
        and gp4_ws.get("exists") is True
        and gp4_ws.get("allowed_cloud_path") is True
        and is_allowed_cloud_path(gp4_ws_path, gp4_ws_policy)
        and gp4_ws_path.exists()
        and gp4_ws.get("branch") == gp4_ws.get("expected_branch")
        and gp4_ws.get("expected_branch") == _expected_source_branch(_read_optional_yaml(DATASET_SPEC))
        and bool(gp4_ws.get("head"))
        and bool(gp4_ws.get("expected_commit"))
        and gp4_ws.get("expected_commit_matches") is True
        and _commit_matches(
            str(gp4_ws.get("head") or ""),
            str(gp4_ws.get("expected_commit") or ""),
        )
        and gp4_ws.get("is_dirty") is False
        and report.get("install_action_performed") is False
    )


def _observed_colab_readiness(
    cloud_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(manifest.get("run_id") or "")
    report = _read_optional_json(cloud_root / "reports" / f"colab_readiness_{run_id}.json")
    old_dataset = report.get("old_dataset", {})
    if not isinstance(old_dataset, dict):
        old_dataset = {}
    previous_adapter = report.get("previous_adapter", {})
    if not isinstance(previous_adapter, dict):
        previous_adapter = {}
    previous_run = report.get("previous_run", {})
    if not isinstance(previous_run, dict):
        previous_run = {}
    gp4_ws = report.get("gp4_ws", {})
    if not isinstance(gp4_ws, dict):
        gp4_ws = {}
    drive_account = report.get("drive_account", {})
    if not isinstance(drive_account, dict):
        drive_account = {}
    return {
        "passed": report.get("passed"),
        "drive_account_matches": drive_account.get("matches_expected"),
        "drive_account_confirmed": drive_account.get("confirmed"),
        "old_dataset_path": old_dataset.get("path"),
        "old_dataset_rows": old_dataset.get("rows"),
        "previous_adapter_path": previous_adapter.get("path"),
        "previous_adapter_artifact_exists": previous_adapter.get("artifact_exists"),
        "previous_run_old_dataset_run_id": previous_run.get("old_dataset_run_id"),
        "previous_run_adapter_run_id": previous_run.get("previous_adapter_run_id"),
        "previous_run_matched": previous_run.get("matched"),
        "gp4_ws_path": gp4_ws.get("path"),
        "gp4_ws_branch": gp4_ws.get("branch"),
        "gp4_ws_expected_commit": gp4_ws.get("expected_commit"),
        "gp4_ws_expected_commit_matches": gp4_ws.get("expected_commit_matches"),
    }


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
    expected_branch = _expected_source_branch(_read_optional_yaml(DATASET_SPEC))
    expected_commit = str(
        contract.get("expected_commit")
        or _local_install_expected_commit(manifest, policy)
        or ""
    )
    return (
        contract.get("passed") is True
        and isinstance(contract.get("hashes"), dict)
        and bool(expected_branch)
        and contract.get("branch") == expected_branch
        and bool(contract.get("head"))
        and _commit_matches(str(contract.get("head") or ""), expected_commit)
        and contract.get("is_dirty") is False
    )

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

def _merge_accepted_rows_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    cloud_root: Path,
) -> bool:
    report = _phase_report(manifest, "merge-accepted", policy)
    if report is None or report.get("passed") is not True:
        return False
    return (
        _int_value(report.get("output_rows")) >= 300000
        and _cloud_file_exists(
            cloud_root / "data/validated/accepted_300k.jsonl",
            policy,
        )
    )


def _old_dataset_reuse_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    cloud_root: Path,
) -> bool:
    import_report = _phase_report(manifest, "import-old", policy)
    validate_report = _phase_report(manifest, "validate-old-v2", policy)
    plan_report = _phase_report(manifest, "plan-v2-target", policy)
    merge_report = _phase_report(manifest, "merge-accepted", policy)
    if not all(
        isinstance(report, dict) and report.get("passed") is True
        for report in (import_report, validate_report, plan_report, merge_report)
    ):
        return False
    assert import_report is not None
    assert validate_report is not None
    assert plan_report is not None
    assert merge_report is not None
    fingerprints = import_report.get("old_dataset_fingerprints", [])
    if not isinstance(fingerprints, list) or not fingerprints:
        return False
    for fingerprint in fingerprints:
        if not isinstance(fingerprint, dict):
            return False
        path = Path(str(fingerprint.get("path") or ""))
        reuse_policy = CloudStoragePolicy(
            (cloud_root, cloud_root.parent),
            allow_tmp=policy.allow_tmp,
        )
        if not _cloud_file_exists(path, reuse_policy):
            return False
        if str(fingerprint.get("sha256") or "") != sha256_file(path):
            return False
        if _int_value(fingerprint.get("rows")) <= 0:
            return False
    old_validated_path = Path(str(validate_report.get("old_validated_merged_path") or ""))
    target_rows = _int_value(plan_report.get("target_rows"))
    return (
        _int_value(import_report.get("old_dataset_count")) > 0
        and _int_value(validate_report.get("old_rows_valid")) > 0
        and _cloud_file_exists(old_validated_path, policy)
        and target_rows >= 300000
        and _int_value(plan_report.get("new_rows_requested")) < target_rows
        and _int_value(merge_report.get("old_rows_input")) > 0
        and _int_value(merge_report.get("old_rows_kept")) > 0
        and _cloud_file_exists(cloud_root / "data/validated/accepted_300k.jsonl", policy)
    )


def _split_phase_outputs_verified(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
    cloud_root: Path,
) -> bool:
    report = _phase_report(manifest, "split", policy)
    if report is None or report.get("passed") is not True:
        return False
    merge_report = _phase_report(manifest, "merge-accepted", policy)
    if merge_report is None or merge_report.get("passed") is not True:
        return False
    expected_rows = _int_value(merge_report.get("output_rows"))
    rows = _int_value(report.get("rows"))
    train = _int_value(report.get("train"))
    validation = _int_value(report.get("validation"))
    test = _int_value(report.get("test"))
    return (
        rows >= 300000
        and rows == expected_rows
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
    cloud_root: Path,
    run_id: str,
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
    expected_resume = _colab_previous_adapter_path(
        cloud_root=cloud_root,
        run_id=run_id,
        policy=policy,
    )
    resume_path = Path(str(report.get("resume_from_adapter") or ""))
    reuse_policy = CloudStoragePolicy(
        (cloud_root, cloud_root.parent),
        allow_tmp=policy.allow_tmp,
    )
    if expected_resume is None or not _same_path(resume_path, expected_resume):
        return False
    if report.get("resume_from_adapter_allowed_cloud_path") is not True:
        return False
    if report.get("resume_from_adapter_artifact_exists") is not True:
        return False
    if not is_allowed_cloud_path(resume_path, reuse_policy):
        return False
    if not adapter_artifact_exists(resume_path):
        return False
    return (
        _int_value(report.get("train_rows")) > 0
        and _int_value(report.get("val_rows")) > 0
    )


def _colab_previous_adapter_path(
    *,
    cloud_root: Path,
    run_id: str,
    policy: CloudStoragePolicy,
) -> Path | None:
    report_path = cloud_root / "reports" / f"colab_readiness_{run_id}.json"
    if not _cloud_file_exists(report_path, policy):
        return None
    report = _read_optional_json(report_path)
    previous_adapter = report.get("previous_adapter", {})
    if not isinstance(previous_adapter, dict):
        return None
    if previous_adapter.get("artifact_exists") is not True:
        return None
    raw_path = str(previous_adapter.get("path") or "")
    return Path(raw_path) if raw_path else None


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
    eval_report = _read_optional_json(eval_report_path)
    expected_commit = _local_install_expected_commit(manifest, policy)
    return (
        phase_eval_path.resolve(strict=False) == eval_report_path.resolve(strict=False)
        and phase_acceptance_path.resolve(strict=False)
        == acceptance_path.resolve(strict=False)
        and eval_report_path.exists()
        and _eval_contract_verified(eval_report, expected_commit=expected_commit)
    )

def _local_install_expected_commit(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
) -> str:
    report = _phase_report(manifest, "local-install-manifest", policy)
    if report is None:
        return ""
    target_state = report.get("target_repo_state", {})
    if not isinstance(target_state, dict):
        return ""
    return str(target_state.get("expected_commit") or "")


def _eval_contract_verified(
    eval_report: dict[str, Any],
    *,
    expected_commit: str,
) -> bool:
    contract = eval_report.get("contract", {})
    if not isinstance(contract, dict):
        return False
    expected_branch = str(
        _read_optional_yaml(DATASET_SPEC).get("project", {}).get("source_branch", "")
    )
    return (
        contract.get("source") == "repo"
        and bool(contract.get("repo_path"))
        and bool(expected_branch)
        and contract.get("branch") == expected_branch
        and bool(contract.get("head"))
        and _commit_matches(str(contract.get("head") or ""), expected_commit)
        and contract.get("is_dirty") is False
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

def _benchmark_report_visualized(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
) -> bool:
    report = _phase_report(manifest, "benchmark-report", policy)
    if report is None or report.get("passed") is not True:
        return False
    if report.get("benchmark_columns") != list(BENCHMARK_COLUMNS):
        return False
    benchmark_rows = report.get("benchmark_rows")
    charts = report.get("charts", {})
    if not isinstance(benchmark_rows, list) or not benchmark_rows:
        return False
    benchmark_metrics = [
        str(row.get("metric") or "")
        for row in benchmark_rows
        if isinstance(row, dict)
    ]
    if not all(
        any(token in metric for metric in benchmark_metrics)
        for token in REQUIRED_BENCHMARK_ROW_TOKENS
    ):
        return False
    if not isinstance(charts, dict):
        return False
    if not all(key in charts for key in REQUIRED_CHART_KEYS):
        return False
    for required_non_empty_chart in (
        CHART_KEY_ACTUAL_VS_THRESHOLD,
        CHART_KEY_ACCEPTANCE_GATE_STATUS,
        CHART_KEY_SCENARIO_TAG_DISTRIBUTION,
    ):
        chart_rows = charts.get(required_non_empty_chart)
        if not isinstance(chart_rows, list) or not chart_rows:
            return False
    html_report = report.get("html_report")
    if not html_report:
        return False
    html_path = Path(str(html_report))
    if not _cloud_file_exists(html_path, policy):
        return False
    try:
        html = html_path.read_text(encoding="utf-8")
    except OSError:
        return False
    markdown_report = report.get("markdown_report")
    if not markdown_report:
        return False
    markdown_path = Path(str(markdown_report))
    if not _cloud_file_exists(markdown_path, policy):
        return False
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except OSError:
        return False
    maintenance_markdown = _markdown_section(markdown, "Maintenance Reference")
    return (
        all(item in html for item in REQUIRED_HTML_TOKENS)
        and all(item in markdown for item in REQUIRED_MARKDOWN_TOKENS)
        and all(item in html for item in REQUIRED_BENCHMARK_ROW_TOKENS)
        and all(item in markdown for item in REQUIRED_BENCHMARK_ROW_TOKENS)
        and all(item in html for item in REQUIRED_PROVENANCE_TOKENS)
        and all(item in markdown for item in REQUIRED_PROVENANCE_TOKENS)
        and all(item in html for item in REQUIRED_MAINTENANCE_REFERENCE_TOKENS)
        and all(
            item in maintenance_markdown
            for item in REQUIRED_MAINTENANCE_REFERENCE_TOKENS
        )
    )


def _markdown_section(markdown: str, heading: str) -> str:
    start_token = f"## {heading}"
    start = markdown.find(start_token)
    if start == -1:
        return ""
    next_start = markdown.find("\n## ", start + len(start_token))
    if next_start == -1:
        return markdown[start:]
    return markdown[start:next_start]


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

def _local_install_manifest_ready(
    manifest: dict[str, Any],
    policy: CloudStoragePolicy,
) -> bool:
    report = _phase_report(manifest, "local-install-manifest", policy)
    if report is None:
        return False
    target_state = report.get("target_repo_state", {})
    if not isinstance(target_state, dict):
        return False
    return (
        report.get("passed") is True
        and report.get("ready_for_local_install") is True
        and report.get("install_action_performed") is False
        and target_state.get("exists") is True
        and bool(target_state.get("expected_branch"))
        and target_state.get("current_branch") == target_state.get("expected_branch")
        and bool(target_state.get("head"))
        and bool(target_state.get("expected_commit"))
        and target_state.get("expected_commit_matches") is True
        and target_state.get("allowed_cloud_path") is True
        and _commit_matches(
            str(target_state.get("head_full") or target_state.get("head") or ""),
            str(target_state.get("expected_commit") or ""),
        )
        and target_state.get("is_dirty") is False
    )


def _commit_matches(head: str, expected_commit: str) -> bool:
    head = head.strip().lower()
    expected_commit = expected_commit.strip().lower()
    if not head or len(expected_commit) < MIN_EXPECTED_COMMIT_LENGTH:
        return False
    return head.startswith(expected_commit)


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
