from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_card_states_no_production_adapter_until_acceptance_gate_passes() -> None:
    text = (ROOT / "model_card.md").read_text(encoding="utf-8")

    assert (
        "Not production-ready until acceptance_gate_report_<run_id>.json has "
        "passed=true"
    ) in text
    assert "local install readiness manifest" in text
    assert "does not install the adapter" in text
    assert "dangerous_os_command_output_max" in text
    assert "local_artifact_usage_max" in text
    assert "locked_v2_eval_intent_accuracy_min >= 95%" in text
    assert "locked_v2_eval_exact_match_min >= 95%" in text
    assert "v2_dangerous_os_command_rows_min" in text
    assert "v2_unsupported_tool_hallucination_rows_min" in text


def test_readme_documents_v2_300k_workflow_and_local_readiness() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "v2 300k" in text
    assert "old rows are kept only when they pass v2 validation" in text
    assert "local-install-manifest" in text
    assert "benchmark_report_<run_id>.html" in text
    assert "benchmark_report_<run_id>.md" in text
    assert "Benchmark Columns" in text
    assert "Maintenance Reference" in text
    assert "provider_policy_sha256" in text
    assert "contract_manifest_sha256" in text
    assert "adapter_total_bytes" in text
    assert "drive_account_matches" in text
    assert "old_rows_kept" in text
    assert "adapter_aggregate_sha256" in text
    assert "dangerous OS command" in text
    assert "unsupported or hallucinated tool" in text
    assert "python3 -m pip install -r requirements.txt" not in text
    assert "Kaggle-backed storage" not in text
    assert "Google Drive" in text
    assert "contract_snapshots/gp4_ws_ws-deep-rebuild-3526" in text
    assert "strict `GP4_WS` contract snapshot" in text
    assert "GP4_WS_EXPECTED_COMMIT" in text
    assert "must be set before the notebook clones or reuses `GP4_WS`" in text
    assert "--old-dataset \"$GP4_OLD_DATASET\"" in text
    assert "python3 scripts/colab_readiness_report.py" in text
    assert (
        "robot execution remains behind validation, safety, planning, "
        "approval, and hardware gates"
    ) in text


def test_local_install_readiness_doc_exists() -> None:
    text = (ROOT / "docs/local_install_readiness.md").read_text(encoding="utf-8")

    assert "This document is a readiness checklist, not an install procedure." in text
    assert "completion_audit_<run_id>.json has passed=true" in text
    assert "completion_audit_v2_<run_id>.json" not in text
    assert "install_action_performed=false" in text
    assert "target_repo_state.exists=true" in text
    assert "target_repo_state.current_branch=ws-deep-rebuild-3526" in text
    assert "target_repo_state.expected_commit_matches=true" in text
    assert "benchmark_report_<run_id>.html" in text
    assert "provider_policy_sha256" in text
    assert "contract_manifest_sha256" in text
    assert "adapter_total_bytes" in text
    assert "drive_account_matches" in text
    assert "old_rows_kept" in text
    assert "adapter_aggregate_sha256" in text
