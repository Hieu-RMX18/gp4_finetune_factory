import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARDENED_COLAB_READINESS_COMMIT = "fb5f2a6dbf23a9db325ae507eb74cc426822679c"


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
    assert "Actual vs Threshold" in text
    assert "Acceptance Gate Status" in text
    assert "Maintenance Reference" in text
    assert "provider_policy_sha256" in text
    assert "contract_manifest_sha256" in text
    assert "adapter_total_bytes" in text
    assert "drive_account_matches" in text
    assert "old_rows_kept" in text
    assert "adapter-only reuse" in text
    assert "generate the full 300k accepted-row target from new rows" in text
    assert "adapter_aggregate_sha256" in text
    assert "previous_adapter_artifact_exists" in text
    assert "previous_run_matched" in text
    assert "gp4_ws_branch" in text
    assert "gp4_ws_expected_commit" in text
    assert "gp4_ws_expected_commit_matches" in text
    assert "Adapter files remain in Google Drive storage" in text
    assert "Google Drive or approved cloud storage" not in text
    assert "dangerous OS command" in text
    assert "unsupported or hallucinated tool" in text
    assert "python3 -m pip install -r requirements.txt" not in text
    assert "Kaggle-backed storage" not in text
    assert "Google Drive" in text
    assert "contract_snapshots/gp4_ws_ws-deep-rebuild-3526" in text
    assert "strict `GP4_WS` contract snapshot" in text
    assert "GP4_WS_EXPECTED_COMMIT" in text
    assert "GP4_FACTORY_SOURCE_EXPECTED_COMMIT" in text
    assert "must be set before the notebook clones or reuses `GP4_WS`" in text
    assert "must be set before the notebook clones or unpacks the factory source" in text
    assert "--old-dataset \"$GP4_OLD_DATASET\"" in text
    assert "--previous-adapter \"$GP4_PREVIOUS_ADAPTER\"" in text
    preflight = text.split("python3 scripts/colab_readiness_report.py", 1)[1].split(
        "Then run the cloud workflow",
        1,
    )[0]
    assert "--previous-adapter \"$GP4_PREVIOUS_ADAPTER\"" in preflight
    assert "GP4_PREVIOUS_ADAPTER" in text
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
    assert "local-install-manifest_<run_id>.json" in text
    assert "local_install_manifest_<run_id>.json" not in text
    assert "install_action_performed=false" in text
    assert "target_repo_state.exists=true" in text
    assert "target_repo_state.current_branch=ws-deep-rebuild-3526" in text
    assert "target_repo_state.expected_commit_matches=true" in text
    assert "target_repo_state.allowed_cloud_path=true" in text
    assert "target_repo_state.is_dirty=false" in text
    assert "benchmark_report_<run_id>.html" in text
    assert "provider_policy_sha256" in text
    assert "contract_manifest_sha256" in text
    assert "adapter_total_bytes" in text
    assert "drive_account_matches" in text
    assert "old_rows_kept" in text
    assert "adapter_aggregate_sha256" in text
    assert "previous_adapter_artifact_exists" in text
    assert "previous_run_matched" in text
    assert "gp4_ws_branch" in text
    assert "gp4_ws_expected_commit" in text
    assert "gp4_ws_expected_commit_matches" in text
    assert "Adapter files remain in Google Drive storage" in text


def test_colab_v2_300k_runbook_pins_drive_reuse_and_reports() -> None:
    text = (ROOT / "docs/colab_v2_300k_runbook.md").read_text(encoding="utf-8")

    assert "johnwickiller4444@gmail.com" in text
    assert "/content/drive/MyDrive/gp4_finetune_factory" in text
    assert "GP4_PREVIOUS_RUN_ID" in text
    assert "GP4_PREVIOUS_ADAPTER" in text
    assert "models/qwen25_gp4_lora" in text
    assert "GP4_OLD_DATASET" in text
    assert "accepted_300k.jsonl" in text
    assert "adapter-only reuse" in text
    assert "generates the full 300k accepted-row target from new rows" in text
    assert "GP4_WS_EXPECTED_COMMIT" in text
    assert "GP4_FACTORY_SOURCE_EXPECTED_COMMIT" in text
    assert "ws-deep-rebuild-3526" in text
    assert "3bbcb0726a4c3305c086c93e2b1e4a320471090b" in text
    assert "export GP4_FACTORY_SOURCE_EXPECTED_COMMIT=" in text
    assert "benchmark_report_${RUN_ID}.html" in text
    assert "benchmark_report_${RUN_ID}.md" in text
    assert "Actual vs Threshold" in text
    assert "Acceptance Gate Status" in text
    assert "local-install-manifest_${RUN_ID}.json" in text
    assert "`local-install-manifest_${RUN_ID}.json` is a Drive-stored readiness artifact, not a local install action." in text
    assert "completion_audit_${RUN_ID}.json" in text
    assert "<label>_failure_${RUN_ID}.json" in text
    assert "stdout/stderr tails" in text
    assert "Do not run training locally" in text
    assert "does not perform a local adapter install" in text


def test_colab_v2_300k_runbook_source_commit_includes_hardened_gates() -> None:
    text = (ROOT / "docs/colab_v2_300k_runbook.md").read_text(encoding="utf-8")
    match = re.search(r"Factory source commit: `([0-9a-f]{40})`", text)

    assert match is not None

    pinned_commit = match.group(1)
    subprocess.run(
        ["git", "cat-file", "-e", f"{pinned_commit}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            HARDENED_COLAB_READINESS_COMMIT,
            pinned_commit,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_makefile_local_manifest_defaults_match_v2_cloud_workflow() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "ADAPTER_DIR ?= $(CLOUD_ROOT)/models/qwen25_gp4_lora\n" in text
    assert "qwen25_gp4_lora_pilot" not in text
    assert (
        "LOCAL_INSTALL_MANIFEST ?= "
        "$(CLOUD_ROOT)/reports/local-install-manifest_$(RUN_ID).json\n"
    ) in text
    assert "local_install_manifest_$(RUN_ID).json" not in text


def test_env_example_requires_factory_source_commit_pin() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "GP4_FACTORY_SOURCE_EXPECTED_COMMIT=<required-factory-source-commit-sha>" in text


def test_makefile_local_manifest_target_renders_cloud_only_readiness_command() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "local-install-manifest",
            "CLOUD_ROOT=/tmp/gp4_cloud",
            "RUN_ID=test-run",
            "GP4_WS=/tmp/gp4_ws",
            "GP4_WS_EXPECTED_COMMIT=abc",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--adapter-dir \"/tmp/gp4_cloud/models/qwen25_gp4_lora\"" in result.stdout
    assert (
        "--output \"/tmp/gp4_cloud/reports/local-install-manifest_test-run.json\""
        in result.stdout
    )
    assert "scripts/build_local_install_manifest.py" in result.stdout
    assert "pip install" not in result.stdout
