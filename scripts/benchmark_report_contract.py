#!/usr/bin/env python3
from __future__ import annotations

BENCHMARK_COLUMNS = ("source", "metric", "actual", "operator", "threshold", "passed")
BENCHMARK_MARKDOWN_HEADER = "| source | metric | actual | operator | threshold | passed |"

REPORT_SCHEMA_VERSION = "gp4_v2_benchmark_report_v1"

CHART_KEY_ACTUAL_VS_THRESHOLD = "benchmark_actual_vs_threshold"
CHART_KEY_ACCEPTANCE_GATE_STATUS = "acceptance_gate_status"
CHART_KEY_SCENARIO_TAG_DISTRIBUTION = "scenario_tag_distribution"
CHART_KEY_V2_QUOTA_FAILURES = "v2_quota_failures"
REQUIRED_CHART_KEYS = (
    CHART_KEY_ACTUAL_VS_THRESHOLD,
    CHART_KEY_ACCEPTANCE_GATE_STATUS,
    CHART_KEY_V2_QUOTA_FAILURES,
    CHART_KEY_SCENARIO_TAG_DISTRIBUTION,
)

REQUIRED_HTML_TOKENS = (
    "<table",
    "<svg",
    "Benchmark Columns",
    "Actual vs Threshold",
    "Acceptance Gate Status",
    "V2 Quota Failures",
    "Scenario Tag Distribution",
    "Maintenance Reference",
)

REQUIRED_MARKDOWN_TOKENS = (
    "# GP4 V2 Benchmark Report",
    "## Benchmark Columns",
    "## V2 Quota Failures",
    "## Maintenance Reference",
    "## Provenance",
    BENCHMARK_MARKDOWN_HEADER,
)

REQUIRED_BENCHMARK_ROW_TOKENS = (
    "provider_cloud_storage_ready",
    "install_action_performed",
    "gp4_ws_branch",
    "gp4_ws_expected_commit",
    "gp4_ws_expected_commit_matches",
    "eval_contract_branch",
    "target_repo_branch",
    "target_repo_commit",
    "locked_v2_eval_intent_accuracy",
    "locked_v2_eval_exact_match",
    "locked_v2_eval_rows",
    "dangerous_os_command_output",
    "local_artifact_usage",
    "previous_adapter_artifact_exists",
    "previous_run_matched",
    "dangerous_os_command",
    "unsupported_tool_hallucination",
)

REQUIRED_PROVENANCE_TOKENS = (
    "provider_policy_sha256",
    "contract_manifest_sha256",
    "adapter_total_bytes",
)

REQUIRED_MAINTENANCE_REFERENCE_TOKENS = (
    "drive_account_confirmed",
    "drive_account_matches",
    "runtime_account_confirmed",
    "cross_account_runner",
    "gp4_ws_branch",
    "gp4_ws_expected_commit",
    "gp4_ws_expected_commit_matches",
    "old_dataset_count",
    "old_rows_kept",
    "new_rows_requested",
    "raw_candidate_rows_requested",
    "previous_adapter_artifact_exists",
    "previous_run_matched",
    "adapter_aggregate_sha256",
)
