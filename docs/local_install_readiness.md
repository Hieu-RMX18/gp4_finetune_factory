# Local Install Readiness

This document is a readiness checklist, not an install procedure.

The factory may produce a local install readiness manifest only after the cloud
adapter passes acceptance gates. The manifest records adapter checksums, the
accepted gate report path, the intended target repository, and
install_action_performed=false.

Required evidence:

- acceptance_gate_report_<run_id>.json has passed=true.
- completion_audit_<run_id>.json has passed=true.
- local-install-manifest_<run_id>.json has ready_for_local_install=true.
- local-install-manifest_<run_id>.json has install_action_performed=false.
- local-install-manifest_<run_id>.json has target_repo_state.exists=true.
- local-install-manifest_<run_id>.json has target_repo_state.allowed_cloud_path=true.
- local-install-manifest_<run_id>.json has
  target_repo_state.current_branch=ws-deep-rebuild-3526.
- local-install-manifest_<run_id>.json has a non-empty
  target_repo_state.expected_commit.
- local-install-manifest_<run_id>.json has
  target_repo_state.expected_commit_matches=true.
- local-install-manifest_<run_id>.json has a non-empty target_repo_state.head.
- local-install-manifest_<run_id>.json has target_repo_state.is_dirty=false.
- Adapter files remain in Google Drive storage until a separate install task
  is reviewed and approved.
- benchmark_report_<run_id>.html and benchmark_report_<run_id>.md exist under
  CLOUD_ROOT/reports.
- colab_readiness_<run_id>.json has drive_account.confirmed=true and
  drive_account.confirmed_email=johnwickiller4444@gmail.com.
- The benchmark report provenance keeps provider_policy_sha256,
  contract_manifest_sha256, adapter_total_bytes, gp4_ws_branch,
  gp4_ws_expected_commit, and gp4_ws_expected_commit_matches.
- The benchmark report Maintenance Reference keeps drive_account_confirmed,
  drive_account_matches, old_dataset_count, old_rows_kept, new_rows_requested,
  previous_adapter_artifact_exists, previous_run_matched, and
  adapter_aggregate_sha256.

The adapter may draft Semantic IR only. It does not replace command validation,
safety checks, MoveIt planning, human approval, hardware mode gates, or MotoROS2
execution controls.
