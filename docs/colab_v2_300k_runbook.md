# Colab V2 300k Runbook

Use this checkpoint when resuming the v2 300k fine-tune from Colab. Do not run training locally.

## Required Colab State

- Google account: `johnwickiller4444@gmail.com`
- Drive root: `/content/drive/MyDrive/gp4_finetune_factory`
- Source branch: `codex/gp4-react-ir-cloud-workflow`
- Factory source commit: `201d1a43bb01a3c1ee6dd97c5dfb2c68e16c51bd`
- `GP4_FACTORY_SOURCE_EXPECTED_COMMIT=201d1a43bb01a3c1ee6dd97c5dfb2c68e16c51bd`
- `gp4_ws` branch: `ws-deep-rebuild-3526`
- `GP4_WS_EXPECTED_COMMIT=3bbcb0726a4c3305c086c93e2b1e4a320471090b`

## Reuse Inputs

Set either `GP4_PREVIOUS_RUN_ID` or both explicit paths before running the notebook.

```bash
export GP4_DRIVE_ROOT=/content/drive/MyDrive/gp4_finetune_factory
export GP4_DRIVE_ACCOUNT_CONFIRMED=johnwickiller4444@gmail.com
export GP4_PREVIOUS_RUN_ID=<previous_run_id>
export GP4_OLD_DATASET=/content/drive/MyDrive/gp4_finetune_factory/<previous_run_id>/data/validated/accepted_300k.jsonl
export GP4_PREVIOUS_ADAPTER=/content/drive/MyDrive/gp4_finetune_factory/<previous_run_id>/models/qwen25_gp4_lora
export GP4_FACTORY_SOURCE_EXPECTED_COMMIT=201d1a43bb01a3c1ee6dd97c5dfb2c68e16c51bd
export GP4_WS_EXPECTED_COMMIT=3bbcb0726a4c3305c086c93e2b1e4a320471090b
```

The notebook rejects `GP4_OLD_DATASET`, `GP4_PREVIOUS_ADAPTER`, and `GP4_WS` when they are outside the approved Google Drive roots. It rejects clone fallback or source bundles when the factory source commit does not match `GP4_FACTORY_SOURCE_EXPECTED_COMMIT`.

## Expected Evidence

After Run All, keep these files under `$CLOUD_ROOT/reports/`:

- `colab_readiness_${RUN_ID}.json`
- `platform_status_${RUN_ID}.json`
- `acceptance_gate_report_${RUN_ID}.json`
- `local-install-manifest_${RUN_ID}.json`
- `benchmark_report_${RUN_ID}.html`
- `benchmark_report_${RUN_ID}.md`
- `benchmark-report_${RUN_ID}.json`
- `completion_audit_${RUN_ID}.json`

The benchmark reports must include Benchmark Columns, Actual vs Threshold,
Acceptance Gate Status, V2 Quota Failures, Scenario Tag Distribution,
Provenance, and Maintenance Reference sections. The completion audit must pass
before any downstream install readiness claim is accepted.
`local-install-manifest_${RUN_ID}.json` is a Drive-stored readiness artifact, not a local install action.
This runbook produces cloud readiness evidence only and does not perform a local adapter install.
