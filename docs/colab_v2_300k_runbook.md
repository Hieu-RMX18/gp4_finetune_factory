# Colab V2 300k Runbook

Use this checkpoint when resuming the v2 300k fine-tune from Colab. Do not run training locally.

## Required Colab State

- Google account: `johnwickiller4444@gmail.com`
- Drive root: `/content/drive/MyDrive/gp4_finetune_factory`
- Source branch: `codex/gp4-react-ir-cloud-workflow`
- Factory source commit: `299f5384c1e544b21a4547850685b8c8315f4b3d`
- `GP4_FACTORY_SOURCE_EXPECTED_COMMIT=299f5384c1e544b21a4547850685b8c8315f4b3d`
- `gp4_ws` branch: `ws-deep-rebuild-3526`
- `GP4_WS_EXPECTED_COMMIT=3bbcb0726a4c3305c086c93e2b1e4a320471090b`

If `drive.mount('/content/drive')` opens a Google Drive permission tab and the
setup cell reports `Google Drive mount failed`, stop at that prompt. Approve the Google Drive permission prompt, return to the notebook, and rerun the first setup cell.

## Reuse Inputs

Set `GP4_PREVIOUS_RUN_ID`, or set explicit Drive paths before running the
notebook. The normal reuse path uses both
`GP4_OLD_DATASET=<previous_run>/data/validated/accepted_300k.jsonl` and
`GP4_PREVIOUS_ADAPTER=<previous_run>/models/qwen25_gp4_lora`. If no old dataset
is available, adapter-only reuse is allowed when `GP4_PREVIOUS_ADAPTER` points
at a prior Drive adapter; that mode generates the full 300k accepted-row target from new rows and keeps `old_rows_kept=0`.
When Drive already has a prior 300k accepted dataset but the reusable adapter is
from a different prior run, the notebook uses explicit mixed-prior-artifacts
readiness. Current-run datasets or adapters are still rejected.

If all three reuse inputs are unset, the Colab notebook scans the Drive root
for the newest prior accepted dataset and the newest prior adapter. It can
select the newest adapter checkpoint under
`models/qwen25_gp4_lora/checkpoint-*` when the root adapter folder does not yet
contain final adapter weights.

```bash
export GP4_DRIVE_ROOT=/content/drive/MyDrive/gp4_finetune_factory
export GP4_DRIVE_ACCOUNT_CONFIRMED=johnwickiller4444@gmail.com
export GP4_PREVIOUS_RUN_ID=<previous_run_id>
export GP4_OLD_DATASET=/content/drive/MyDrive/gp4_finetune_factory/<previous_run_id>/data/validated/accepted_300k.jsonl
export GP4_PREVIOUS_ADAPTER=/content/drive/MyDrive/gp4_finetune_factory/<previous_run_id>/models/qwen25_gp4_lora
export GP4_FACTORY_SOURCE_EXPECTED_COMMIT=299f5384c1e544b21a4547850685b8c8315f4b3d
export GP4_WS_EXPECTED_COMMIT=3bbcb0726a4c3305c086c93e2b1e4a320471090b
```

The notebook rejects `GP4_OLD_DATASET`, `GP4_PREVIOUS_ADAPTER`, and `GP4_WS` when they are outside the approved Google Drive roots. It rejects clone fallback or source bundles when the factory source commit does not match `GP4_FACTORY_SOURCE_EXPECTED_COMMIT`.
If `GP4_FACTORY_SOURCE_EXPECTED_COMMIT` is unset, the notebook resolves the
current pushed `codex/gp4-react-ir-cloud-workflow` branch commit and records it
in `manifests/factory_source_revision.json`; set the explicit commit above for
fully reproducible reruns.

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

The executed notebook source is also copied to
`$CLOUD_ROOT/notebooks/colab_gp4_react_qwen25_qlora.ipynb`, with copy metadata in
`$CLOUD_ROOT/manifests/colab_notebook_copy.json`.

The benchmark reports must include Benchmark Columns, Actual vs Threshold,
Acceptance Gate Status, V2 Quota Failures, Scenario Tag Distribution,
Provenance, and Maintenance Reference sections. The completion audit must pass
before any downstream install readiness claim is accepted.
If a notebook subprocess fails, inspect `<label>_failure_${RUN_ID}.json` under
`$CLOUD_ROOT/reports/`; it records the command, return code, and stdout/stderr tails.
`local-install-manifest_${RUN_ID}.json` is a Drive-stored readiness artifact, not a local install action.
This runbook produces cloud readiness evidence only and does not perform a local adapter install.
