# GP4 Qwen2.5 Semantic IR Fine-Tune Factory

This repository builds a reproducible cloud-run dataset and evaluation factory for fine-tuning `Qwen/Qwen2.5-7B-Instruct` to draft GP4 Semantic IR JSON for the existing `llm_gateway`.

The model is not a robot controller. Training targets must only produce Semantic IR with an `intent` field or a safe error object. Runtime execution still belongs to the ROS2 path:

`HMI -> llm_gateway -> validator -> safety -> motion_core -> hw_adapter -> MotoROS2`

## Cloud-Only Runtime Policy

Runtime datasets, reports, checkpoints, adapters, inference outputs, and final
packages must be written to Google Drive only. The local
repo is for source code, tests, configs, schemas, and notebook templates only.
Non-dry-run cloud phases require `CLOUD_ROOT`; model caches (`HF_HOME`,
`TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`,
`WANDB_DIR`, and `TMPDIR`) must resolve under that cloud root before training
libraries are imported.

## Safety Contract

- No training target may contain `primitive_type`; that is reserved for the raw/backward-compatible path.
- No target may contain raw trajectories, ROS topic/service calls, MotoROS2 calls, or hardware execution claims.
- Ambiguous, unsafe, unknown IO, or unverified D435i object commands must become safe error payloads.
- Synthetic generation is blocked until at least 50 handwritten seed rows pass strict validation.
- API keys and platform tokens must stay in environment variables or platform secret managers, never in source files.
- DeepSeek generation uses `DEEPSEEK_API_KEY` with `DEEPSEEK_BASE_URL=https://api.deepseek.com`. If DeepSeek fails, the generator can fall back to an OpenAI-compatible 9router endpoint from `OPENAI_BASE_URL` with `OPENAI_MODEL=gpt-5.4`.

## Source Checks

```bash
make test
make validate-react-ir
```

The starter seed file has only a small sample. Add handwritten examples under `data/seed/` until the seed gate reaches 50 validated rows before running any synthetic generation.

## Dependency Manifests

`requirements.txt` is the lightweight source/test dependency set.
`requirements-cloud.txt` declares the Colab train/infer runtime used by the
Google Drive workflow. `requirements-local-adapter.txt` declares the
packages needed to load an accepted adapter locally after a readiness manifest
has passed. This repository does not install the local adapter automatically.

## v2 300k Readiness Workflow

The v2 target is 300k accepted training rows total. The old rows are kept only when they pass v2 validation, then new generated rows fill the remaining target.
The v2 gates require explicit coverage for singularity, wrist flip, joint wrap,
timeout/abort/recovery, approval-required, collision/limit edge, dangerous OS command,
and unsupported or hallucinated tool scenarios.

In Colab, run the readiness preflight before orchestration:

```bash
python3 scripts/colab_readiness_report.py \
  --cloud-root "$CLOUD_ROOT" \
  --run-id "$RUN_ID" \
  --gp4-ws "$GP4_WS" \
  --old-dataset "$GP4_OLD_DATASET" \
  --previous-adapter "$GP4_PREVIOUS_ADAPTER" \
  --expected-commit "$GP4_WS_EXPECTED_COMMIT" \
  --report "$CLOUD_ROOT/reports/colab_readiness_$RUN_ID.json"
```

Then run the cloud workflow with the previous accepted dataset and adapter:

```bash
python3 scripts/cloud_orchestrator.py \
  --run-id "$RUN_ID" \
  --cloud-root "$CLOUD_ROOT" \
  --seed data/seed/gp4_seed_starter.jsonl \
  --source-plan docs/superpowers/plans/2026-05-20-gp4-v2-300k-readiness.md \
  --phases ignored \
  --preset v2-300k \
  --old-dataset "$GP4_OLD_DATASET" \
  --previous-adapter "$GP4_PREVIOUS_ADAPTER"
```

For adapter-only reuse, omit `--old-dataset` and keep
`--previous-adapter "$GP4_PREVIOUS_ADAPTER"` pointed at a prior Drive adapter.
That mode does not import old rows; use it to generate the full 300k accepted-row target from new rows while resuming training from the prior adapter.

In Colab, use the Google Drive account `johnwickiller4444@gmail.com` and keep
`GP4_DRIVE_ROOT=/content/drive/MyDrive/gp4_finetune_factory` unless the Drive
folder is intentionally moved. The notebook writes a Drive account hint and
requires an operator confirmation artifact for the same email; set
`GP4_DRIVE_ACCOUNT_CONFIRMED=johnwickiller4444@gmail.com` only after confirming
the mounted Drive account. To reuse a previous accepted run, set
`GP4_PREVIOUS_RUN_ID=<previous_run_id>` or point `GP4_OLD_DATASET` directly at
the prior `data/validated/accepted_300k.jsonl` file under Google Drive before
running the notebook. Set `GP4_PREVIOUS_ADAPTER` to the prior
`models/qwen25_gp4_lora` folder under Google Drive, or let the notebook resolve
it from `GP4_PREVIOUS_RUN_ID`.
If Drive has a previous adapter but no usable previous accepted dataset, the
notebook uses adapter-only reuse and passes `--allow-adapter-only-reuse` to the
readiness report; the orchestrator then requests 300k new rows and keeps
`old_rows_kept=0`.

The Colab notebook also prepares a read-only `gp4_ws` contract snapshot at
`$CLOUD_ROOT/contract_snapshots/gp4_ws_ws-deep-rebuild-3526`; any pre-set
`GP4_WS` must still live under `$CLOUD_ROOT/contract_snapshots`. Eval uses that
strict `GP4_WS` contract snapshot with fallback disabled.
`GP4_WS_EXPECTED_COMMIT` must be set before the notebook clones or reuses `GP4_WS`;
the notebook fails if the snapshot commit does not match that pin.
`GP4_FACTORY_SOURCE_EXPECTED_COMMIT` must be set before the notebook clones or unpacks the factory source; clone fallback checks out that commit, and Drive
source bundles must include `source_revision.json` with `factory_source_commit`.
The readiness manifest must record the expected branch and matching
expected commit before it can report `ready_for_local_install=true`.

After acceptance passes, build readiness metadata only:

```bash
make local-install-manifest CLOUD_ROOT="$CLOUD_ROOT" RUN_ID="$RUN_ID" GP4_WS="$GP4_WS" GP4_WS_EXPECTED_COMMIT="$GP4_WS_EXPECTED_COMMIT"
```

Adapter files remain in Google Drive storage; this target records readiness
metadata and does not perform a local adapter install.

The v2 run writes `benchmark_report_<run_id>.html` and
`benchmark_report_<run_id>.md` under `CLOUD_ROOT/reports/`. The HTML report
includes the Benchmark Columns table plus Actual vs Threshold, Acceptance Gate Status,
V2 Quota Failures, and Scenario Tag Distribution charts. The Markdown
report includes the same benchmark columns plus Provenance and a Maintenance Reference
section for comparing later upgrade runs. The audit-required report
contract is centralized in
`scripts/benchmark_report_contract.py`; keep README/docs/tests aligned with
that module when adding columns, chart keys, or reference rows.

The benchmark report must keep provenance rows for `provider_policy_sha256`,
`contract_manifest_sha256`, `adapter_total_bytes`, `gp4_ws_branch`,
`gp4_ws_expected_commit`, and `gp4_ws_expected_commit_matches`. Its
Maintenance Reference must keep `drive_account_confirmed`,
`drive_account_matches`, `old_dataset_count`, `old_rows_kept`,
`new_rows_requested`, `previous_adapter_artifact_exists`,
`previous_run_matched`, and `adapter_aggregate_sha256` so later upgrade runs
can compare Drive account continuity, old dataset reuse, previous adapter
reuse, gp4_ws contract pinning, and adapter checksums.

This repository does not install the adapter into the local robot workspace.
The robot execution remains behind validation, safety, planning, approval, and hardware gates.

## Files

- `configs/dataset_spec.yaml` defines the target mix, seed gate, LoRA defaults, and acceptance gates.
- `schemas/master_example.schema.json` validates local master JSONL rows.
- `schemas/semantic_ir.schema.json` is the simplified generator-facing schema.
- `scripts/extract_repo_contract.py` reads the current contract from `GP4_WS` or `--repo`.
- `scripts/validate_dataset.py` enforces JSON-only Semantic IR and safety leakage checks.
- `scripts/dedupe_dataset.py`, `scripts/build_splits.py`, and `scripts/render_review_html.py` prepare reviewable training data.
- `scripts/generate_batch_openai.py` currently enforces the seed gate and refuses generation until the seed set is ready.
- `scripts/eval_model_outputs.py` computes offline JSON, Semantic IR, and intent metrics.

## Cloud Notebook Training

Use `notebooks/colab_gp4_react_qwen25_qlora.ipynb` with `CLOUD_ROOT` pointing
at Google Drive. Build a source-only bundle directly into cloud storage:

```bash
CLOUD_ROOT=/content/drive/MyDrive/gp4_finetune_factory/<run_id> make retrain-bundle
```

To open the approved Colab notebook in Brave from this branch:

```bash
make open-colab-brave
```

No adapter is accepted until the cloud run writes
`acceptance_gate_report_<run_id>.json` under `CLOUD_ROOT` with `passed=true`.
Notebook completion or local artifact files are not acceptance evidence. After
a cloud run finishes, audit the evidence with:

```bash
CLOUD_ROOT=/content/drive/MyDrive/gp4_finetune_factory/<run_id> RUN_ID=<run_id> make audit-cloud-run
```
