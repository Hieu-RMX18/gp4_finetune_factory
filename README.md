# GP4 Qwen2.5 Semantic IR Fine-Tune Factory

This repository builds a reproducible cloud-run dataset and evaluation factory for fine-tuning `Qwen/Qwen2.5-7B-Instruct` to draft GP4 Semantic IR JSON for the existing `llm_gateway`.

The model is not a robot controller. Training targets must only produce Semantic IR with an `intent` field or a safe error object. Runtime execution still belongs to the ROS2 path:

`HMI -> llm_gateway -> validator -> safety -> motion_core -> hw_adapter -> MotoROS2`

## Cloud-Only Runtime Policy

Runtime datasets, reports, checkpoints, adapters, inference outputs, and final
packages must be written to Google Drive or approved cloud storage. The local
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
- DeepSeek generation uses `DEEPSEEK_API_KEY` with `DEEPSEEK_BASE_URL=https://api.deepseek.com`; `OPENAI_API_KEY` is not a DeepSeek fallback.

## First-Wave Workflow

```bash
python3 -m pip install -r requirements.txt
make test
make validate-react-ir
```

The starter seed file has only a small sample. Add handwritten examples under `data/seed/` until the seed gate reaches 50 validated rows before running any synthetic generation.

## Files

- `configs/dataset_spec.yaml` defines the target mix, seed gate, LoRA defaults, and acceptance gates.
- `schemas/master_example.schema.json` validates local master JSONL rows.
- `schemas/semantic_ir.schema.json` is the simplified generator-facing schema.
- `scripts/extract_repo_contract.py` reads the current contract from `/home/hieu2/gp4_ws`.
- `scripts/validate_dataset.py` enforces JSON-only Semantic IR and safety leakage checks.
- `scripts/dedupe_dataset.py`, `scripts/build_splits.py`, and `scripts/render_review_html.py` prepare reviewable training data.
- `scripts/generate_batch_openai.py` currently enforces the seed gate and refuses generation until the seed set is ready.
- `scripts/eval_model_outputs.py` computes offline JSON, Semantic IR, and intent metrics.

## Cloud Notebook Training

Use `notebooks/colab_gp4_react_qwen25_qlora.ipynb` or
`notebooks/kaggle_gp4_react_qwen25_qlora.ipynb` with `CLOUD_ROOT` pointing at
Google Drive, Kaggle-backed storage, or another approved cloud root. Build a
source-only bundle directly into cloud storage:

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
