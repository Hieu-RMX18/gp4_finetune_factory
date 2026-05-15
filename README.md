# GP4 Qwen2.5 Semantic IR Fine-Tune Factory

This repository builds a local, reproducible dataset and evaluation factory for fine-tuning `Qwen/Qwen2.5-7B-Instruct` to draft GP4 Semantic IR JSON for the existing `llm_gateway`.

The model is not a robot controller. Training targets must only produce Semantic IR with an `intent` field or a safe error object. Runtime execution still belongs to the ROS2 path:

`HMI -> llm_gateway -> validator -> safety -> motion_core -> hw_adapter -> MotoROS2`

## Safety Contract

- No training target may contain `primitive_type`; that is reserved for the raw/backward-compatible path.
- No target may contain raw trajectories, ROS topic/service calls, MotoROS2 calls, or hardware execution claims.
- Ambiguous, unsafe, unknown IO, or unverified D435i object commands must become safe error payloads.
- Synthetic generation is blocked until at least 50 handwritten seed rows pass strict validation.
- API keys and platform tokens must stay in environment variables or platform secret managers, never in source files.

## First-Wave Workflow

```bash
python3 -m pip install -r requirements.txt
make contract
make test
make validate-seed
make review
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

## Browser And Training

Brave Profile 9 and `notebooks/colab_qwen25_gp4_unsloth.ipynb` are available
for Colab/Kaggle/Lightning automation after seed review. The current imported
pilot adapter is stored at `models/qwen25_gp4_lora_pilot`, with Colab training
status in `reports/colab_training_status.json`.

The current adapter does not pass the acceptance gates in
`reports/acceptance_gate_report.json`. Treat it as a pilot artifact only until
the failed Semantic IR and intent-accuracy gates pass on fresh held-out
inference.

A refreshed retrain bundle with targeted `set_speed` and `draw_shape` coverage
is available at `artifact_downloads/gp4_finetune_factory_retrain_bundle.zip`;
see `reports/retrain_readiness_report.json` for the current blocker and next
required Colab steps.
