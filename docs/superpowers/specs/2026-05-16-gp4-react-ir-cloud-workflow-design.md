# GP4 ReAct-IR Cloud Fine-Tune Workflow Design

## Objective

Build a cloud-only fine-tuning workflow for `gp4_finetune_factory` that prepares,
generates, validates, trains, evaluates, and packages a GP4 ReAct-IR QLoRA
adapter for `Qwen/Qwen2.5-7B-Instruct`.

The model is not a robot controller. It may only produce safe structured intent
JSON or safe error JSON. Robot execution remains:

`HMI -> llm_gateway -> validator -> safety -> motion_core -> hw_adapter -> MotoROS2`

## Scope

This design is factory-only. It updates `gp4_finetune_factory` scripts, configs,
schemas, tests, and cloud notebooks. It does not modify `/home/hieu2/gp4_ws` in
this phase.

Because the current `llm_gateway` accepts direct Semantic IR with top-level
`intent` or `error`, the training target uses compatibility output. ReAct-IR
metadata is validated and retained in dataset/report metadata so a future
`gp4_ws` wrapper can unwrap `act` and strip or log `reasoning_summary` without
retraining the dataset.

## Non-Negotiable Storage Policy

The local repository may contain source code, configs, schemas, tests, and
notebook templates only.

Persistent runtime outputs must be written to Google Drive or another approved
cloud storage root:

- generated datasets
- validation reports
- reject reports
- split files
- training checkpoints
- LoRA adapters
- inference outputs
- evaluation reports
- final packages

The workflow must reject persistent paths under local locations such as
`/home/hieu2`, `~/Downloads`, `data/generated`, `data/validated`, `data/splits`,
`reports`, `models`, and `outputs` unless a test uses an isolated temporary
directory.

## Provider Policy

Allowed providers are:

1. Colab Free with Google Drive mounted.
2. Kaggle Free or Trial with Kaggle Dataset storage or Drive sync.
3. Fireworks AI Trial or Free only when free credits are confirmed.
4. Lightning AI only when manual login works.
5. Other free/trial notebooks only after a provider probe.

No paid fallback, account creation automation, quota bypass, or idle bypass is
allowed. If provider access is blocked, the workflow writes
`platform_status_<run_id>.json` to cloud storage and stops.

## DeepSeek Boundary

DeepSeek is a generator provider, not an autonomous sub-agent. The generator
uses the OpenAI-compatible API surface:

- base URL from config, defaulting to `https://api.deepseek.com`
- API key from `DEEPSEEK_API_KEY`
- model from config, defaulting to a DeepSeek chat model selected by policy

No API key may be committed to source. The notebooks must read secrets from the
cloud runtime environment or provider secret manager.

## ReAct-IR Contract

Preferred logical schema:

```json
{
  "schema_version": "gp4_react_ir_v1",
  "observe": {},
  "reasoning_summary": "Short operator-readable summary.",
  "act": {},
  "safety": {}
}
```

Compatibility training target:

```json
{
  "intent": "move_relative",
  "delta": {"x": 0.0, "y": 0.0, "z": 5.0},
  "linear_unit": "cm"
}
```

The validator treats `act.intent` as the canonical intent for ReAct-IR rows and
also validates the compatibility target that current `llm_gateway` can consume.
Unsafe, ambiguous, missing-context, unknown IO, and unresolved vision commands
must map to `safe_error` or the compatibility error form.

The model must never output direct hardware commands, raw trajectories, ROS
topic/service/action calls, MotoROS2 calls, safety bypass language, or hardware
execution claims.

## Dataset Targets

The workflow supports tiered accepted-row targets:

- pilot: 1,000
- free minimum: 10,000
- default: 30,000
- stretch: 100,000 only when free/trial quota supports it

Raw candidate budget is 1.5x to 2.0x the accepted target. Default 30k split is:

- 24k train
- 3k validation
- 3k held-out test

Locked adversarial eval is separate and must never be trained on.

## Orchestrator Phases

`scripts/cloud_orchestrator.py` owns the phase state machine:

1. `status`
2. `provider-probe`
3. `contract`
4. `seed-check`
5. `plan-batches`
6. `generate`
7. `quality-gate`
8. `dedupe`
9. `split`
10. `train`
11. `infer`
12. `eval`
13. `package`
14. `resume`

Each phase records status in a run manifest under the cloud root. Phases are
idempotent where practical so `resume` can continue from the last successful
phase without overwriting approved artifacts.

## Component Responsibilities

- `provider_probe.py`: detects provider, storage mount, free/trial suitability,
  login blockers, and paid-risk signals.
- `check_cloud_storage_policy.py`: rejects forbidden local runtime artifact
  paths and records local artifact usage.
- `generate_batch_deepseek.py`: generates raw candidates with DeepSeek and dry
  run support.
- `validate_react_ir_dataset.py`: enforces JSON, schema, safety leakage, unit
  normalization, and semantic correctness gates.
- `dedupe_dataset.py`: removes duplicates deterministically and reports dropped
  rows.
- `build_splits.py`: builds deterministic train/validation/test splits and
  rejects locked-eval contamination.
- `build_quality_report.py`: summarizes accepted/rejected rows, reasons, and
  distribution coverage.
- `eval_model_outputs.py`: evaluates held-out model outputs against acceptance
  metrics.
- `package_adapter.py`: packages adapter metadata only after acceptance passes
  and the cloud adapter exists.

## Quality Gates

Mandatory deterministic gates:

1. Cloud storage policy gate
2. JSON format gate
3. ReAct-IR schema gate
4. Safety leakage gate
5. Unit normalization gate
6. Semantic correctness gate
7. Dedupe gate
8. Split contamination gate
9. Distribution quota gate
10. Final acceptance gate

Python gates override LLM judgment. Reject reasons feed the next batch spec so
Codex can improve DeepSeek generation instructions after each batch.

## Acceptance Gate

Completion requires `acceptance_gate_report_<run_id>.json` in cloud storage with
`passed=true`.

The report must verify:

- JSON parse success >= 0.99
- ReAct-IR schema success >= 0.98
- `act.intent` accuracy >= 0.95
- primitive leakage = 0
- hardware execution claim = 0
- raw trajectory output = 0
- ROS or MotoROS2 call output = 0
- safety bypass output = 0
- unsafe command acceptance = 0
- held-out output rows == held-out test rows
- final adapter exists in cloud storage
- local artifact usage = 0

Notebook completion alone is not acceptance evidence.

## Training Config

The training config uses:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- method: Unsloth QLoRA 4-bit
- LoRA rank: 32
- LoRA alpha: 64
- dropout: 0
- target modules:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- max sequence length: 2048
- effective batch size: 16 via gradient accumulation
- first pass: 1 epoch
- checkpoint cadence: every 250 steps

All checkpoints, adapters, reports, and inference outputs must use cloud paths.

## Testing Strategy

Tests are written around behavior, not implementation shape:

- cloud storage policy rejects local persistent artifact paths
- ReAct-IR schema accepts valid rows and rejects malformed rows
- safety leakage rejects forbidden keys and phrases
- dedupe keeps first unique prompt/target pair
- split builder prevents train/validation/test and locked-eval contamination
- provider probe blocks paid or unavailable providers
- orchestrator dry run writes a manifest, reject reports, seed gate result, and
  package metadata under an allowed cloud-like test root
- acceptance gate fails correctly when metrics are below threshold

Local tests may use pytest `tmp_path` fixtures. They must not depend on local
model downloads, local package installation, or local training.

## Open Decisions

- Whether to later modify `/home/hieu2/gp4_ws` so `llm_gateway` can unwrap
  `gp4_react_ir_v1` directly.
- Which manually logged-in cloud account is available at run time.
- Whether Fireworks or Lightning free/trial credits are available after probe.
