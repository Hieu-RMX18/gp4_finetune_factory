# GP4 Qwen2.5-7B Semantic IR LoRA

## Intended Use

This adapter is intended to improve JSON-only Semantic IR drafting for the GP4 ROS2 + LLM HMI flow. It must not be used as a direct motion controller.

## Output Contract

Valid outputs are exactly one JSON object:

- Semantic IR with an `intent` field for safe, sufficiently specified requests.
- A safe error object for ambiguous, unsafe, missing-slot, or unverified perception requests.

The adapter must not output `primitive_type`, raw trajectories, ROS commands, MotoROS2 calls, or hardware execution claims.

## Safety Boundary

All real or simulated execution remains behind the existing ROS2 validation and safety gates:

`HMI -> llm_gateway -> validator -> safety -> motion_core -> hw_adapter -> MotoROS2`

The model does not replace `/validate_command`, safety checks, MoveIt planning, human approval, or hardware mode gates.

## Evaluation Gates

- JSON parse success >= 99%.
- Semantic IR success >= 98%.
- Normal-output `primitive_type` leakage = 0.
- Hardware execution claims = 0.
- Raw trajectory outputs = 0.
- Safety bypass outputs = 0.
- Unsafe command acceptance = 0.
- Held-out intent accuracy >= 95%.

## Current Status

No accepted adapter is stored in this local repository. The next accepted
training run must execute in an approved cloud runtime, write checkpoints,
adapter files, reports, inference outputs, and packages under `CLOUD_ROOT`, and
produce `acceptance_gate_report_<run_id>.json` with `passed=true`.

Previous pilot evaluation showed alias drift on `set_speed` and `draw_shape`
circle requests. The next accepted cloud run should include canonical coverage
for those intents and re-run held-out inference plus acceptance gates before
the adapter is used outside the factory.
