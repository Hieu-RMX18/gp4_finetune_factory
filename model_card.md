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

A Colab pilot adapter has been imported at `models/qwen25_gp4_lora_pilot`.
The adapter artifact exists, but the current held-out evaluation does not pass
the acceptance gates:

- JSON parse success: 1.000.
- Semantic IR success: 0.818, below the 0.98 gate.
- Held-out intent accuracy: 0.818, below the 0.95 gate.
- Safety leakage gates are clean: no `primitive_type`, hardware execution
  claims, raw trajectories, safety bypasses, or unsafe command acceptance.

Known raw-output failures are alias drift on `set_speed` and `draw_shape`
circle requests. The next accepted training run should include more canonical
examples for those intents and re-run inference plus acceptance gates before
this adapter is used outside the factory.

A refreshed retrain input bundle has been prepared at
`artifact_downloads/gp4_finetune_factory_retrain_bundle.zip`, but it has not
produced a replacement adapter yet.
