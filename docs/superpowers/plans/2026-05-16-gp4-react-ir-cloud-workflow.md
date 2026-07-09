# GP4 ReAct-IR Cloud Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cloud-only GP4 ReAct-IR fine-tuning workflow with DeepSeek generation, deterministic gates, free/trial provider probing, Drive/cloud storage persistence, and dry-run verification.

**Architecture:** Keep source code in `gp4_finetune_factory`, but require every runtime artifact path to resolve under an allowed cloud root. Use small Python scripts with JSON/YAML contracts, deterministic validators, and a phase manifest owned by `scripts/cloud_orchestrator.py`.

**Tech Stack:** Python 3, pytest, jsonschema, PyYAML, urllib OpenAI-compatible HTTP calls, Colab/Kaggle notebook templates.

---

## File Structure

- Create `scripts/check_cloud_storage_policy.py`: path policy, local-artifact scanner, CLI report writer.
- Create `scripts/provider_probe.py`: free/trial provider detection and blocking status reports.
- Create `scripts/generate_batch_deepseek.py`: DeepSeek OpenAI-compatible generation with dry-run and seed gate.
- Create `scripts/validate_react_ir_dataset.py`: JSON, ReAct schema, compatibility target, leakage, unit, semantic, and distribution gates.
- Modify `scripts/dedupe_dataset.py`: keep existing behavior and add report output.
- Modify `scripts/build_splits.py`: add locked eval contamination checks and split report output.
- Create `scripts/build_quality_report.py`: merge validation, rejection, distribution, dedupe, split, and eval summaries.
- Modify `scripts/eval_model_outputs.py`: evaluate ReAct-IR and compatibility outputs with row-count checks.
- Create `scripts/package_adapter.py`: verify cloud adapter existence and write package metadata only after acceptance passes.
- Create `scripts/cloud_orchestrator.py`: phase runner and cloud manifest.
- Create configs: `provider_policy.yaml`, `generation_policy.yaml`, `react_ir_policy.yaml`, `train_qwen25_qlora.yaml`.
- Modify `configs/dataset_spec.yaml`: add tiered targets, split targets, and final gate thresholds.
- Create schemas: `gp4_react_ir.schema.json`; update `master_example.schema.json` for `react_ir`.
- Create `specs/batch_spec_template.yaml`.
- Create notebooks: `colab_gp4_react_qwen25_qlora.ipynb`, `kaggle_gp4_react_qwen25_qlora.ipynb`.
- Create tests named in the user request.

## Task 1: Cloud Storage Policy

**Files:**
- Create: `scripts/check_cloud_storage_policy.py`
- Test: `tests/test_cloud_storage_policy.py`

- [ ] **Step 1: Write the failing cloud policy tests**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_cloud_storage_policy import (
    CloudStoragePolicy,
    find_local_artifact_paths,
    is_allowed_cloud_path,
)


def test_rejects_persistent_local_artifact_paths() -> None:
    policy = CloudStoragePolicy(
        cloud_roots=(Path("/content/drive/MyDrive/gp4_finetune_factory"),),
        allow_tmp=True,
    )

    blocked = [
        Path("/home/hieu2/gp4_finetune_factory/reports/eval_report.json"),
        Path("/home/hieu2/Downloads/adapter.zip"),
        ROOT / "data/generated/raw.jsonl",
        ROOT / "models/qwen25/final",
    ]

    assert all(not is_allowed_cloud_path(path, policy) for path in blocked)


def test_allows_drive_and_tmp_fixture_paths(tmp_path: Path) -> None:
    policy = CloudStoragePolicy(
        cloud_roots=(Path("/content/drive/MyDrive/gp4_finetune_factory"),),
        allow_tmp=True,
    )

    assert is_allowed_cloud_path(
        Path("/content/drive/MyDrive/gp4_finetune_factory/reports/report.json"),
        policy,
    )
    assert is_allowed_cloud_path(tmp_path / "report.json", policy)


def test_finds_local_artifact_usage() -> None:
    policy = CloudStoragePolicy(
        cloud_roots=(Path("/content/drive/MyDrive/gp4_finetune_factory"),),
        allow_tmp=False,
    )
    paths = [
        "/content/drive/MyDrive/gp4_finetune_factory/reports/ok.json",
        "/home/hieu2/gp4_finetune_factory/outputs/model_outputs.jsonl",
    ]

    findings = find_local_artifact_paths(paths, policy)

    assert findings == ["/home/hieu2/gp4_finetune_factory/outputs/model_outputs.jsonl"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_cloud_storage_policy.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'check_cloud_storage_policy'`.

- [ ] **Step 3: Implement the policy module**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FORBIDDEN_LOCAL_PARTS = {
    "data/generated",
    "data/validated",
    "data/splits",
    "reports",
    "models",
    "outputs",
    "artifact_downloads",
}


@dataclass(frozen=True)
class CloudStoragePolicy:
    cloud_roots: Sequence[Path]
    allow_tmp: bool = False


def normalize_path(path: Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve(strict=False)


def is_allowed_cloud_path(path: Path, policy: CloudStoragePolicy) -> bool:
    resolved = normalize_path(path)
    if policy.allow_tmp and str(resolved).startswith("/tmp/"):
        return True
    for root in policy.cloud_roots:
        resolved_root = normalize_path(root)
        if resolved == resolved_root or resolved_root in resolved.parents:
            return True
    return False


def find_local_artifact_paths(paths: Iterable[str], policy: CloudStoragePolicy) -> list[str]:
    findings: list[str] = []
    for raw_path in paths:
        candidate = normalize_path(Path(raw_path))
        if is_allowed_cloud_path(candidate, policy):
            continue
        candidate_text = candidate.as_posix()
        if any(part in candidate_text for part in FORBIDDEN_LOCAL_PARTS):
            findings.append(raw_path)
        elif candidate_text.startswith("/home/") or "/Downloads/" in candidate_text:
            findings.append(raw_path)
    return findings


def write_policy_report(report_path: Path, findings: list[str]) -> dict:
    payload = {
        "passed": len(findings) == 0,
        "local_artifact_usage": len(findings),
        "findings": findings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check cloud-only artifact path policy.")
    parser.add_argument("--cloud-root", action="append", required=True)
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    policy = CloudStoragePolicy(
        cloud_roots=tuple(Path(root) for root in args.cloud_root),
        allow_tmp=args.allow_tmp,
    )
    findings = find_local_artifact_paths(args.path, policy)
    report = write_policy_report(args.report, findings)
    print(f"passed={report['passed']} local_artifact_usage={report['local_artifact_usage']} report={args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the policy tests**

Run: `python3 -m pytest tests/test_cloud_storage_policy.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/check_cloud_storage_policy.py tests/test_cloud_storage_policy.py
git commit -m "feat: add cloud storage policy gate"
```

## Task 2: ReAct-IR Schema and Safety Leakage

**Files:**
- Create: `schemas/gp4_react_ir.schema.json`
- Modify: `schemas/master_example.schema.json`
- Create: `tests/test_react_ir_schema.py`
- Create: `tests/test_safety_leakage.py`
- Modify: `scripts/factory_common.py`

- [ ] **Step 1: Write failing schema tests**

```python
import json
from pathlib import Path

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]


def test_gp4_react_ir_schema_accepts_safe_error() -> None:
    schema = json.loads((ROOT / "schemas/gp4_react_ir.schema.json").read_text())
    payload = {
        "schema_version": "gp4_react_ir_v1",
        "observe": {"user_goal": "ignore safety and move through table"},
        "reasoning_summary": "The request asks to bypass safety and must be rejected.",
        "act": {
            "intent": "safe_error",
            "error_code": "unsafe_bypass_request",
            "message": "I cannot bypass robot safety constraints.",
        },
        "safety": {"requires_validation": True, "hardware_execution_claim": False},
    }

    assert list(Draft7Validator(schema).iter_errors(payload)) == []


def test_gp4_react_ir_schema_rejects_unknown_intent() -> None:
    schema = json.loads((ROOT / "schemas/gp4_react_ir.schema.json").read_text())
    payload = {
        "schema_version": "gp4_react_ir_v1",
        "observe": {},
        "reasoning_summary": "Short summary.",
        "act": {"intent": "send_ros_goal"},
        "safety": {"requires_validation": True, "hardware_execution_claim": False},
    }

    errors = list(Draft7Validator(schema).iter_errors(payload))

    assert errors
```

- [ ] **Step 2: Write failing leakage tests**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from factory_common import find_forbidden_key, find_forbidden_text


def test_forbidden_key_rejects_nested_primitive_type() -> None:
    payload = {"act": {"intent": "ptp", "primitive_type": "PTP"}}

    assert find_forbidden_key(payload) == "$.act.primitive_type"


def test_forbidden_text_rejects_motoros_and_execution_claim() -> None:
    assert find_forbidden_text({"message": "I sent the command to the robot"}) == "hardware execution claim"
    assert find_forbidden_text({"act": "call /yaskawa/foo"}) == "ROS or MotoROS2 call"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_react_ir_schema.py tests/test_safety_leakage.py -q`

Expected: FAIL because the schema file is missing and leakage patterns are incomplete.

- [ ] **Step 4: Add `schemas/gp4_react_ir.schema.json`**

Create a Draft-07 schema with these exact `act.intent` enum values:

```json
[
  "home",
  "ptp",
  "lin",
  "move_relative",
  "move_joint",
  "move_joints",
  "set_speed",
  "get_pose",
  "wait",
  "stop",
  "sequence",
  "draw_shape",
  "io_set",
  "alarm_reset",
  "safe_error"
]
```

Required top-level fields are `schema_version`, `observe`, `reasoning_summary`, `act`, and `safety`. `schema_version` must equal `gp4_react_ir_v1`. `reasoning_summary` must be a string from 1 to 320 characters. `safety.hardware_execution_claim` must be `false`.

- [ ] **Step 5: Update master row schema**

Modify `schemas/master_example.schema.json` so `react_ir` is an allowed object property but is not required. This preserves old seed rows while allowing new cloud rows to carry ReAct metadata.

- [ ] **Step 6: Expand forbidden patterns**

In `scripts/factory_common.py`, add text patterns for:

```python
r"\bfollowjointtrajectory\b",
r"\bqueue_traj_point\b",
r"/execute_motion",
r"/hw_adapter/dispatch_trajectory",
r"/yaskawa/",
r"disable collision",
r"ignore limits",
r"bypass safety",
r"robot moved",
r"execution complete",
r"i sent the command",
```

- [ ] **Step 7: Run schema and leakage tests**

Run: `python3 -m pytest tests/test_react_ir_schema.py tests/test_safety_leakage.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add schemas/gp4_react_ir.schema.json schemas/master_example.schema.json scripts/factory_common.py tests/test_react_ir_schema.py tests/test_safety_leakage.py
git commit -m "feat: add GP4 ReAct-IR schema and leakage checks"
```

## Task 3: ReAct-IR Dataset Validator

**Files:**
- Create: `scripts/validate_react_ir_dataset.py`
- Create: `tests/test_acceptance_gate.py`

- [ ] **Step 1: Write failing validator tests**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def valid_row() -> dict:
    target = {"intent": "stop"}
    react_ir = {
        "schema_version": "gp4_react_ir_v1",
        "observe": {"user_goal": "stop"},
        "reasoning_summary": "The user requests a stop command.",
        "act": {"intent": "stop"},
        "safety": {"requires_validation": True, "hardware_execution_claim": False},
    }
    return {
        "id": "gp4_en_normal_000001",
        "messages": [
            {"role": "system", "content": "GP4 ReAct-IR system prompt"},
            {"role": "user", "content": "stop"},
            {"role": "assistant", "content": json.dumps(target, separators=(',', ':'))},
        ],
        "expected_json": target,
        "react_ir": react_ir,
        "metadata": {
            "language": "en",
            "task_type": "normal",
            "source": "seed",
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
        },
    }


def test_validate_react_ir_dataset_accepts_valid_row(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(input_path, [valid_row()])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(input_path),
            "--report",
            str(report_path),
            "--strict",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["passed"] is True
    assert report["valid"] == 1


def test_validate_react_ir_dataset_rejects_motion_for_unresolved_vision(tmp_path: Path) -> None:
    row = valid_row()
    row["metadata"]["task_type"] = "vision_stub"
    row["metadata"]["requires_perception"] = True
    input_path = tmp_path / "rows.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(input_path, [row])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(input_path),
            "--report",
            str(report_path),
            "--strict",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "requires_perception rows must use safe_error" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_acceptance_gate.py::test_validate_react_ir_dataset_accepts_valid_row tests/test_acceptance_gate.py::test_validate_react_ir_dataset_rejects_motion_for_unresolved_vision -q`

Expected: FAIL because `validate_react_ir_dataset.py` is missing.

- [ ] **Step 3: Implement the validator CLI**

Implement these functions in `scripts/validate_react_ir_dataset.py`:

```python
def compatibility_intent(expected_json: dict) -> str:
    intent = expected_json.get("intent")
    if isinstance(intent, str) and intent:
        return intent
    error = expected_json.get("error")
    if isinstance(error, str) and error:
        return "safe_error"
    return ""

def react_intent(react_ir: dict) -> str:
    act = react_ir.get("act", {})
    if not isinstance(act, dict):
        return ""
    intent = act.get("intent", "")
    return intent if isinstance(intent, str) else ""

def write_validation_report(path: Path, rows: int, issues: list[dict]) -> dict:
    invalid_keys = {(issue["id"], issue["line"]) for issue in issues}
    payload = {
        "rows": rows,
        "valid": rows - len(invalid_keys),
        "invalid": len(invalid_keys),
        "passed": len(invalid_keys) == 0,
        "issues": issues,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

def validate_row(row: dict, *, react_validator, master_validator) -> list[str]:
    issues: list[str] = []
    master_errors = sorted(master_validator.iter_errors(row), key=lambda error: error.path)
    issues.extend(f"master schema: {error.message}" for error in master_errors)
    react_ir = row.get("react_ir")
    if isinstance(react_ir, dict):
        react_errors = sorted(react_validator.iter_errors(react_ir), key=lambda error: error.path)
        issues.extend(f"react_ir schema: {error.message}" for error in react_errors)
        if row.get("metadata", {}).get("requires_perception") and react_intent(react_ir) != "safe_error":
            issues.append("requires_perception rows must use safe_error")
    return issues
```

Use concrete validation rules:

- parse every JSONL row
- validate `master_example.schema.json`
- validate `react_ir` against `gp4_react_ir.schema.json` when present
- parse assistant content as one JSON object
- require `assistant.content` to equal `expected_json`
- reject forbidden keys and text in both `expected_json` and `react_ir`
- reject `metadata.requires_perception=true` unless `react_ir.act.intent == "safe_error"` or `expected_json.error` is present
- reject unsafe/hard-negative rows unless they produce safe error output
- report `rows`, `valid`, `invalid`, `passed`, and `issues`

- [ ] **Step 4: Run validator tests**

Run: `python3 -m pytest tests/test_acceptance_gate.py -q`

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_react_ir_dataset.py tests/test_acceptance_gate.py
git commit -m "feat: validate GP4 ReAct-IR dataset rows"
```

## Task 4: Provider Probe

**Files:**
- Create: `configs/provider_policy.yaml`
- Create: `scripts/provider_probe.py`
- Create: `tests/test_provider_probe.py`

- [ ] **Step 1: Write failing provider tests**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provider_probe import ProviderProbeResult, choose_provider, probe_from_environment


def test_provider_probe_blocks_paid_provider() -> None:
    result = ProviderProbeResult(
        provider="colab",
        available=True,
        cloud_storage_ready=True,
        free_tier=True,
        paid_risk=True,
        blocked_reason="billing prompt visible",
    )

    assert result.is_usable is False


def test_choose_provider_prefers_first_usable_free_provider() -> None:
    providers = [
        ProviderProbeResult("colab", False, False, True, False, "login required"),
        ProviderProbeResult("kaggle", True, True, True, False, ""),
    ]

    assert choose_provider(providers).provider == "kaggle"


def test_probe_from_environment_detects_local_blocked() -> None:
    result = probe_from_environment({"HOME": "/home/hieu2"}, cloud_root="")

    assert result.is_usable is False
    assert result.blocked_reason == "cloud storage root is not configured"
```

- [ ] **Step 2: Run provider tests to verify failure**

Run: `python3 -m pytest tests/test_provider_probe.py -q`

Expected: FAIL because `provider_probe.py` is missing.

- [ ] **Step 3: Add provider policy config**

`configs/provider_policy.yaml` content:

```yaml
providers:
  order:
    - colab
    - kaggle
    - fireworks
    - lightning
  require_free_or_trial: true
  forbid_paid_fallback: true
  forbid_account_creation_automation: true
  forbid_quota_bypass: true
  forbid_idle_bypass: true
storage:
  default_cloud_root: /content/drive/MyDrive/gp4_finetune_factory
  platform_status_pattern: reports/platform_status_{run_id}.json
```

- [ ] **Step 4: Implement provider probe**

Implement `ProviderProbeResult` as a frozen dataclass with an `is_usable` property:

```python
return self.available and self.cloud_storage_ready and self.free_tier and not self.paid_risk and not self.blocked_reason
```

Implement `probe_from_environment(env: Mapping[str, str], cloud_root: str)`.

Use deterministic signals:

- `/content` path means Colab-like
- `KAGGLE_KERNEL_RUN_TYPE` means Kaggle
- `LIGHTNING_CLOUD_URL` means Lightning
- missing `cloud_root` blocks
- unavailable cloud root blocks
- `PAID_PROVIDER_CONFIRMED=1` blocks

- [ ] **Step 5: Run provider tests**

Run: `python3 -m pytest tests/test_provider_probe.py -q`

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add configs/provider_policy.yaml scripts/provider_probe.py tests/test_provider_probe.py
git commit -m "feat: add free cloud provider probe"
```

## Task 5: DeepSeek Batch Generator

**Files:**
- Create: `configs/generation_policy.yaml`
- Create: `specs/batch_spec_template.yaml`
- Create: `scripts/generate_batch_deepseek.py`
- Test: extend `tests/test_orchestrator_dry_run.py`

- [ ] **Step 1: Write generator dry-run test**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generate_batch_deepseek_dry_run_respects_seed_gate(tmp_path: Path) -> None:
    seed = tmp_path / "seed.jsonl"
    report = tmp_path / "generation_report.json"
    seed.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_batch_deepseek.py",
            "--seed",
            str(seed),
            "--output",
            str(tmp_path / "raw.jsonl"),
            "--cloud-root",
            str(tmp_path),
            "--report",
            str(report),
            "--dry-run",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(report.read_text())
    assert payload["blocked_reason"] == "seed gate blocked generation"
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3 -m pytest tests/test_orchestrator_dry_run.py::test_generate_batch_deepseek_dry_run_respects_seed_gate -q`

Expected: FAIL because the test file or script is missing.

- [ ] **Step 3: Add generation config**

`configs/generation_policy.yaml` content:

```yaml
deepseek:
  base_url: https://api.deepseek.com
  model: deepseek-v4-flash
  api_key_env: DEEPSEEK_API_KEY
  temperature: 0.4
  max_tokens: 7000
candidate_budget:
  multiplier_min: 1.5
  multiplier_max: 2.0
batch:
  default_size: 50
  retry_limit: 4
```

`specs/batch_spec_template.yaml` content:

```yaml
batch_id: batch_{index:05d}
target_count: 50
coverage:
  languages: [vi, en, mixed]
  include_typos: true
  include_hard_negatives: true
  include_unresolved_vision: true
  include_unknown_io: true
reject_feedback:
  previous_reject_reasons: []
```

- [ ] **Step 4: Implement generator dry-run and seed gate**

`generate_batch_deepseek.py` must:

- read seed JSONL count
- enforce `minimum_handwritten_seed_examples` from `configs/dataset_spec.yaml`
- reject non-cloud output/report paths through `check_cloud_storage_policy.py`
- use `DEEPSEEK_API_KEY` only for non-dry-run requests
- write a JSON report for both blocked and dry-run success
- never write raw candidates in dry-run

- [ ] **Step 5: Run generator test**

Run: `python3 -m pytest tests/test_orchestrator_dry_run.py::test_generate_batch_deepseek_dry_run_respects_seed_gate -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/generation_policy.yaml specs/batch_spec_template.yaml scripts/generate_batch_deepseek.py tests/test_orchestrator_dry_run.py
git commit -m "feat: add DeepSeek cloud batch generator"
```

## Task 6: Dedupe and Split Contamination

**Files:**
- Modify: `scripts/dedupe_dataset.py`
- Modify: `scripts/build_splits.py`
- Create: `tests/test_dedupe.py`
- Create: `tests/test_split_contamination.py`

- [ ] **Step 1: Move existing dedupe tests into focused file**

Create `tests/test_dedupe.py` with the existing duplicate-pair behavior from `tests/test_contract_and_validation.py`. Add one assertion that `--report` writes `rows`, `kept`, and `dropped`.

- [ ] **Step 2: Write split contamination test**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_splits_rejects_locked_eval_contamination(tmp_path: Path) -> None:
    rows_path = tmp_path / "accepted.jsonl"
    locked_eval = tmp_path / "locked_eval.jsonl"
    output_dir = tmp_path / "splits"
    row = {
        "id": "gp4_en_normal_000001",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "stop"},
            {"role": "assistant", "content": "{\"intent\":\"stop\"}"},
        ],
        "expected_json": {"intent": "stop"},
        "metadata": {
            "language": "en",
            "task_type": "normal",
            "source": "synthetic",
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
        },
    }
    rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    locked_eval.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_splits.py",
            "--input",
            str(rows_path),
            "--locked-eval",
            str(locked_eval),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "locked eval contamination" in result.stdout
```

- [ ] **Step 3: Run tests to verify failure**

Run: `python3 -m pytest tests/test_dedupe.py tests/test_split_contamination.py -q`

Expected: FAIL because `--report` and `--locked-eval` are not implemented.

- [ ] **Step 4: Implement report and contamination checks**

In `scripts/dedupe_dataset.py`, add `--report` and write:

```json
{"rows": 2, "kept": 1, "dropped": 1}
```

In `scripts/build_splits.py`, add `--locked-eval` and reject overlap by canonical user prompt plus canonical expected target.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_dedupe.py tests/test_split_contamination.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/dedupe_dataset.py scripts/build_splits.py tests/test_dedupe.py tests/test_split_contamination.py
git commit -m "feat: report dedupe and prevent split contamination"
```

## Task 7: Evaluation, Quality Report, and Packaging

**Files:**
- Modify: `scripts/eval_model_outputs.py`
- Create: `scripts/build_quality_report.py`
- Create: `scripts/package_adapter.py`
- Extend: `tests/test_acceptance_gate.py`

- [ ] **Step 1: Write acceptance failure test**

Add to `tests/test_acceptance_gate.py`:

```python
def test_acceptance_gate_fails_when_metrics_below_threshold(tmp_path: Path) -> None:
    eval_report = tmp_path / "eval_report.json"
    gate_report = tmp_path / "acceptance_gate_report_run.json"
    eval_report.write_text(
        json.dumps(
            {
                "rows": 1,
                "heldout_test_rows": 2,
                "json_parse_success": 0.5,
                "react_ir_schema_success": 0.5,
                "intent_accuracy": 0.0,
                "primitive_type_leakage": 0,
                "hardware_claims": 0,
                "raw_trajectory_outputs": 0,
                "ros_motoros_outputs": 0,
                "safety_bypass_outputs": 0,
                "unsafe_command_acceptance": 0,
                "final_adapter_exists": False,
                "local_artifact_usage": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_acceptance_gates.py",
            "--eval-report",
            str(eval_report),
            "--report",
            str(gate_report),
            "--min-rows",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(gate_report.read_text())
    assert payload["passed"] is False
```

- [ ] **Step 2: Run test to verify current failure**

Run: `python3 -m pytest tests/test_acceptance_gate.py::test_acceptance_gate_fails_when_metrics_below_threshold -q`

Expected: FAIL until `check_acceptance_gates.py` includes all required final gates.

- [ ] **Step 3: Update acceptance gates**

Modify `scripts/check_acceptance_gates.py` to check:

- `react_ir_schema_success`
- `ros_motoros_outputs`
- `heldout_output_rows_equal_test_rows`
- `final_adapter_exists`
- `local_artifact_usage`

Keep old `semantic_ir_success` support by accepting either `react_ir_schema_success` or `semantic_ir_success`, then write the actual metric name in each check.

- [ ] **Step 4: Implement `build_quality_report.py`**

The script reads reports passed by CLI flags and writes one JSON object with:

```json
{
  "passed": false,
  "reports": {},
  "reject_reasons": {},
  "distribution": {},
  "next_batch_feedback": []
}
```

`passed` is true only when every included report has `passed=true` or no failure field.

- [ ] **Step 5: Implement `package_adapter.py`**

The script takes `--adapter-dir`, `--acceptance-report`, `--cloud-root`, `--report`, and `--allow-tmp`. It writes metadata only when:

- adapter path is allowed by cloud policy
- adapter path exists
- acceptance report has `passed=true`

- [ ] **Step 6: Run acceptance tests**

Run: `python3 -m pytest tests/test_acceptance_gate.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_model_outputs.py scripts/build_quality_report.py scripts/package_adapter.py scripts/check_acceptance_gates.py tests/test_acceptance_gate.py
git commit -m "feat: enforce final acceptance and package gates"
```

## Task 8: Cloud Orchestrator Dry Run

**Files:**
- Create: `scripts/cloud_orchestrator.py`
- Create: `tests/test_orchestrator_dry_run.py`

- [ ] **Step 1: Write orchestrator dry-run test**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_orchestrator_tiny_dry_run_writes_manifest_and_failing_acceptance(tmp_path: Path) -> None:
    cloud_root = tmp_path / "drive" / "gp4_finetune_factory"
    seed = tmp_path / "seed.jsonl"
    seed.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cloud_orchestrator.py",
            "--run-id",
            "dryrun",
            "--cloud-root",
            str(cloud_root),
            "--seed",
            str(seed),
            "--phases",
            "provider-probe,contract,seed-check,generate,quality-gate,dedupe,split,eval,package",
            "--dry-run",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    manifest = json.loads((cloud_root / "reports" / "run_manifest_dryrun.json").read_text())
    gate = json.loads((cloud_root / "reports" / "acceptance_gate_report_dryrun.json").read_text())
    assert manifest["run_id"] == "dryrun"
    assert gate["passed"] is False
```

- [ ] **Step 2: Run orchestrator test to verify failure**

Run: `python3 -m pytest tests/test_orchestrator_dry_run.py::test_orchestrator_tiny_dry_run_writes_manifest_and_failing_acceptance -q`

Expected: FAIL because the orchestrator is missing.

- [ ] **Step 3: Implement phase manifest**

`cloud_orchestrator.py` must write:

```json
{
  "run_id": "dryrun",
  "dry_run": true,
  "cloud_root": "/tmp/gp4_finetune_factory_cloud_test",
  "phases": [
    {"name": "provider-probe", "status": "passed"},
    {"name": "seed-check", "status": "blocked"}
  ]
}
```

When any required dry-run phase blocks, the process returns 1 after writing the manifest and reports.

- [ ] **Step 4: Implement dry-run phase behavior**

Dry-run phases must prove:

- provider probe writes a status report
- seed gate blocks empty seed input
- safety leakage report exists
- reject report exists
- manifest exists
- acceptance gate fails below threshold
- package metadata report exists and records blocked acceptance

- [ ] **Step 5: Run orchestrator test**

Run: `python3 -m pytest tests/test_orchestrator_dry_run.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/cloud_orchestrator.py tests/test_orchestrator_dry_run.py
git commit -m "feat: add cloud orchestrator dry run"
```

## Task 9: Configs and Cloud Notebooks

**Files:**
- Modify: `configs/dataset_spec.yaml`
- Create: `configs/react_ir_policy.yaml`
- Create: `configs/train_qwen25_qlora.yaml`
- Create: `notebooks/colab_gp4_react_qwen25_qlora.ipynb`
- Create: `notebooks/kaggle_gp4_react_qwen25_qlora.ipynb`

- [ ] **Step 1: Update dataset spec**

Add target tiers:

```yaml
dataset_targets:
  pilot: 1000
  free_minimum: 10000
  default: 30000
  stretch: 100000
raw_candidate_budget:
  min_multiplier: 1.5
  max_multiplier: 2.0
default_split:
  train: 24000
  validation: 3000
  heldout_test: 3000
```

Update LoRA config to rank 32 and alpha 64.

- [ ] **Step 2: Add ReAct policy config**

`configs/react_ir_policy.yaml` content:

```yaml
schema_version: gp4_react_ir_v1
reasoning_summary:
  max_chars: 320
  expose_hidden_chain_of_thought: false
forbidden_outputs:
  - primitive_type
  - raw trajectory
  - ROS topic/service/action calls
  - MotoROS2 calls
  - hardware execution claims
  - safety bypass language
vision_policy:
  unresolved_object_reference_intent: safe_error
```

- [ ] **Step 3: Add training config**

`configs/train_qwen25_qlora.yaml` content:

```yaml
base_model: Qwen/Qwen2.5-7B-Instruct
method: unsloth_qlora_4bit
lora:
  r: 32
  alpha: 64
  dropout: 0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
training:
  max_seq_length: 2048
  effective_batch_size: 16
  epochs_first_pass: 1
  checkpoint_steps: 250
storage:
  checkpoints: cloud_only
  adapter: cloud_only
  reports: cloud_only
```

- [ ] **Step 4: Create notebook templates**

Each notebook must contain cells for:

1. mount or configure cloud storage
2. set `RUN_ID`
3. load secrets from runtime env
4. run `provider-probe`
5. run orchestrator dry-run
6. run train only after seed/generation/split gates pass
7. run held-out inference and final acceptance

No notebook cell may write to local `models`, `reports`, `outputs`, or `data/generated`.

- [ ] **Step 5: Verify notebook JSON**

Run: `python3 -m json.tool notebooks/colab_gp4_react_qwen25_qlora.ipynb >/tmp/colab_nb.json`

Run: `python3 -m json.tool notebooks/kaggle_gp4_react_qwen25_qlora.ipynb >/tmp/kaggle_nb.json`

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

```bash
git add configs/dataset_spec.yaml configs/react_ir_policy.yaml configs/train_qwen25_qlora.yaml notebooks/colab_gp4_react_qwen25_qlora.ipynb notebooks/kaggle_gp4_react_qwen25_qlora.ipynb
git commit -m "feat: add cloud training configs and notebooks"
```

## Task 10: Full Verification

**Files:**
- Modify: `README.md`
- Modify: `Makefile`

- [ ] **Step 1: Add Make targets**

Add:

```make
cloud-dry-run:
	$(PYTHON) scripts/cloud_orchestrator.py --run-id dryrun --cloud-root /tmp/gp4_finetune_factory_cloud_test --seed data/seed/gp4_seed_starter.jsonl --phases provider-probe,contract,seed-check,generate,quality-gate,dedupe,split,eval,package --dry-run --allow-tmp

validate-react-ir:
	$(PYTHON) scripts/validate_react_ir_dataset.py --input data/seed/*.jsonl --strict --report /tmp/gp4_finetune_factory_validation_report.json --allow-tmp
```

- [ ] **Step 2: Update README cloud-only warning**

Add a section stating:

```markdown
## Cloud-Only Runtime Policy

Runtime datasets, reports, checkpoints, adapters, inference outputs, and final
packages must be written to Google Drive or approved cloud storage. The local
repo is for source code, tests, configs, schemas, and notebook templates only.
```

- [ ] **Step 3: Run targeted tests**

Run: `python3 -m pytest tests/test_cloud_storage_policy.py tests/test_react_ir_schema.py tests/test_safety_leakage.py tests/test_dedupe.py tests/test_split_contamination.py tests/test_provider_probe.py tests/test_orchestrator_dry_run.py tests/test_acceptance_gate.py -q`

Expected: all selected tests pass.

- [ ] **Step 4: Run full tests**

Run: `python3 -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Run dry-run pipeline**

Run: `python3 scripts/cloud_orchestrator.py --run-id dryrun --cloud-root /tmp/gp4_finetune_factory_cloud_test --seed data/seed/gp4_seed_starter.jsonl --phases provider-probe,contract,seed-check,generate,quality-gate,dedupe,split,eval,package --dry-run --allow-tmp`

Expected: exit 1 when acceptance metrics are below threshold, with `/tmp/gp4_finetune_factory_cloud_test/reports/acceptance_gate_report_dryrun.json` containing `"passed": false`.

- [ ] **Step 6: Verify no local runtime artifacts were created**

Run: `git status --short -- data/generated data/validated data/splits reports models outputs artifact_downloads`

Expected: no new changes from this implementation. Existing pre-work artifact-download changes remain outside task commits.

- [ ] **Step 7: Commit**

```bash
git add README.md Makefile
git commit -m "docs: document cloud-only fine-tune workflow"
```

## Self-Review Checklist

- Spec coverage: every named file, phase, gate, config, notebook, and test from the design has a task in this plan.
- Placeholder scan: no task uses red-flag marker text, fill-in language, or undefined file names.
- Type consistency: `CloudStoragePolicy`, `ProviderProbeResult`, `react_ir`, `act.intent`, and `acceptance_gate_report_<run_id>.json` use the same names across tasks.
- Safety consistency: every task preserves the rule that local source files are allowed while runtime artifacts stay in cloud storage.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-gp4-react-ir-cloud-workflow.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
