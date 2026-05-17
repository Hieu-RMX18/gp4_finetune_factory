#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cloud_runtime import (
    CloudPathError,
    copy_source_plan_to_cloud,
    phase_report_path,
    report_passed,
    require_cloud_root,
    sha256_file,
    validate_cloud_run_paths,
    write_phase_report,
)
from check_acceptance_gates import evaluate_gates
from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from factory_common import read_json, read_jsonl, read_yaml, write_json, write_jsonl
from package_adapter import adapter_artifact_exists
from provider_probe import ProviderProbeResult, write_platform_status
from typo_noise_policy import write_locked_typo_eval

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"

PREVIOUS_GATE_REPORTS = {
    "generate": "seed-check",
    "quality-gate": "generate",
    "dedupe": "quality-gate",
    "split": "dedupe",
    "train": "split",
    "infer": "train",
    "eval": "infer",
    "package": "eval",
    "generate-smoke": "seed-check",
    "generate-1k": "generate-smoke",
    "quality-gate-1k": "generate-1k",
    "generate-30k": "quality-gate-1k",
    "quality-gate-30k": "generate-30k",
    "generate-50k": "quality-gate-30k",
    "quality-gate-50k": "generate-50k",
    "generate-100k": "quality-gate-50k",
    "quality-gate-100k": "generate-100k",
}

GENERATION_COUNTS = {
    "generate-smoke": 10,
    "generate-1k": 1000,
    "generate-30k": 30000,
    "generate-50k": 50000,
    "generate-100k": 100000,
    "generate": 50,
}
CLOUD_TRAIN_RATIO = 0.998
CLOUD_VAL_RATIO = 0.001


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GP4 cloud-only workflow phases.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=args.dry_run,
            inputs=[args.seed],
            outputs=[args.cloud_root / "reports" / f"run_manifest_{args.run_id}.json"],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"blocked_reason={exc}")
        return 1

    args.cloud_root.mkdir(parents=True, exist_ok=True)
    reports_dir = args.cloud_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    policy = CloudStoragePolicy(cloud_roots=(args.cloud_root,), allow_tmp=args.allow_tmp)
    manifest_path = reports_dir / f"run_manifest_{args.run_id}.json"
    if not is_allowed_cloud_path(manifest_path, policy):
        print(f"manifest path is not allowed by cloud storage policy: {manifest_path}")
        return 1

    context = {
        "run_id": args.run_id,
        "cloud_root": args.cloud_root,
        "reports_dir": reports_dir,
        "seed": args.seed,
        "policy": policy,
        "provider": args.provider,
        "source_plan": args.source_plan,
        "dry_run": args.dry_run,
        "allow_tmp": args.allow_tmp,
    }
    requested_phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    phase_results: list[dict[str, Any]] = []
    for phase in requested_phases:
        if args.dry_run:
            phase_results.append(_run_dry_phase(phase, context))
        else:
            phase_result = _run_cloud_phase(phase, context)
            phase_results.append(phase_result)
            if phase_result["status"] in {"failed", "blocked"}:
                break

    manifest = {
        "run_id": args.run_id,
        "dry_run": args.dry_run,
        "cloud_root": str(args.cloud_root),
        "phases": phase_results,
    }
    write_json(manifest_path, manifest)
    failed = any(result["status"] in {"failed", "blocked"} for result in phase_results)
    print(f"run_id={args.run_id} dry_run={args.dry_run} failed={failed} manifest={manifest_path}")
    return 1 if failed else 0


def _run_cloud_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = require_cloud_root(
        context["cloud_root"],
        allow_tmp=bool(context["allow_tmp"]),
    )
    run_id = str(context["run_id"])
    report_path = phase_report_path(cloud_root, run_id, phase)
    if report_passed(report_path):
        return {"name": phase, "status": "passed", "report": str(report_path), "resumed": True}

    previous_gate = PREVIOUS_GATE_REPORTS.get(phase)
    if previous_gate:
        previous_report = phase_report_path(cloud_root, run_id, previous_gate)
        if not report_passed(previous_report):
            return _block_cloud_phase(
                phase,
                context,
                f"previous gate report has not passed: {previous_gate}",
            )

    if phase in {"cloud-setup", "setup"}:
        return _run_cloud_setup(phase, context)
    if phase == "provider-probe":
        return _run_cloud_provider_probe(phase, context)
    if phase == "contract":
        return _run_contract_manifest(phase, context)
    if phase == "seed-check":
        return _run_cloud_seed_check(phase, context)
    if phase.startswith("generate"):
        return _run_generation_phase(phase, context)
    if phase.startswith("quality-gate"):
        return _run_quality_gate_phase(phase, context)
    if phase == "dedupe":
        return _run_script_phase(
            phase,
            context,
            [
                "scripts/dedupe_dataset.py",
                "--input",
                str(_preferred_generated_path(cloud_root)),
                "--output",
                str(cloud_root / "data/validated/accepted.jsonl"),
                "--report",
                str(report_path),
                "--cloud-root",
                str(cloud_root),
            ],
        )
    if phase == "split":
        locked_eval = cloud_root / "eval/locked_typo_eval.jsonl"
        write_locked_typo_eval(locked_eval)
        command = [
            "scripts/build_splits.py",
            "--input",
            str(cloud_root / "data/validated/accepted.jsonl"),
            "--output-dir",
            str(cloud_root / "data/splits"),
            "--report",
            str(report_path),
            "--cloud-root",
            str(cloud_root),
            "--locked-eval",
            str(locked_eval),
            "--train-ratio",
            str(CLOUD_TRAIN_RATIO),
            "--val-ratio",
            str(CLOUD_VAL_RATIO),
        ]
        return _run_script_phase(phase, context, command)
    if phase == "train":
        return _run_script_phase(
            phase,
            context,
            [
                "scripts/train_unsloth_qlora.py",
                "--train",
                str(cloud_root / "data/splits/train.jsonl"),
                "--val",
                str(cloud_root / "data/splits/val.jsonl"),
                "--output-dir",
                str(cloud_root / "models/qwen25_gp4_lora"),
                "--report",
                str(report_path),
                "--cloud-root",
                str(cloud_root),
            ],
        )
    if phase == "infer":
        heldout_input = _prepare_heldout_eval_input(cloud_root)
        return _run_script_phase(
            phase,
            context,
            [
                "scripts/run_adapter_inference.py",
                "--input",
                str(heldout_input),
                "--adapter-dir",
                str(cloud_root / "models/qwen25_gp4_lora"),
                "--output",
                str(cloud_root / "outputs/model_outputs.jsonl"),
                "--report",
                str(report_path),
                "--cloud-root",
                str(cloud_root),
            ],
        )
    if phase == "eval":
        return _run_eval_phase(phase, context)
    if phase == "package":
        return _run_script_phase(
            phase,
            context,
            [
                "scripts/package_adapter.py",
                "--adapter-dir",
                str(cloud_root / "models/qwen25_gp4_lora"),
                "--acceptance-report",
                str(cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json"),
                "--cloud-root",
                str(cloud_root),
                "--report",
                str(report_path),
            ],
        )

    return _block_cloud_phase(phase, context, "unknown phase")


def _run_cloud_setup(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    copied = copy_source_plan_to_cloud(
        source_plan=context.get("source_plan"),
        cloud_root=cloud_root,
        dry_run=False,
        allow_tmp=bool(context["allow_tmp"]),
    )
    manifest = _contract_manifest_payload(context, source_plan_sha=copied.sha256)
    manifest_path = cloud_root / "manifests/contract_manifest.json"
    write_json(manifest_path, manifest)
    report = {
        "passed": True,
        "source_plan": str(copied.path),
        "source_plan_sha256": copied.sha256,
        "contract_manifest": str(manifest_path),
    }
    report_path = write_phase_report(
        cloud_root=cloud_root,
        run_id=str(context["run_id"]),
        phase=phase,
        payload=report,
        allow_tmp=bool(context["allow_tmp"]),
    )
    return {"name": phase, "status": "passed", "report": str(report_path)}


def _run_cloud_provider_probe(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    from provider_probe import probe_from_environment

    cloud_root: Path = context["cloud_root"]
    result = probe_from_environment(
        os.environ,
        str(cloud_root),
        provider=context.get("provider"),
    )
    report_path = phase_report_path(cloud_root, str(context["run_id"]), phase)
    write_platform_status(report_path, result)
    status_report = cloud_root / "reports" / f"platform_status_{context['run_id']}.json"
    write_platform_status(status_report, result)
    return {
        "name": phase,
        "status": "passed" if result.is_usable else "blocked",
        "report": str(report_path),
    }


def _run_contract_manifest(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    manifest = _contract_manifest_payload(context, source_plan_sha=None)
    manifest_path = cloud_root / "manifests/contract_manifest.json"
    write_json(manifest_path, manifest)
    report = {"passed": True, "contract_manifest": str(manifest_path)}
    report_path = write_phase_report(
        cloud_root=cloud_root,
        run_id=str(context["run_id"]),
        phase=phase,
        payload=report,
        allow_tmp=bool(context["allow_tmp"]),
    )
    return {"name": phase, "status": "passed", "report": str(report_path)}


def _run_cloud_seed_check(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    seed_rows = len(read_jsonl(context["seed"]))
    minimum = int(read_yaml(DATASET_SPEC)["seed_gate"]["minimum_handwritten_seed_examples"])
    passed = seed_rows >= minimum
    report = {
        "passed": passed,
        "seed_rows": seed_rows,
        "minimum": minimum,
        "blocked_reason": "" if passed else "seed gate blocked generation",
    }
    report_path = write_phase_report(
        cloud_root=context["cloud_root"],
        run_id=str(context["run_id"]),
        phase=phase,
        payload=report,
        allow_tmp=bool(context["allow_tmp"]),
    )
    return {"name": phase, "status": "passed" if passed else "blocked", "report": str(report_path)}


def _run_generation_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    count = GENERATION_COUNTS.get(phase, 50)
    output = cloud_root / "data/generated" / f"raw_{phase}.jsonl"
    return _run_script_phase(
        phase,
        context,
        [
            "scripts/generate_batch_deepseek.py",
            "--seed",
            str(context["seed"]),
            "--output",
            str(output),
            "--cloud-root",
            str(cloud_root),
            "--report",
            str(phase_report_path(cloud_root, str(context["run_id"]), phase)),
            "--count",
            str(count),
        ],
    )


def _run_quality_gate_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    source_phase = phase.replace("quality-gate", "generate")
    input_path = cloud_root / "data/generated" / f"raw_{source_phase}.jsonl"
    return _run_script_phase(
        phase,
        context,
        [
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(input_path),
            "--strict",
            "--report",
            str(phase_report_path(cloud_root, str(context["run_id"]), phase)),
            "--cloud-root",
            str(cloud_root),
        ],
    )


def _run_eval_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    run_id = str(context["run_id"])
    eval_metrics = cloud_root / "reports" / f"eval_report_{run_id}.json"
    acceptance_report = cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json"
    infer_report_path = phase_report_path(cloud_root, run_id, "infer")
    infer_report = read_json(infer_report_path) if infer_report_path.exists() else {}
    heldout_rows_args = (
        ["--heldout-test-rows", str(infer_report["rows"])]
        if isinstance(infer_report.get("rows"), int)
        else []
    )
    eval_result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_model_outputs.py",
            "--input",
            str(cloud_root / "outputs/model_outputs.jsonl"),
            "--report",
            str(eval_metrics),
            "--cloud-root",
            str(cloud_root),
            *heldout_rows_args,
            *(
                ["--allow-tmp"]
                if context.get("allow_tmp")
                else []
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if eval_result.returncode != 0:
        report_path = write_phase_report(
            cloud_root=cloud_root,
            run_id=run_id,
            phase=phase,
            payload={
                "passed": False,
                "blocked_reason": "eval_model_outputs failed",
                "eval_report": str(eval_metrics),
                "stdout_tail": eval_result.stdout[-4000:],
                "stderr_tail": eval_result.stderr[-4000:],
            },
            allow_tmp=bool(context["allow_tmp"]),
        )
        return {"name": phase, "status": "blocked", "report": str(report_path)}

    metrics = read_json(eval_metrics)
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    metrics["final_adapter_exists"] = adapter_artifact_exists(adapter_dir)
    metrics.setdefault("local_artifact_usage", 0)
    write_json(eval_metrics, metrics)

    gate_result = subprocess.run(
        [
            sys.executable,
            "scripts/check_acceptance_gates.py",
            "--eval-report",
            str(eval_metrics),
            "--report",
            str(acceptance_report),
            "--cloud-root",
            str(cloud_root),
            "--min-rows",
            "1",
            *(
                ["--allow-tmp"]
                if context.get("allow_tmp")
                else []
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    report_path = write_phase_report(
        cloud_root=cloud_root,
        run_id=run_id,
        phase=phase,
        payload={
            "passed": gate_result.returncode == 0,
            "eval_report": str(eval_metrics),
            "acceptance_report": str(acceptance_report),
            "stdout_tail": gate_result.stdout[-4000:],
            "stderr_tail": gate_result.stderr[-4000:],
        },
        allow_tmp=bool(context["allow_tmp"]),
    )
    return {
        "name": phase,
        "status": "passed" if gate_result.returncode == 0 else "blocked",
        "report": str(report_path),
    }


def _run_script_phase(
    phase: str,
    context: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    run_id = str(context["run_id"])
    if context.get("allow_tmp"):
        command.append("--allow-tmp")
    result = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    report_path = phase_report_path(cloud_root, run_id, phase)
    if not report_path.exists():
        write_phase_report(
            cloud_root=cloud_root,
            run_id=run_id,
            phase=phase,
            payload={
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            },
            allow_tmp=bool(context["allow_tmp"]),
        )
    return {
        "name": phase,
        "status": "passed" if result.returncode == 0 else "blocked",
        "report": str(report_path),
    }


def _block_cloud_phase(
    phase: str,
    context: dict[str, Any],
    blocked_reason: str,
) -> dict[str, Any]:
    report_path = write_phase_report(
        cloud_root=context["cloud_root"],
        run_id=str(context["run_id"]),
        phase=phase,
        payload={"passed": False, "blocked_reason": blocked_reason},
        allow_tmp=bool(context["allow_tmp"]),
    )
    return {"name": phase, "status": "blocked", "report": str(report_path)}


def _preferred_generated_path(cloud_root: Path) -> Path:
    for phase in ("generate-50k", "generate-30k", "generate-1k", "generate"):
        candidate = cloud_root / "data/generated" / f"raw_{phase}.jsonl"
        if candidate.exists():
            return candidate
    return cloud_root / "data/generated/raw_generate.jsonl"

def _prepare_heldout_eval_input(cloud_root: Path) -> Path:
    test_rows = read_jsonl(cloud_root / "data/splits/test.jsonl")
    locked_eval_path = cloud_root / "eval/locked_typo_eval.jsonl"
    if not locked_eval_path.exists():
        write_locked_typo_eval(locked_eval_path)
    locked_rows = read_jsonl(locked_eval_path)
    heldout_path = cloud_root / "eval/heldout_with_locked_typo.jsonl"
    write_jsonl(heldout_path, [*test_rows, *locked_rows])
    return heldout_path


def _contract_manifest_payload(
    context: dict[str, Any],
    *,
    source_plan_sha: str | None,
) -> dict[str, Any]:
    hash_paths = [
        ROOT / "schemas/semantic_ir.schema.json",
        ROOT / "schemas/master_example.schema.json",
        ROOT / "schemas/gp4_react_ir.schema.json",
        ROOT / "configs/dataset_spec.yaml",
        Path(context["seed"]),
    ]
    hashes = {
        str(path): sha256_file(path)
        for path in hash_paths
        if path.exists()
    }
    if source_plan_sha:
        hashes["source_plan.md"] = source_plan_sha
    return {
        "passed": True,
        "run_id": str(context["run_id"]),
        "hashes": hashes,
        "source_contract": "local source snapshot",
    }


def _run_dry_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    reports_dir: Path = context["reports_dir"]
    cloud_root: Path = context["cloud_root"]
    run_id = str(context["run_id"])
    if phase in {"cloud-setup", "setup"}:
        report = phase_report_path(cloud_root, run_id, phase)
        write_json(report, {"passed": True, "dry_run": True})
        return {"name": phase, "status": "passed", "report": str(report)}

    if phase == "provider-probe":
        result = ProviderProbeResult("dry-run", True, True, True, False, "")
        report = reports_dir / f"platform_status_{run_id}.json"
        write_platform_status(report, result)
        return {"name": phase, "status": "passed", "report": str(report)}

    if phase == "contract":
        report = reports_dir / f"contract_report_{run_id}.json"
        write_json(report, {"passed": True, "dry_run": True})
        return {"name": phase, "status": "passed", "report": str(report)}

    if phase == "seed-check":
        seed_rows = len(read_jsonl(context["seed"]))
        minimum = int(read_yaml(DATASET_SPEC)["seed_gate"]["minimum_handwritten_seed_examples"])
        report = reports_dir / f"seed_report_{run_id}.json"
        passed = seed_rows >= minimum
        write_json(
            report,
            {
                "passed": passed,
                "seed_rows": seed_rows,
                "minimum": minimum,
                "blocked_reason": "" if passed else "seed gate blocked generation",
            },
        )
        context["seed_gate_passed"] = passed
        return {"name": phase, "status": "passed" if passed else "blocked", "report": str(report)}

    if phase == "generate":
        report = reports_dir / f"generation_report_{run_id}.json"
        seed_gate_passed = bool(context.get("seed_gate_passed", False))
        write_json(
            report,
            {
                "passed": seed_gate_passed,
                "dry_run": True,
                "generated": 0,
                "blocked_reason": "" if seed_gate_passed else "seed gate blocked generation",
            },
        )
        return {
            "name": phase,
            "status": "passed" if seed_gate_passed else "blocked",
            "report": str(report),
        }

    if phase.startswith("generate"):
        report = phase_report_path(cloud_root, run_id, phase)
        seed_gate_passed = bool(context.get("seed_gate_passed", False))
        write_json(
            report,
            {
                "passed": seed_gate_passed,
                "dry_run": True,
                "requested": GENERATION_COUNTS.get(phase, 0),
                "generated": 0,
                "blocked_reason": "" if seed_gate_passed else "seed gate blocked generation",
            },
        )
        return {
            "name": phase,
            "status": "passed" if seed_gate_passed else "blocked",
            "report": str(report),
        }

    if phase == "quality-gate":
        report = reports_dir / f"validation_report_{run_id}.json"
        reject_report = reports_dir / f"reject_report_{run_id}.json"
        write_json(report, {"passed": False, "rows": 0, "valid": 0, "invalid": 0})
        write_json(reject_report, {"passed": True, "reject_reasons": ["dry_run_no_rows"]})
        return {"name": phase, "status": "blocked", "report": str(report)}

    if phase.startswith("quality-gate"):
        report = phase_report_path(cloud_root, run_id, phase)
        reject_report = reports_dir / f"reject_report_{phase}_{run_id}.json"
        write_json(report, {"passed": False, "rows": 0, "valid": 0, "invalid": 0})
        write_json(reject_report, {"passed": True, "reject_reasons": ["dry_run_no_rows"]})
        return {"name": phase, "status": "blocked", "report": str(report)}

    if phase == "dedupe":
        report = reports_dir / f"dedupe_report_{run_id}.json"
        write_json(report, {"rows": 0, "kept": 0, "dropped": 0, "passed": True})
        return {"name": phase, "status": "passed", "report": str(report)}

    if phase == "split":
        report = reports_dir / f"split_report_{run_id}.json"
        write_json(report, {"rows": 0, "train": 0, "validation": 0, "test": 0, "passed": False})
        return {"name": phase, "status": "blocked", "report": str(report)}

    if phase == "train":
        report = phase_report_path(cloud_root, run_id, phase)
        write_json(
            report,
            {
                "passed": False,
                "dry_run": True,
                "blocked_reason": "dry run does not train adapters",
            },
        )
        return {"name": phase, "status": "blocked", "report": str(report)}

    if phase == "infer":
        report = phase_report_path(cloud_root, run_id, phase)
        write_json(
            report,
            {
                "passed": False,
                "dry_run": True,
                "blocked_reason": "dry run does not run adapter inference",
            },
        )
        return {"name": phase, "status": "blocked", "report": str(report)}

    if phase == "eval":
        eval_report = {
            "rows": 0,
            "heldout_test_rows": 1,
            "json_parse_success": 0.0,
            "react_ir_schema_success": 0.0,
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
        eval_path = reports_dir / f"eval_report_{run_id}.json"
        gate_path = reports_dir / f"acceptance_gate_report_{run_id}.json"
        write_json(eval_path, eval_report)
        gate_report = evaluate_gates(
            spec=read_yaml(DATASET_SPEC),
            eval_report=eval_report,
            min_rows=1,
        )
        write_json(gate_path, gate_report)
        return {"name": phase, "status": "blocked", "report": str(eval_path)}

    if phase == "package":
        report = reports_dir / f"package_report_{run_id}.json"
        write_json(
            report,
            {
                "passed": False,
                "blocked_reason": "acceptance gate has not passed",
                "dry_run": True,
            },
        )
        return {"name": phase, "status": "blocked", "report": str(report)}

    report = reports_dir / f"{phase}_report_{run_id}.json"
    write_json(report, {"passed": False, "blocked_reason": "unknown phase"})
    return {"name": phase, "status": "failed", "report": str(report)}


if __name__ == "__main__":
    raise SystemExit(main())
