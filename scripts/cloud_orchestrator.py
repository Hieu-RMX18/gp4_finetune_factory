#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from check_acceptance_gates import evaluate_gates
from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
from factory_common import read_jsonl, read_yaml, write_json
from provider_probe import ProviderProbeResult, write_platform_status

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GP4 cloud-only workflow phases.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        print("cloud_orchestrator currently requires --dry-run for local execution")
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
    }
    requested_phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    phase_results: list[dict[str, Any]] = []
    for phase in requested_phases:
        phase_results.append(_run_dry_phase(phase, context))

    manifest = {
        "run_id": args.run_id,
        "dry_run": True,
        "cloud_root": str(args.cloud_root),
        "phases": phase_results,
    }
    write_json(manifest_path, manifest)
    failed = any(result["status"] in {"failed", "blocked"} for result in phase_results)
    print(f"run_id={args.run_id} dry_run=True failed={failed} manifest={manifest_path}")
    return 1 if failed else 0


def _run_dry_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    reports_dir: Path = context["reports_dir"]
    run_id = str(context["run_id"])
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

    if phase == "quality-gate":
        report = reports_dir / f"validation_report_{run_id}.json"
        reject_report = reports_dir / f"reject_report_{run_id}.json"
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
