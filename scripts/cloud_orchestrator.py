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
from dataset_keys import dataset_identity_key
from factory_common import git_value, read_json, read_jsonl, read_yaml, write_json, write_jsonl
from locked_v2_eval import write_locked_v2_eval
from package_adapter import adapter_artifact_exists
from provider_probe import ProviderProbeResult, write_platform_status
from typo_noise_policy import write_locked_typo_eval
from v2_taxonomy import distribution_summary

ROOT = Path(__file__).resolve().parents[1]
DATASET_SPEC = ROOT / "configs/dataset_spec.yaml"
EXPECTED_DRIVE_ACCOUNT_EMAIL = "johnwickiller4444@gmail.com"

PREVIOUS_GATE_REPORTS = {
    "generate": "seed-check",
    "quality-gate": "generate",
    "dedupe": "quality-gate-50k",
    "split": "quality-gate-v2",
    "train": "split",
    "infer": "train",
    "eval": "infer",
    "package": "local-install-manifest",
    "generate-smoke": "seed-check",
    "generate-1k": "generate-smoke",
    "quality-gate-1k": "generate-1k",
    "generate-30k": "quality-gate-1k",
    "quality-gate-30k": "generate-30k",
    "generate-50k": "quality-gate-30k",
    "quality-gate-50k": "generate-50k",
    "generate-100k": "quality-gate-50k",
    "quality-gate-100k": "generate-100k",
    "validate-old-v2": "import-old",
    "plan-v2-target": "validate-old-v2",
    "generate-v2": "plan-v2-target",
    "validate-new-v2": "generate-v2",
    "merge-accepted": "validate-new-v2",
    "quality-gate-v2": "merge-accepted",
    "local-install-manifest": "eval",
    "benchmark-report": "package",
}

V2_300K_PHASES = [
    "cloud-setup",
    "provider-probe",
    "contract",
    "seed-check",
    "import-old",
    "validate-old-v2",
    "plan-v2-target",
    "generate-smoke",
    "generate-v2",
    "validate-new-v2",
    "merge-accepted",
    "quality-gate-v2",
    "split",
    "train",
    "infer",
    "eval",
    "local-install-manifest",
    "package",
    "benchmark-report",
]

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
    parser.add_argument("--old-dataset", action="append", default=[])
    parser.add_argument("--previous-adapter", type=Path)
    parser.add_argument("--preset", choices=["legacy", "v2-300k"], default="legacy")
    args = parser.parse_args()

    if args.preset == "v2-300k" and not args.dry_run and args.previous_adapter is None:
        print("blocked_reason=v2 300k run requires --previous-adapter")
        return 1

    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=args.dry_run,
            inputs=[args.seed],
            outputs=[args.cloud_root / "reports" / f"run_manifest_{args.run_id}.json"],
            allow_tmp=args.allow_tmp,
        )
        _validate_old_dataset_paths(
            old_datasets=[Path(path) for path in args.old_dataset],
            cloud_root=args.cloud_root,
            allow_tmp=args.allow_tmp,
        )
        _validate_previous_adapter_path(
            previous_adapter=args.previous_adapter,
            cloud_root=args.cloud_root,
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
        "old_datasets": [Path(path) for path in args.old_dataset],
        "previous_adapter": args.previous_adapter,
    }
    requested_phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    if args.preset == "v2-300k":
        requested_phases = V2_300K_PHASES
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


def _validate_old_dataset_paths(
    *,
    old_datasets: list[Path],
    cloud_root: Path,
    allow_tmp: bool,
) -> None:
    if not old_datasets:
        return
    policy = CloudStoragePolicy(
        cloud_roots=(cloud_root, cloud_root.parent),
        allow_tmp=allow_tmp,
    )
    for path in old_datasets:
        if not is_allowed_cloud_path(path, policy):
            raise CloudPathError(
                f"old dataset path is not under approved cloud storage: {path}"
            )


def _validate_previous_adapter_path(
    *,
    previous_adapter: Path | None,
    cloud_root: Path,
    allow_tmp: bool,
) -> None:
    if previous_adapter is None:
        return
    policy = CloudStoragePolicy(
        cloud_roots=(cloud_root, cloud_root.parent),
        allow_tmp=allow_tmp,
    )
    if not is_allowed_cloud_path(previous_adapter, policy):
        raise CloudPathError(
            f"previous adapter path is not under approved cloud storage: {previous_adapter}"
        )


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
    if phase == "import-old":
        return _run_import_old_phase(phase, context)
    if phase == "validate-old-v2":
        return _run_validate_old_v2_phase(phase, context)
    if phase == "plan-v2-target":
        return _run_plan_v2_target_phase(phase, context)
    if phase == "generate-v2":
        return _run_generate_v2_phase(phase, context)
    if phase == "validate-new-v2":
        return _run_validate_new_v2_phase(phase, context)
    if phase == "merge-accepted":
        return _run_merge_accepted_phase(phase, context)
    if phase == "quality-gate-v2":
        return _run_quality_gate_v2_phase(phase, context)
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
            str(cloud_root / "data/validated/accepted_300k.jsonl"),
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
        command = [
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
        ]
        previous_adapter: Path | None = context.get("previous_adapter")
        if previous_adapter is not None:
            command.extend(["--resume-from-adapter", str(previous_adapter)])
        return _run_script_phase(phase, context, command)
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
    if phase == "benchmark-report":
        return _run_benchmark_report_phase(phase, context)
    if phase == "local-install-manifest":
        return _run_local_install_manifest_phase(phase, context)
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
    drive_hint_path = cloud_root / "manifests/drive_account_hint.txt"
    drive_hint_path.write_text(EXPECTED_DRIVE_ACCOUNT_EMAIL + "\n", encoding="utf-8")
    report = {
        "passed": True,
        "source_plan": str(copied.path),
        "source_plan_sha256": copied.sha256,
        "contract_manifest": str(manifest_path),
        "drive_account_hint": str(drive_hint_path),
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


def _run_import_old_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    old_datasets: list[Path] = context.get("old_datasets", [])
    if not old_datasets:
        report_path = write_phase_report(
            cloud_root=cloud_root,
            run_id=str(context["run_id"]),
            phase=phase,
            payload={
                "passed": False,
                "old_datasets": [],
                "old_dataset_count": 0,
                "old_dataset_fingerprints": [],
                "blocked_reason": "v2 300k run requires a previous accepted dataset",
            },
            allow_tmp=bool(context["allow_tmp"]),
        )
        return {"name": phase, "status": "blocked", "report": str(report_path)}
    report = {
        "passed": True,
        "old_datasets": [str(path) for path in old_datasets],
        "old_dataset_count": len(old_datasets),
        "old_dataset_fingerprints": [
            {
                "path": str(path),
                "rows": len(read_jsonl(path)),
                "sha256": sha256_file(path),
            }
            for path in old_datasets
        ],
    }
    report_path = write_phase_report(
        cloud_root=cloud_root,
        run_id=str(context["run_id"]),
        phase=phase,
        payload=report,
        allow_tmp=bool(context["allow_tmp"]),
    )
    return {"name": phase, "status": "passed", "report": str(report_path)}


def _run_validate_old_v2_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    old_datasets: list[Path] = context.get("old_datasets", [])
    report_path = phase_report_path(cloud_root, str(context["run_id"]), phase)
    if not old_datasets:
        write_json(
            report_path,
            {
                "passed": False,
                "old_rows_valid": 0,
                "old_validated_paths": [],
                "blocked_reason": "v2 300k run requires a previous accepted dataset",
            },
        )
        return {"name": phase, "status": "blocked", "report": str(report_path)}

    result = _run_script_phase(
        phase,
        context,
        [
            "scripts/validate_react_ir_dataset.py",
            "--input",
            *[str(path) for path in old_datasets],
            "--strict",
            "--report",
            str(report_path),
            "--cloud-root",
            str(cloud_root),
        ],
    )
    if result["status"] == "passed":
        old_rows: list[dict[str, Any]] = []
        for path in old_datasets:
            old_rows.extend(read_jsonl(path))
        merged_old = cloud_root / "data/validated/old_validated_v2.jsonl"
        write_jsonl(merged_old, old_rows)
        payload = read_json(report_path)
        payload["old_rows_valid"] = len(old_rows)
        payload["old_validated_paths"] = [str(path) for path in old_datasets]
        payload["old_validated_merged_path"] = str(merged_old)
        write_json(report_path, payload)
    return result


def _run_plan_v2_target_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    spec = read_yaml(DATASET_SPEC)
    old_report = read_json(phase_report_path(cloud_root, str(context["run_id"]), "validate-old-v2"))
    old_path = old_report.get("old_validated_merged_path", "")
    old_rows = read_jsonl(Path(str(old_path))) if old_path else []
    plan = _build_v2_target_plan(old_rows, spec)
    report_path = write_phase_report(
        cloud_root=cloud_root,
        run_id=str(context["run_id"]),
        phase=phase,
        payload={"passed": True, **plan},
        allow_tmp=bool(context["allow_tmp"]),
    )
    return {"name": phase, "status": "passed", "report": str(report_path)}


def _run_generate_v2_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    plan = read_json(phase_report_path(cloud_root, str(context["run_id"]), "plan-v2-target"))
    count = int(plan.get("new_rows_requested", 0))
    return _run_script_phase(
        phase,
        context,
        [
            "scripts/generate_batch_deepseek.py",
            "--seed",
            str(context["seed"]),
            "--output",
            str(cloud_root / "data/generated/raw_generate-v2.jsonl"),
            "--cloud-root",
            str(cloud_root),
            "--report",
            str(phase_report_path(cloud_root, str(context["run_id"]), phase)),
            "--count",
            str(count),
        ],
    )


def _build_v2_target_plan(
    old_rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    target_rows = int(spec["v2_merge_policy"]["target_total_accepted_rows"])
    unique_old_rows = _unique_rows(old_rows)
    distribution = distribution_summary(unique_old_rows)
    scenario_counts = distribution.get("scenario_tags", {})
    gates = spec.get("v2_distribution_gates", {})
    minimums = gates.get("scenario_tag_min_counts", {}) if isinstance(gates, dict) else {}
    quota_deficit_rows = 0
    if isinstance(minimums, dict):
        for tag, minimum in minimums.items():
            quota_deficit_rows += max(
                0,
                int(minimum) - int(scenario_counts.get(str(tag), 0)),
            )
    unique_row_shortfall = max(0, target_rows - len(unique_old_rows))
    return {
        "target_rows": target_rows,
        "old_rows_valid": len(old_rows),
        "old_unique_rows": len(unique_old_rows),
        "old_distribution": distribution,
        "unique_row_shortfall": unique_row_shortfall,
        "quota_deficit_rows": quota_deficit_rows,
        "new_rows_requested": max(unique_row_shortfall, quota_deficit_rows),
    }


def _unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = dataset_identity_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _run_validate_new_v2_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    return _run_script_phase(
        phase,
        context,
        [
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(cloud_root / "data/generated/raw_generate-v2.jsonl"),
            "--strict",
            "--report",
            str(phase_report_path(cloud_root, str(context["run_id"]), phase)),
            "--cloud-root",
            str(cloud_root),
        ],
    )


def _run_merge_accepted_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    spec = read_yaml(DATASET_SPEC)
    target_rows = int(spec["v2_merge_policy"]["target_total_accepted_rows"])
    old_report = read_json(phase_report_path(cloud_root, str(context["run_id"]), "validate-old-v2"))
    old_path = old_report.get("old_validated_merged_path", "")
    command = [
        "scripts/merge_accepted_datasets.py",
        "--new",
        str(cloud_root / "data/generated/raw_generate-v2.jsonl"),
        "--output",
        str(cloud_root / "data/validated/accepted_300k.jsonl"),
        "--report",
        str(phase_report_path(cloud_root, str(context["run_id"]), phase)),
        "--target-rows",
        str(target_rows),
        "--distribution-spec",
        str(DATASET_SPEC),
        "--cloud-root",
        str(cloud_root),
    ]
    if old_path:
        command[1:1] = ["--old", str(old_path)]
    return _run_script_phase(phase, context, command)


def _run_quality_gate_v2_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    return _run_script_phase(
        phase,
        context,
        [
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(cloud_root / "data/validated/accepted_300k.jsonl"),
            "--strict",
            "--report",
            str(phase_report_path(cloud_root, str(context["run_id"]), phase)),
            "--cloud-root",
            str(cloud_root),
            "--distribution-spec",
            str(DATASET_SPEC),
            "--enforce-v2-distribution",
        ],
    )


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


def _run_local_install_manifest_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    run_id = str(context["run_id"])
    target_repo = os.environ.get("GP4_WS", "").strip()
    if not target_repo:
        return _block_cloud_phase(
            phase,
            context,
            "GP4_WS is required to record local install target repository",
        )
    return _run_script_phase(
        phase,
        context,
        [
            "scripts/build_local_install_manifest.py",
            "--adapter-dir",
            str(cloud_root / "models/qwen25_gp4_lora"),
            "--acceptance-report",
            str(cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json"),
            "--target-repo",
            target_repo,
            "--cloud-root",
            str(cloud_root),
            "--output",
            str(phase_report_path(cloud_root, run_id, phase)),
        ],
    )


def _run_benchmark_report_phase(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    cloud_root: Path = context["cloud_root"]
    run_id = str(context["run_id"])
    return _run_script_phase(
        phase,
        context,
        [
            "scripts/build_quality_report.py",
            "--input-report",
            str(phase_report_path(cloud_root, run_id, "cloud-setup")),
            "--input-report",
            str(phase_report_path(cloud_root, run_id, "import-old")),
            "--input-report",
            str(phase_report_path(cloud_root, run_id, "validate-old-v2")),
            "--input-report",
            str(phase_report_path(cloud_root, run_id, "plan-v2-target")),
            "--input-report",
            str(phase_report_path(cloud_root, run_id, "merge-accepted")),
            "--input-report",
            str(cloud_root / "reports" / f"quality-gate-v2_{run_id}.json"),
            "--input-report",
            str(cloud_root / "reports" / f"eval_report_{run_id}.json"),
            "--input-report",
            str(cloud_root / "reports" / f"acceptance_gate_report_{run_id}.json"),
            "--input-report",
            str(cloud_root / "reports" / f"platform_status_{run_id}.json"),
            "--input-report",
            str(cloud_root / "reports" / f"colab_readiness_{run_id}.json"),
            "--input-report",
            str(cloud_root / "reports" / f"local-install-manifest_{run_id}.json"),
            "--input-report",
            str(phase_report_path(cloud_root, run_id, "package")),
            "--report",
            str(phase_report_path(cloud_root, run_id, phase)),
            "--html-report",
            str(cloud_root / "reports" / f"benchmark_report_{run_id}.html"),
            "--markdown-report",
            str(cloud_root / "reports" / f"benchmark_report_{run_id}.md"),
            "--distribution-spec",
            str(DATASET_SPEC),
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
    contract_repo = os.environ.get("GP4_WS", "").strip()
    if not contract_repo:
        return _block_cloud_phase(
            phase,
            context,
            "GP4_WS is required to evaluate against the target gp4_ws contract",
        )
    eval_result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_model_outputs.py",
            "--input",
            str(cloud_root / "outputs/model_outputs.jsonl"),
            "--contract-repo",
            contract_repo,
            "--require-contract-repo",
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
    _add_v2_distribution_metrics(metrics, cloud_root, run_id)
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
    locked_v2_path = cloud_root / "eval/locked_v2_eval.jsonl"
    if not locked_v2_path.exists():
        write_locked_v2_eval(locked_v2_path)
    locked_v2_rows = read_jsonl(locked_v2_path)
    heldout_path = cloud_root / "eval/heldout_with_locked_typo.jsonl"
    write_jsonl(heldout_path, [*test_rows, *locked_rows, *locked_v2_rows])
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
    payload = {
        "passed": True,
        "run_id": str(context["run_id"]),
        "hashes": hashes,
        "source_contract": "local source snapshot",
    }
    raw_gp4_ws = os.environ.get("GP4_WS", "").strip()
    if raw_gp4_ws:
        gp4_ws = Path(raw_gp4_ws)
        payload.update(
            {
                "repo_path": str(gp4_ws),
                "branch": git_value(gp4_ws, "branch", "--show-current"),
                "head": git_value(gp4_ws, "rev-parse", "HEAD"),
                "expected_commit": os.environ.get("GP4_WS_EXPECTED_COMMIT", "").strip(),
                "is_dirty": bool(git_value(gp4_ws, "status", "--short")),
            }
        )
    return payload


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

    if phase == "import-old":
        report = phase_report_path(cloud_root, run_id, phase)
        old_datasets: list[Path] = context.get("old_datasets", [])
        passed = bool(old_datasets)
        write_json(
            report,
            {
                "passed": passed,
                "dry_run": True,
                "old_datasets": [str(path) for path in old_datasets],
                "old_dataset_count": len(old_datasets),
                "blocked_reason": (
                    "" if passed else "v2 300k run requires a previous accepted dataset"
                ),
            },
        )
        return {"name": phase, "status": "passed" if passed else "blocked", "report": str(report)}

    if phase == "validate-old-v2":
        report = phase_report_path(cloud_root, run_id, phase)
        old_datasets: list[Path] = context.get("old_datasets", [])
        passed = bool(old_datasets)
        write_json(
            report,
            {
                "passed": passed,
                "dry_run": True,
                "old_rows_valid": 0,
                "old_validated_paths": [str(path) for path in old_datasets],
                "blocked_reason": (
                    "" if passed else "v2 300k run requires a previous accepted dataset"
                ),
            },
        )
        return {"name": phase, "status": "passed" if passed else "blocked", "report": str(report)}

    if phase == "plan-v2-target":
        report = phase_report_path(cloud_root, run_id, phase)
        target_rows = int(read_yaml(DATASET_SPEC)["v2_merge_policy"]["target_total_accepted_rows"])
        write_json(
            report,
            {
                "passed": True,
                "dry_run": True,
                "target_rows": target_rows,
                "old_rows_valid": 0,
                "new_rows_requested": target_rows,
            },
        )
        return {"name": phase, "status": "passed", "report": str(report)}

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

    if phase == "validate-new-v2":
        report = phase_report_path(cloud_root, run_id, phase)
        write_json(report, {"passed": False, "rows": 0, "valid": 0, "invalid": 0, "dry_run": True})
        return {"name": phase, "status": "blocked", "report": str(report)}

    if phase == "merge-accepted":
        report = phase_report_path(cloud_root, run_id, phase)
        target_rows = int(read_yaml(DATASET_SPEC)["v2_merge_policy"]["target_total_accepted_rows"])
        write_json(
            report,
            {
                "passed": False,
                "dry_run": True,
                "target_rows": target_rows,
                "output_rows": 0,
                "blocked_reason": "dry run does not generate accepted rows",
            },
        )
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
            "dangerous_os_command_outputs": 0,
            "unsafe_command_acceptance": 0,
            "locked_v2_eval_rows": 0,
            "locked_v2_eval_intent_accuracy": 0.0,
            "locked_v2_eval_exact_match": 0.0,
            "final_adapter_exists": False,
            "local_artifact_usage": 0,
            "v2_total_rows": 0,
            "v2_singularity_rows": 0,
            "v2_wrist_flip_rows": 0,
            "v2_joint_wrap_rows": 0,
            "v2_timeout_abort_recovery_rows": 0,
            "v2_approval_required_rows": 0,
            "v2_collision_limit_edge_rows": 0,
            "v2_dangerous_os_command_rows": 0,
            "v2_unsupported_tool_hallucination_rows": 0,
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

    if phase == "local-install-manifest":
        report = phase_report_path(cloud_root, run_id, phase)
        write_json(
            report,
            {
                "ready_for_local_install": False,
                "install_action_performed": False,
                "blocked_reason": "dry run does not produce an accepted adapter",
                "dry_run": True,
            },
        )
        return {"name": phase, "status": "blocked", "report": str(report)}

    if phase == "benchmark-report":
        report = phase_report_path(cloud_root, run_id, phase)
        html_report = cloud_root / "reports" / f"benchmark_report_{run_id}.html"
        markdown_report = cloud_root / "reports" / f"benchmark_report_{run_id}.md"
        html_report.write_text(
            "<!doctype html><h1>GP4 V2 Benchmark Report</h1>"
            "<h2>Benchmark Columns</h2><table></table>"
            "<h2>Scenario Tag Distribution</h2><svg></svg>\n",
            encoding="utf-8",
        )
        markdown_report.write_text(
            "# GP4 V2 Benchmark Report\n\n"
            "## Maintenance Reference\n\n"
            "| source | metric | actual | operator | threshold | passed |\n",
            encoding="utf-8",
        )
        write_json(
            report,
            {
                "passed": False,
                "dry_run": True,
                "benchmark_columns": [
                    "source",
                    "metric",
                    "actual",
                    "operator",
                    "threshold",
                    "passed",
                ],
                "benchmark_rows": [],
                "charts": {
                    "benchmark_actual_vs_threshold": [],
                    "scenario_tag_distribution": [],
                },
                "html_report": str(html_report),
                "markdown_report": str(markdown_report),
                "blocked_reason": "dry run does not produce accepted benchmarks",
            },
        )
        return {"name": phase, "status": "blocked", "report": str(report)}

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


def _add_v2_distribution_metrics(
    metrics: dict[str, Any],
    cloud_root: Path,
    run_id: str,
) -> None:
    quality_report_path = phase_report_path(cloud_root, run_id, "quality-gate-v2")
    if not quality_report_path.exists():
        return
    quality_report = read_json(quality_report_path)
    metrics["v2_total_rows"] = int(quality_report.get("rows", 0) or 0)
    distribution = quality_report.get("distribution", {})
    if not isinstance(distribution, dict):
        return
    scenario_tags = distribution.get("scenario_tags", {})
    if not isinstance(scenario_tags, dict):
        return
    for tag, count in scenario_tags.items():
        if isinstance(count, bool):
            continue
        try:
            metrics[f"v2_{tag}_rows"] = int(count)
        except (TypeError, ValueError):
            continue


if __name__ == "__main__":
    raise SystemExit(main())
