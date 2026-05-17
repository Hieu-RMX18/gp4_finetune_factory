import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cloud_orchestrator import CLOUD_TRAIN_RATIO, CLOUD_VAL_RATIO, PREVIOUS_GATE_REPORTS


def test_cloud_split_ratio_keeps_fast_eval_holdout() -> None:
    test_ratio = 1 - CLOUD_TRAIN_RATIO - CLOUD_VAL_RATIO

    assert CLOUD_TRAIN_RATIO == 0.998
    assert CLOUD_VAL_RATIO == 0.001
    assert int(50000 * test_ratio) == 50

def test_dedupe_waits_for_final_tiered_quality_gate() -> None:
    assert PREVIOUS_GATE_REPORTS["dedupe"] == "quality-gate-50k"


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
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["blocked_reason"] == "seed gate blocked generation"


def test_orchestrator_tiny_dry_run_writes_manifest_and_failing_acceptance(
    tmp_path: Path,
) -> None:
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
    manifest = json.loads(
        (cloud_root / "reports" / "run_manifest_dryrun.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (cloud_root / "reports" / "acceptance_gate_report_dryrun.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["run_id"] == "dryrun"
    assert gate["passed"] is False

def test_orchestrator_full_cloud_phase_dry_run_has_no_unknown_phases(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "drive" / "gp4_finetune_factory"
    seed = tmp_path / "seed.jsonl"
    seed.write_text("", encoding="utf-8")
    phases = [
        "cloud-setup",
        "provider-probe",
        "contract",
        "seed-check",
        "generate-smoke",
        "generate-1k",
        "quality-gate-1k",
        "generate-30k",
        "quality-gate-30k",
        "generate-50k",
        "quality-gate-50k",
        "dedupe",
        "split",
        "train",
        "infer",
        "eval",
        "package",
    ]

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
            ",".join(phases),
            "--dry-run",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    manifest = json.loads(
        (cloud_root / "reports" / "run_manifest_dryrun.json").read_text(
            encoding="utf-8"
        )
    )
    assert [phase["name"] for phase in manifest["phases"]] == phases
    assert {phase["status"] for phase in manifest["phases"]} <= {"passed", "blocked"}
