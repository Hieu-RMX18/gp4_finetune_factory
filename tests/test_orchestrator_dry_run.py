import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cloud_orchestrator import (
    CLOUD_TRAIN_RATIO,
    CLOUD_VAL_RATIO,
    PREVIOUS_GATE_REPORTS,
    V2_300K_PHASES,
    _run_cloud_phase,
)
from check_cloud_storage_policy import CloudStoragePolicy
from factory_common import write_json
from provider_probe import ProviderProbeResult


def test_cloud_split_ratio_keeps_fast_eval_holdout() -> None:
    test_ratio = 1 - CLOUD_TRAIN_RATIO - CLOUD_VAL_RATIO

    assert CLOUD_TRAIN_RATIO == 0.90
    assert CLOUD_VAL_RATIO == 0.05
    assert round(300000 * test_ratio) == 15000

def test_dedupe_waits_for_final_tiered_quality_gate() -> None:
    assert PREVIOUS_GATE_REPORTS["dedupe"] == "quality-gate-50k"


def test_v2_300k_phase_chain_is_ordered() -> None:
    from cloud_orchestrator import V2_300K_PHASES

    assert V2_300K_PHASES == [
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
        "gpu-preflight",
        "train",
        "infer",
        "eval",
        "local-install-manifest",
        "package",
        "benchmark-report",
    ]


def test_v2_split_waits_for_quality_gate() -> None:
    from cloud_orchestrator import PREVIOUS_GATE_REPORTS

    assert PREVIOUS_GATE_REPORTS["split"] == "quality-gate-v2"
    assert PREVIOUS_GATE_REPORTS["gpu-preflight"] == "split"
    assert PREVIOUS_GATE_REPORTS["train"] == "gpu-preflight"
    assert PREVIOUS_GATE_REPORTS["local-install-manifest"] == "eval"
    assert PREVIOUS_GATE_REPORTS["package"] == "local-install-manifest"
    assert PREVIOUS_GATE_REPORTS["benchmark-report"] == "package"


def test_orchestrator_writes_root_manifest_and_runtime_state(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    seed = tmp_path / "seed.jsonl"
    seed.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cloud_orchestrator.py",
            "--run-id",
            "root-state",
            "--cloud-root",
            str(cloud_root),
            "--seed",
            str(seed),
            "--phases",
            "cloud-setup",
            "--dry-run",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    root_manifest = json.loads(
        (cloud_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    runtime_state = json.loads(
        (cloud_root / "runtime_state.json").read_text(encoding="utf-8")
    )
    assert root_manifest["run_id"] == "root-state"
    assert root_manifest["phases"][0]["name"] == "cloud-setup"
    assert runtime_state["run_id"] == "root-state"
    assert runtime_state["last_phase"] == "cloud-setup"
    assert runtime_state["status"] == "passed"


def test_benchmark_report_phase_uses_full_v2_evidence_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "benchmark-full-evidence"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    write_json(
        reports_dir / f"package_report_{run_id}.json",
        {"passed": True},
    )
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = [str(part) for part in command]
        report_path = Path(captured["command"][captured["command"].index("--report") + 1])
        write_json(report_path, {"passed": True})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cloud_orchestrator.subprocess.run", fake_run)

    result = _run_cloud_phase(
        "benchmark-report",
        {
            "run_id": run_id,
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
        },
    )

    assert result["status"] == "passed"
    command = captured["command"]
    input_reports = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--input-report"
    ]
    expected_names = {
        f"cloud-setup_{run_id}.json",
        f"import-old_{run_id}.json",
        f"validate-old-v2_{run_id}.json",
        f"plan-v2-target_{run_id}.json",
        f"merge-accepted_{run_id}.json",
        f"quality-gate-v2_{run_id}.json",
        f"eval_report_{run_id}.json",
        f"acceptance_gate_report_{run_id}.json",
        f"platform_status_{run_id}.json",
        f"colab_readiness_{run_id}.json",
        f"local-install-manifest_{run_id}.json",
        f"package_report_{run_id}.json",
    }
    assert {Path(path).name for path in input_reports} == expected_names


def test_cloud_import_old_allows_base_model_start_without_prior_artifacts(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)

    result = _run_cloud_phase(
        "import-old",
        {
            "run_id": "base-model-start",
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
        },
    )

    assert result["status"] == "passed"
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["old_dataset_count"] == 0
    assert report["adapter_only_reuse"] is False
    assert report["base_model_start"] is True
    assert report["blocked_reason"] == ""

def test_cloud_import_old_allows_adapter_only_reuse_with_previous_adapter(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    previous_adapter = cloud_root.parent / "previous/models/qwen25_gp4_lora"
    previous_adapter.mkdir(parents=True)

    result = _run_cloud_phase(
        "import-old",
        {
            "run_id": "adapter-only",
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
            "previous_adapter": previous_adapter,
        },
    )

    assert result["status"] == "passed"
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["old_dataset_count"] == 0
    assert report["adapter_only_reuse"] is True
    assert report["blocked_reason"] == ""


def test_v2_split_reads_accepted_300k_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "split-source"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    write_json(
        reports_dir / f"quality-gate-v2_{run_id}.json",
        {"passed": True, "rows": 300000, "valid": 300000, "invalid": 0},
    )
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = [str(part) for part in command]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cloud_orchestrator.subprocess.run", fake_run)

    result = _run_cloud_phase(
        "split",
        {
            "run_id": run_id,
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
        },
    )

    assert result["status"] == "passed"
    command = captured["command"]
    assert str(cloud_root / "data/validated/accepted_300k.jsonl") in command
    assert str(cloud_root / "data/validated/accepted.jsonl") not in command
    assert command[command.index("--seed") + 1] == "3526"
    assert command[command.index("--train-ratio") + 1] == "0.9"
    assert command[command.index("--val-ratio") + 1] == "0.05"


def test_train_phase_passes_previous_adapter_to_training_script(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "train-resume"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    write_json(reports_dir / f"gpu-preflight_{run_id}.json", {"passed": True})
    previous_adapter = cloud_root.parent / "previous/models/qwen25_gp4_lora"
    previous_adapter.mkdir(parents=True)
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = [str(part) for part in command]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cloud_orchestrator.subprocess.run", fake_run)

    result = _run_cloud_phase(
        "train",
        {
            "run_id": run_id,
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
            "previous_adapter": previous_adapter,
        },
    )

    assert result["status"] == "passed"
    command = captured["command"]
    assert "--resume-from-adapter" in command
    assert command[command.index("--resume-from-adapter") + 1] == str(previous_adapter)


def test_gpu_preflight_requires_cuda_and_updates_final_platform_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "gpu-preflight"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    write_json(reports_dir / f"split_{run_id}.json", {"passed": True})
    calls: list[bool] = []

    def fake_probe(env, cloud_root_value, *, provider=None, require_gpu=False):
        calls.append(require_gpu)
        return ProviderProbeResult(
            provider="colab",
            available=True,
            cloud_storage_ready=True,
            free_tier=True,
            paid_risk=False,
            blocked_reason="",
            gpu_required=require_gpu,
            gpu_available=require_gpu,
            gpu_name="Tesla T4",
        )

    monkeypatch.setattr("provider_probe.probe_from_environment", fake_probe)

    result = _run_cloud_phase(
        "gpu-preflight",
        {
            "run_id": run_id,
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
        },
    )

    assert result["status"] == "passed"
    assert calls == [True]
    phase_report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    final_status = json.loads(
        (reports_dir / f"platform_status_{run_id}.json").read_text(encoding="utf-8")
    )
    assert phase_report["gpu_required"] is True
    assert phase_report["gpu_available"] is True
    assert final_status["gpu_required"] is True
    assert final_status["gpu_available"] is True


def test_v2_300k_preset_allows_base_model_start_without_previous_adapter(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"
    seed = ROOT / "data/seed/gp4_seed_starter.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cloud_orchestrator.py",
            "--run-id",
                "base-model-start",
            "--cloud-root",
            str(cloud_root),
            "--seed",
            str(seed),
            "--phases",
            "train",
            "--preset",
            "v2-300k",
                "--dry-run",
                "--allow-tmp",
            ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "v2 300k run requires --previous-adapter" not in result.stdout
    import_report = json.loads(
        (cloud_root / "reports" / "import-old_base-model-start.json").read_text(
            encoding="utf-8"
        )
    )
    assert import_report["passed"] is True
    assert import_report["old_dataset_count"] == 0
    assert import_report["base_model_start"] is True


def test_eval_phase_uses_explicit_gp4_ws_contract_repo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "eval-contract"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    write_json(reports_dir / f"infer_{run_id}.json", {"passed": True, "rows": 3})
    target_repo = tmp_path / "gp4_ws_snapshot"
    monkeypatch.setenv("GP4_WS", str(target_repo))
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        string_command = [str(part) for part in command]
        captured.append(string_command)
        if "scripts/eval_model_outputs.py" in string_command:
            report_path = Path(string_command[string_command.index("--report") + 1])
            write_json(
                report_path,
                {
                    "rows": 3,
                    "heldout_test_rows": 3,
                    "json_parse_success": 1.0,
                    "react_ir_schema_success": 1.0,
                    "semantic_ir_success": 1.0,
                    "intent_accuracy": 1.0,
                    "primitive_type_leakage": 0,
                    "hardware_claims": 0,
                    "raw_trajectory_outputs": 0,
                    "ros_motoros_outputs": 0,
                    "safety_bypass_outputs": 0,
                    "dangerous_os_command_outputs": 0,
                    "unsafe_command_acceptance": 0,
                    "locked_typo_eval_rows": 1,
                    "locked_typo_eval_intent_accuracy": 1.0,
                    "locked_v2_eval_rows": 80,
                    "locked_v2_eval_intent_accuracy": 1.0,
                    "locked_v2_eval_exact_match": 1.0,
                },
            )
        if "scripts/check_acceptance_gates.py" in string_command:
            report_path = Path(string_command[string_command.index("--report") + 1])
            write_json(report_path, {"passed": True, "checks": []})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cloud_orchestrator.subprocess.run", fake_run)

    result = _run_cloud_phase(
        "eval",
        {
            "run_id": run_id,
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
        },
    )

    assert result["status"] == "passed"
    eval_command = captured[0]
    assert "--contract-repo" in eval_command
    assert eval_command[eval_command.index("--contract-repo") + 1] == str(target_repo)
    assert "--require-contract-repo" in eval_command


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

    assert result.returncode != 0
    assert "No rule to make target" not in result.stderr
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


def test_v2_target_plan_accounts_for_unique_old_rows_and_quota_deficits() -> None:
    from cloud_orchestrator import _build_v2_target_plan

    duplicate_old_rows = [
        {
            "messages": [{"content": "same"}],
            "expected_json": {"intent": "stop"},
            "metadata": {"scenario_tags": ["normal_motion"]},
        },
        {
            "messages": [{"content": "same"}],
            "expected_json": {"intent": "stop"},
            "metadata": {"scenario_tags": ["normal_motion"]},
        },
    ]
    spec = {
        "v2_merge_policy": {"target_total_accepted_rows": 3},
        "v2_distribution_gates": {
            "scenario_tag_min_counts": {
                "singularity": 1,
                "dangerous_os_command": 1,
            }
        },
    }

    plan = _build_v2_target_plan(duplicate_old_rows, spec)

    assert plan["old_rows_valid"] == 2
    assert plan["old_unique_rows"] == 1
    assert plan["unique_row_shortfall"] == 2
    assert plan["quota_deficit_rows"] == 2
    assert plan["new_rows_requested"] == 2


def test_v2_target_plan_applies_raw_candidate_budget() -> None:
    from cloud_orchestrator import _build_v2_target_plan

    spec = {
        "v2_merge_policy": {"target_total_accepted_rows": 4},
        "v2_distribution_gates": {"scenario_tag_min_counts": {}},
        "raw_candidate_budget": {"min_multiplier": 1.5, "max_multiplier": 2.0},
    }

    plan = _build_v2_target_plan(
        [
            {
                "messages": [{"role": "user", "content": "covered row"}],
                "expected_json": {"intent": "stop"},
                "metadata": {"scenario_tags": ["normal_motion"]},
            }
        ],
        spec,
    )

    assert plan["new_rows_requested"] == 3
    assert plan["raw_candidate_budget_multiplier"] == 1.5
    assert plan["raw_candidate_rows_requested"] == 5


def test_generate_v2_uses_raw_candidate_request_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cloud_root = tmp_path / "cloud"
    run_id = "generate-v2-raw-budget"
    reports_dir = cloud_root / "reports"
    reports_dir.mkdir(parents=True)
    write_json(
        reports_dir / f"plan-v2-target_{run_id}.json",
        {
            "passed": True,
            "new_rows_requested": 280000,
            "raw_candidate_rows_requested": 420000,
        },
    )
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = [str(part) for part in command]
        report_path = Path(captured["command"][captured["command"].index("--report") + 1])
        write_json(report_path, {"passed": True})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cloud_orchestrator.subprocess.run", fake_run)

    result = _run_cloud_phase(
        "generate-v2",
        {
            "run_id": run_id,
            "cloud_root": cloud_root,
            "reports_dir": reports_dir,
            "seed": Path("data/seed/gp4_seed_starter.jsonl"),
            "policy": CloudStoragePolicy((cloud_root,), allow_tmp=True),
            "provider": None,
            "source_plan": None,
            "dry_run": False,
            "allow_tmp": True,
            "old_datasets": [],
        },
    )

    assert result["status"] == "passed"
    command = captured["command"]
    assert command[command.index("--count") + 1] == "420000"


def test_v2_target_plan_requests_rows_to_satisfy_legacy_cap() -> None:
    from cloud_orchestrator import _build_v2_target_plan

    old_rows = [
        {
            "messages": [{"role": "user", "content": f"legacy {index}"}],
            "expected_json": {"intent": "stop"},
            "metadata": {},
        }
        for index in range(3)
    ]
    spec = {
        "v2_merge_policy": {"target_total_accepted_rows": 3},
        "v2_distribution_gates": {
            "scenario_tag_min_counts": {},
            "max_legacy_rows_without_scenario_tags": 1,
        },
    }

    plan = _build_v2_target_plan(old_rows, spec)

    assert plan["unique_row_shortfall"] == 0
    assert plan["legacy_cap_new_rows"] == 2
    assert plan["new_rows_requested"] == 2


def test_v2_target_plan_requests_full_quota_seed_when_any_tag_deficit_exists() -> None:
    from cloud_orchestrator import _build_v2_target_plan

    old_rows = [
        {
            "messages": [{"role": "user", "content": f"covered singularity {index}"}],
            "expected_json": {"intent": "stop"},
            "metadata": {"scenario_tags": ["singularity"]},
        }
        for index in range(4)
    ]
    spec = {
        "v2_merge_policy": {"target_total_accepted_rows": 4},
        "v2_distribution_gates": {
            "scenario_tag_min_counts": {
                "singularity": 3,
                "dangerous_os_command": 1,
            }
        },
    }

    plan = _build_v2_target_plan(old_rows, spec)

    assert plan["unique_row_shortfall"] == 0
    assert plan["quota_deficit_rows"] == 1
    assert plan["quota_seed_rows"] == 4
    assert plan["new_rows_requested"] == 4


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
        "gpu-preflight",
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


def test_orchestrator_v2_preset_dry_run_has_no_unknown_phases(
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
            "dryrun-v2",
            "--cloud-root",
            str(cloud_root),
            "--seed",
            str(seed),
            "--phases",
            "ignored",
            "--preset",
            "v2-300k",
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
        (cloud_root / "reports" / "run_manifest_dryrun-v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert [phase["name"] for phase in manifest["phases"]] == V2_300K_PHASES
    assert {phase["status"] for phase in manifest["phases"]} <= {"passed", "blocked"}
    phase_statuses = {phase["name"]: phase["status"] for phase in manifest["phases"]}
    assert phase_statuses["import-old"] == "passed"
    import_report = json.loads(
        (cloud_root / "reports" / "import-old_dryrun-v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert import_report["base_model_start"] is True
    local_manifest_report = json.loads(
        (
            cloud_root / "reports" / "local-install-manifest_dryrun-v2.json"
        ).read_text(encoding="utf-8")
    )
    assert local_manifest_report["install_action_performed"] is False


def test_make_cloud_v2_dry_run_writes_v2_manifest() -> None:
    cloud_root = Path("/tmp/gp4_finetune_factory_cloud_test_v2")
    shutil.rmtree(cloud_root, ignore_errors=True)

    result = subprocess.run(
        ["make", "cloud-v2-dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "No rule to make target" not in result.stderr
    manifest = json.loads(
        (cloud_root / "reports" / "run_manifest_dryrun-v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert [phase["name"] for phase in manifest["phases"]] == V2_300K_PHASES
