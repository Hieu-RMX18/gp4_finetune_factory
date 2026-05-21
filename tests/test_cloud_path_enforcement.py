import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cloud_runtime import (
    CloudPathError,
    configure_cloud_caches,
    copy_source_plan_to_cloud,
    phase_report_path,
    validate_cloud_run_paths,
)


def test_repository_runtime_artifact_dirs_are_empty() -> None:
    artifact_dirs = [
        ROOT / "artifact_downloads",
        ROOT / "data/generated",
        ROOT / "data/validated",
        ROOT / "data/splits",
        ROOT / "models",
        ROOT / "outputs",
        ROOT / "reports",
    ]
    runtime_files = [
        path.relative_to(ROOT).as_posix()
        for directory in artifact_dirs
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]

    assert runtime_files == []


def test_react_cloud_notebooks_use_source_bundle_and_cloud_root() -> None:
    notebook_paths = [
        ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb",
        ROOT / "notebooks/kaggle_gp4_react_qwen25_qlora.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )

        assert "gp4_finetune_factory_source_bundle.zip" in source
        assert "zipfile.ZipFile" in source
        assert "os.chdir(WORK_DIR)" in source
        assert "python -m pip install -q -r requirements-cloud.txt" in source
        assert "unsloth datasets trl" not in source
        assert "gp4-react-v2-300k-" in source
        assert "gp4-react-50k-" not in source
        assert "2026-05-20-gp4-v2-300k-readiness.md" in source
        assert "2026-05-16-gp4-react-ir-cloud-workflow.md" not in source
        assert "--source-plan" in source
        assert "SOURCE_PLAN" in source
        assert "--seed" in source
        assert "SEED_PATH" in source
        assert "--preset" in source
        assert "v2-300k" in source
        assert "benchmark_report_${RUN_ID}.html" in source
        assert "scripts/audit_cloud_completion.py" in source
        assert "$CLOUD_ROOT/data/seed/seed.jsonl" not in source


def test_colab_notebook_records_expected_drive_account_and_reuses_old_dataset() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "johnwickiller4444@gmail.com" in source
    assert "GP4_DRIVE_ROOT" in source
    assert "GP4_PREVIOUS_RUN_ID" in source
    assert "GP4_OLD_DATASET" in source
    assert "drive_account_hint.txt" in source
    assert "drive_account_confirmation.json" in source
    assert "GP4_DRIVE_ACCOUNT_CONFIRMED" in source
    assert "--old-dataset" in source
    assert "subprocess.run(orchestrator_command, check=True)" in source


def test_package_phase_report_path_matches_completion_audit(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"

    report_path = phase_report_path(cloud_root, "accepted", "package")

    assert report_path == cloud_root / "reports/package_report_accepted.json"

def test_non_dry_run_requires_cloud_root(tmp_path: Path) -> None:
    with pytest.raises(CloudPathError, match="CLOUD_ROOT"):
        validate_cloud_run_paths(
            cloud_root=None,
            dry_run=False,
            inputs=[],
            outputs=[tmp_path / "report.json"],
        )


def test_non_dry_run_blocks_output_outside_cloud_root(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"

    with pytest.raises(CloudPathError, match="outside CLOUD_ROOT"):
        validate_cloud_run_paths(
            cloud_root=cloud_root,
            dry_run=False,
            inputs=[],
            outputs=[tmp_path / "outside" / "report.json"],
            allow_tmp=True,
        )


def test_non_dry_run_blocks_local_runtime_artifact_inputs(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"

    with pytest.raises(CloudPathError, match="forbidden local runtime artifact input"):
        validate_cloud_run_paths(
            cloud_root=cloud_root,
            dry_run=False,
            inputs=[ROOT / "models" / "qwen25_gp4_lora_pilot"],
            outputs=[cloud_root / "reports" / "report.json"],
            allow_tmp=True,
        )


def test_copy_source_plan_to_cloud_and_hashes_it(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    source_plan = tmp_path / "PLAN.md"
    source_plan.write_text("# source plan\n", encoding="utf-8")

    copied = copy_source_plan_to_cloud(
        source_plan=source_plan,
        cloud_root=cloud_root,
        dry_run=False,
        allow_tmp=True,
    )

    assert copied.path == cloud_root / "manifests" / "source_plan.md"
    assert copied.path.read_text(encoding="utf-8") == "# source plan\n"
    assert len(copied.sha256) == 64


def test_configure_cloud_caches_sets_only_cloud_paths(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"

    cache_env = configure_cloud_caches(
        cloud_root=cloud_root,
        dry_run=False,
        allow_tmp=True,
    )

    assert cache_env["HF_HOME"] == str(cloud_root / ".cache" / "huggingface")
    assert cache_env["TORCH_HOME"] == str(cloud_root / ".cache" / "torch")
    assert all(str(value).startswith(str(cloud_root)) for value in cache_env.values())


def test_adapter_inference_sets_cloud_caches_before_model_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_adapter_inference

    cloud_root = tmp_path / "cloud"
    input_path = cloud_root / "data" / "splits" / "test.jsonl"
    output_path = cloud_root / "outputs" / "model_outputs.jsonl"
    report_path = cloud_root / "reports" / "inference_report.json"
    adapter_dir = cloud_root / "models" / "qwen25_gp4_lora"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        json.dumps(
            {
                "id": "gp4_test_000001",
                "messages": [{"role": "user", "content": "move home"}],
                "expected_json": {"action": "move_home"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    for key in [
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "WANDB_DIR",
        "TMPDIR",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(run_adapter_inference, "_adapter_exists", lambda _: True)

    def fake_run_inference(**kwargs: object) -> list[dict[str, object]]:
        for key in ["HF_HOME", "TORCH_HOME", "TMPDIR"]:
            assert os.environ[key].startswith(str(cloud_root))
        return [
            {
                "id": "gp4_test_000001",
                "expected_json": {"action": "move_home"},
                "metadata": {},
                "model_output": '{"action": "move_home"}',
            }
        ]

    monkeypatch.setattr(run_adapter_inference, "_run_inference", fake_run_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_adapter_inference.py",
            "--input",
            str(input_path),
            "--adapter-dir",
            str(adapter_dir),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--cloud-root",
            str(cloud_root),
            "--allow-tmp",
        ],
    )

    assert run_adapter_inference.main() == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def test_orchestrator_non_dry_run_requires_previous_gate_report(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cloud_orchestrator.py",
            "--run-id",
            "gatecheck",
            "--cloud-root",
            str(cloud_root),
            "--seed",
            "data/seed/gp4_seed_starter.jsonl",
            "--phases",
            "generate-30k",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(
        (cloud_root / "reports" / "generate-30k_gatecheck.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is False
    assert "quality-gate-1k" in report["blocked_reason"]

def test_orchestrator_non_dry_run_stops_after_blocked_provider(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cloud_orchestrator.py",
            "--run-id",
            "provider-blocked",
            "--cloud-root",
            str(cloud_root),
            "--seed",
            "data/seed/gp4_seed_starter.jsonl",
            "--phases",
            "provider-probe,contract,seed-check",
            "--provider",
            "local",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    manifest = json.loads(
        (cloud_root / "reports" / "run_manifest_provider-blocked.json").read_text(
            encoding="utf-8"
        )
    )
    assert [phase["name"] for phase in manifest["phases"]] == ["provider-probe"]
    assert not (cloud_root / "reports" / "contract_provider-blocked.json").exists()


def test_orchestrator_blocks_local_old_dataset_path(
    tmp_path: Path,
) -> None:
    cloud_root = tmp_path / "cloud"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cloud_orchestrator.py",
            "--run-id",
            "local-old-dataset",
            "--cloud-root",
            str(cloud_root),
            "--seed",
            "data/seed/gp4_seed_starter.jsonl",
            "--phases",
            "import-old",
            "--old-dataset",
            "data/validated/.gitkeep",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "old dataset path is not under approved cloud storage" in (
        result.stdout + result.stderr
    )
    assert not (
        cloud_root / "reports" / "run_manifest_local-old-dataset.json"
    ).exists()
