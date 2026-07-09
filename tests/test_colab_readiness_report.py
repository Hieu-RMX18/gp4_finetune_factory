import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_cloud_storage_policy import CloudStoragePolicy
from colab_readiness_report import build_colab_readiness_report


def _target_repo(path: Path, branch: str = "ws-deep-rebuild-3526") -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-b", branch], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("gp4_ws snapshot\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=tests@example.com",
            "-c",
            "user.name=Tests",
            "commit",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        check=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (path / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")


def _previous_adapter(cloud_root: Path) -> Path:
    previous_adapter = cloud_root.parent / "previous/models/qwen25_gp4_lora"
    _adapter(previous_adapter)
    return previous_adapter


def _cloud_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    drive_root = tmp_path / "drive" / "gp4_finetune_factory"
    cloud_root = drive_root / "run"
    cloud_root.mkdir(parents=True)
    (cloud_root / "manifests").mkdir()
    (cloud_root / "manifests/drive_account_hint.txt").write_text(
        "johnwickiller4444@gmail.com\n",
        encoding="utf-8",
    )
    (cloud_root / "manifests/drive_account_confirmation.json").write_text(
        json.dumps(
            {
                "confirmed": True,
                "confirmed_email": "johnwickiller4444@gmail.com",
                "expected_email": "johnwickiller4444@gmail.com",
                "method": "operator_input_after_drive_mount",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    old_dataset = drive_root / "previous/data/validated/accepted_300k.jsonl"
    old_dataset.parent.mkdir(parents=True)
    old_dataset.write_text('{"id":"old-1"}\n', encoding="utf-8")
    gp4_ws = cloud_root / "contract_snapshots/gp4_ws_ws-deep-rebuild-3526"
    expected_commit = _target_repo(gp4_ws)
    return cloud_root, old_dataset, gp4_ws, expected_commit


def test_colab_readiness_report_passes_for_drive_old_dataset_and_pinned_gp4_ws(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is True
    assert report["drive_account"]["email"] == "johnwickiller4444@gmail.com"
    assert report["drive_account"]["confirmed"] is True
    assert report["drive_account"]["confirmed_email"] == "johnwickiller4444@gmail.com"
    assert report["old_dataset"]["rows"] == 1
    assert report["old_dataset"]["sha256"]
    assert report["gp4_ws"]["branch"] == "ws-deep-rebuild-3526"
    assert report["gp4_ws"]["expected_commit"] == expected_commit
    assert report["gp4_ws"]["expected_commit_matches"] is True
    assert report["previous_adapter"]["path"] == str(previous_adapter)
    assert report["previous_adapter"]["artifact_exists"] is True
    assert report["previous_run"]["old_dataset_run_id"] == "previous"
    assert report["previous_run"]["previous_adapter_run_id"] == "previous"
    assert report["previous_run"]["matched"] is True
    assert report["install_action_performed"] is False
    assert all(check["passed"] for check in report["checks"])

def test_colab_readiness_report_records_cross_account_runtime_runner(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    (cloud_root / "manifests/drive_account_confirmation.json").write_text(
        json.dumps(
            {
                "confirmed": True,
                "confirmed_email": "johnwickiller4444@gmail.com",
                "expected_email": "johnwickiller4444@gmail.com",
                "storage_owner_email": "johnwickiller4444@gmail.com",
                "runtime_google_account_confirmed": "gpu.runner@example.com",
                "cross_account_runner": True,
                "method": "operator_or_explicit_requested_account_after_drive_mount",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is True
    assert report["drive_account"]["storage_owner_email"] == "johnwickiller4444@gmail.com"
    assert report["drive_account"]["runtime_account_confirmed"] == "gpu.runner@example.com"
    assert report["drive_account"]["cross_account_runner"] is True
    assert report["drive_account"]["matches_expected"] is True


def test_colab_readiness_report_passes_for_previous_adapter_under_drive(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is True
    assert report["previous_adapter"]["path"] == str(previous_adapter)
    assert report["previous_adapter"]["artifact_exists"] is True
    assert report["previous_adapter"]["allowed_cloud_path"] is True
    assert report["previous_run"]["matched"] is True


def test_colab_readiness_report_passes_for_adapter_only_previous_reuse(
    tmp_path: Path,
) -> None:
    cloud_root, _old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = cloud_root.parent / "previous/models/qwen25_gp4_lora/checkpoint-100"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=None,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        allow_adapter_only_reuse=True,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is True
    assert report["old_dataset"]["adapter_only_reuse"] is True
    assert report["old_dataset"]["rows"] == 0
    assert report["previous_adapter"]["artifact_exists"] is True
    assert report["previous_run"]["previous_adapter_run_id"] == "previous"
    assert report["previous_run"]["adapter_only_reuse"] is True
    assert report["previous_run"]["matched"] is True


def test_colab_readiness_report_passes_for_legacy_drive_root_pilot_adapter(
    tmp_path: Path,
) -> None:
    cloud_root, _old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = cloud_root.parent / "models/qwen25_gp4_lora_pilot"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=None,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        allow_adapter_only_reuse=True,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is True
    assert report["previous_adapter"]["path"] == str(previous_adapter)
    assert report["previous_run"]["previous_adapter_run_id"] == "legacy-drive-root"
    assert report["previous_run"]["adapter_only_reuse"] is True
    assert report["previous_run"]["matched"] is True


def test_colab_readiness_report_rejects_adapter_only_current_run_reuse(
    tmp_path: Path,
) -> None:
    cloud_root, _old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = cloud_root / "models/qwen25_gp4_lora/checkpoint-100"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=None,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        allow_adapter_only_reuse=True,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    assert report["previous_run"]["previous_adapter_run_id"] == "run"
    assert report["previous_run"]["adapter_only_reuse"] is True
    assert report["previous_run"]["matched"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "previous_reuse_same_prior_run" in failed


def test_colab_readiness_report_allows_old_dataset_with_base_model_start(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is True
    assert report["old_dataset"]["rows"] == 1
    assert report["old_dataset"]["fresh_generation_from_base"] is False
    assert report["previous_adapter"]["path"] == ""
    assert report["previous_adapter"]["base_model_start"] is True
    assert report["previous_run"]["base_model_start"] is True
    assert report["previous_run"]["matched"] is True
    assert all(check["passed"] for check in report["checks"])


def test_colab_readiness_report_rejects_mismatched_previous_run_reuse(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = cloud_root.parent / "other_previous/models/qwen25_gp4_lora"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    assert report["previous_run"]["old_dataset_run_id"] == "previous"
    assert report["previous_run"]["previous_adapter_run_id"] == "other_previous"
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "previous_reuse_same_prior_run" in failed


def test_colab_readiness_report_allows_explicit_mixed_prior_artifacts(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = cloud_root.parent / "adapter_run/models/qwen25_gp4_lora"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        allow_mixed_prior_artifacts=True,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is True
    assert report["previous_run"]["old_dataset_run_id"] == "previous"
    assert report["previous_run"]["previous_adapter_run_id"] == "adapter_run"
    assert report["previous_run"]["same_source_run"] is False
    assert report["previous_run"]["mixed_prior_artifacts"] is True
    assert report["previous_run"]["matched"] is True


def test_colab_readiness_report_rejects_current_run_reuse(
    tmp_path: Path,
) -> None:
    cloud_root, _old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    old_dataset = cloud_root / "data/validated/accepted_300k.jsonl"
    old_dataset.parent.mkdir(parents=True)
    old_dataset.write_text('{"id":"current-run-row"}\n', encoding="utf-8")
    previous_adapter = cloud_root / "models/qwen25_gp4_lora"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    assert report["previous_run"]["old_dataset_run_id"] == "run"
    assert report["previous_run"]["previous_adapter_run_id"] == "run"
    assert report["previous_run"]["matched"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "previous_reuse_same_prior_run" in failed


def test_colab_readiness_report_rejects_current_run_dataset_with_base_model_start(
    tmp_path: Path,
) -> None:
    cloud_root, _old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    old_dataset = cloud_root / "data/validated/accepted_300k.jsonl"
    old_dataset.parent.mkdir(parents=True)
    old_dataset.write_text('{"id":"current-run-row"}\n', encoding="utf-8")

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    assert report["previous_adapter"]["base_model_start"] is False
    assert report["previous_run"]["old_dataset_run_id"] == "run"
    assert report["previous_run"]["matched"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "previous_reuse_same_prior_run" in failed

def test_colab_readiness_report_rejects_current_run_dataset_with_mixed_reuse(
    tmp_path: Path,
) -> None:
    cloud_root, _old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    old_dataset = cloud_root / "data/validated/accepted_300k.jsonl"
    old_dataset.parent.mkdir(parents=True)
    old_dataset.write_text('{"id":"current-run-row"}\n', encoding="utf-8")
    previous_adapter = cloud_root.parent / "adapter_run/models/qwen25_gp4_lora"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        allow_mixed_prior_artifacts=True,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    assert report["previous_run"]["old_dataset_run_id"] == "run"
    assert report["previous_run"]["previous_adapter_run_id"] == "adapter_run"
    assert report["previous_run"]["mixed_prior_artifacts"] is True
    assert report["previous_run"]["matched"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "previous_reuse_same_prior_run" in failed


def test_colab_readiness_report_rejects_previous_adapter_outside_cloud_root(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = tmp_path / "local_previous_adapter"
    _adapter(previous_adapter)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=False),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "previous_adapter_allowed_cloud_path" in failed


def test_colab_readiness_report_fails_without_drive_account_confirmation(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    (cloud_root / "manifests/drive_account_confirmation.json").unlink()

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "drive_account_confirmation_matches" in failed


def test_colab_readiness_report_fails_when_old_dataset_missing(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    old_dataset.unlink()

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "old_dataset_exists" in failed
    assert report["install_action_performed"] is False


def test_colab_readiness_report_fails_when_gp4_ws_commit_mismatches(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, _expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit="deadbeef",
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "gp4_ws_expected_commit_matches" in failed


def test_colab_readiness_report_rejects_gp4_ws_outside_cloud_root(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, _gp4_ws, _expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    outside_gp4_ws = tmp_path / "local_gp4_ws"
    expected_commit = _target_repo(outside_gp4_ws)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=outside_gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=False),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "gp4_ws_allowed_cloud_path" in failed


def test_colab_readiness_report_rejects_gp4_ws_from_sibling_run(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, _gp4_ws, _expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    sibling_gp4_ws = (
        cloud_root.parent
        / "sibling_run/contract_snapshots/gp4_ws_ws-deep-rebuild-3526"
    )
    expected_commit = _target_repo(sibling_gp4_ws)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=sibling_gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "gp4_ws_allowed_cloud_path" in failed


def test_colab_readiness_report_fails_when_gp4_ws_branch_is_wrong(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, _gp4_ws, _expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    wrong_branch_gp4_ws = cloud_root / "contract_snapshots/wrong_branch_gp4_ws"
    expected_commit = _target_repo(wrong_branch_gp4_ws, branch="main")

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=wrong_branch_gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "gp4_ws_expected_branch" in failed


def test_colab_readiness_report_fails_when_gp4_ws_snapshot_is_dirty(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    (gp4_ws / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit,
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "gp4_ws_clean" in failed


def test_colab_readiness_report_fails_when_expected_commit_is_too_short(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)

    report = build_colab_readiness_report(
        cloud_root=cloud_root,
        run_id="run",
        gp4_ws=gp4_ws,
        old_dataset=old_dataset,
        previous_adapter=previous_adapter,
        expected_commit=expected_commit[:7],
        policy=CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=True),
    )

    assert report["passed"] is False
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "gp4_ws_expected_commit_matches" in failed


def test_colab_readiness_cli_writes_report_under_cloud_root(tmp_path: Path) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = _previous_adapter(cloud_root)
    report_path = cloud_root / "reports/colab_readiness_run.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/colab_readiness_report.py",
            "--cloud-root",
            str(cloud_root),
            "--run-id",
            "run",
            "--gp4-ws",
            str(gp4_ws),
            "--old-dataset",
            str(old_dataset),
            "--previous-adapter",
            str(previous_adapter),
            "--expected-commit",
            expected_commit,
            "--report",
            str(report_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["report"] == str(report_path)


def test_colab_readiness_cli_allows_explicit_mixed_prior_artifacts(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    previous_adapter = cloud_root.parent / "adapter_run/models/qwen25_gp4_lora"
    _adapter(previous_adapter)
    report_path = cloud_root / "reports/colab_readiness_run.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/colab_readiness_report.py",
            "--cloud-root",
            str(cloud_root),
            "--run-id",
            "run",
            "--gp4-ws",
            str(gp4_ws),
            "--old-dataset",
            str(old_dataset),
            "--previous-adapter",
            str(previous_adapter),
            "--expected-commit",
            expected_commit,
            "--report",
            str(report_path),
            "--allow-mixed-prior-artifacts",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["previous_run"]["matched"] is True
    assert payload["previous_run"]["mixed_prior_artifacts"] is True


def test_colab_readiness_cli_allows_base_model_start_without_previous_adapter(
    tmp_path: Path,
) -> None:
    cloud_root, old_dataset, gp4_ws, expected_commit = _cloud_inputs(tmp_path)
    report_path = cloud_root / "reports/colab_readiness_run.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/colab_readiness_report.py",
            "--cloud-root",
            str(cloud_root),
            "--run-id",
            "run",
            "--gp4-ws",
            str(gp4_ws),
            "--old-dataset",
            str(old_dataset),
            "--expected-commit",
            expected_commit,
            "--report",
            str(report_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["previous_adapter"]["base_model_start"] is True
    assert payload["previous_run"]["matched"] is True
