import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_local_install_manifest import build_manifest
from check_cloud_storage_policy import CloudStoragePolicy


def _adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")


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


def test_manifest_blocks_when_acceptance_report_failed(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    _adapter(adapter_dir)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": False}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=Path("/opt/gp4_ws"),
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "acceptance gate has not passed"
    assert manifest["install_action_performed"] is False


def test_manifest_passes_for_accepted_cloud_adapter(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    target_repo = cloud_root / "contract_snapshots/gp4_ws_target"
    _adapter(adapter_dir)
    expected_commit = _target_repo(target_repo)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=target_repo,
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
        expected_commit=expected_commit,
    )

    assert manifest["ready_for_local_install"] is True
    assert manifest["adapter"]["files"]["adapter_model.safetensors"]["bytes"] == 7
    assert manifest["target_repo"] == str(target_repo)
    assert manifest["target_repo_state"]["expected_branch"] == "ws-deep-rebuild-3526"
    assert manifest["target_repo_state"]["exists"] is True
    assert manifest["target_repo_state"]["current_branch"] == "ws-deep-rebuild-3526"
    assert manifest["target_repo_state"]["head"]
    assert manifest["target_repo_state"]["expected_commit"] == expected_commit
    assert manifest["target_repo_state"]["expected_commit_matches"] is True
    assert "requirements-local-adapter.txt" in manifest["dependency_manifests"]
    assert len(manifest["dependency_manifests"]["requirements-local-adapter.txt"]["sha256"]) == 64
    assert manifest["install_action_performed"] is False


def test_manifest_rejects_target_repo_outside_cloud_root(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    target_repo = tmp_path / "local_gp4_ws_target"
    _adapter(adapter_dir)
    expected_commit = _target_repo(target_repo)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=target_repo,
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=False),
        expected_commit=expected_commit,
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == (
        "target repo snapshot is not allowed by cloud storage policy"
    )
    assert manifest["install_action_performed"] is False


def test_manifest_blocks_when_target_repo_branch_is_wrong(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    target_repo = cloud_root / "contract_snapshots/gp4_ws_target"
    _adapter(adapter_dir)
    expected_commit = _target_repo(target_repo, branch="main")
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=target_repo,
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
        expected_commit=expected_commit,
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "target repo snapshot is not on the expected branch"
    assert manifest["target_repo_state"]["current_branch"] == "main"
    assert manifest["install_action_performed"] is False


def test_manifest_blocks_when_target_repo_snapshot_is_dirty(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    target_repo = cloud_root / "contract_snapshots/gp4_ws_target"
    _adapter(adapter_dir)
    expected_commit = _target_repo(target_repo)
    (target_repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=target_repo,
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
        expected_commit=expected_commit,
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "target repo snapshot is dirty"
    assert manifest["target_repo_state"]["is_dirty"] is True
    assert manifest["install_action_performed"] is False


def test_manifest_blocks_when_expected_commit_is_missing(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    target_repo = tmp_path / "gp4_ws_target"
    _adapter(adapter_dir)
    _target_repo(target_repo)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=target_repo,
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "expected gp4_ws commit is not configured"
    assert manifest["install_action_performed"] is False


def test_manifest_blocks_when_expected_commit_mismatches(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    target_repo = tmp_path / "gp4_ws_target"
    _adapter(adapter_dir)
    _target_repo(target_repo)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=target_repo,
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
        expected_commit="deadbeef",
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "target repo snapshot is not at the expected commit"
    assert manifest["target_repo_state"]["expected_commit"] == "deadbeef"
    assert manifest["target_repo_state"]["expected_commit_matches"] is False
    assert manifest["install_action_performed"] is False


def test_manifest_blocks_when_expected_commit_is_too_short(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    target_repo = tmp_path / "gp4_ws_target"
    _adapter(adapter_dir)
    expected_commit = _target_repo(target_repo)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=target_repo,
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
        expected_commit=expected_commit[:7],
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "target repo snapshot is not at the expected commit"
    assert manifest["target_repo_state"]["expected_commit_matches"] is False


def test_manifest_blocks_when_target_repo_is_missing(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    report = cloud_root / "reports/acceptance_gate_report_run.json"
    _adapter(adapter_dir)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    manifest = build_manifest(
        adapter_dir=adapter_dir,
        acceptance_report=report,
        target_repo=tmp_path / "missing_gp4_ws",
        policy=CloudStoragePolicy((cloud_root,), allow_tmp=True),
    )

    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "target repo snapshot is missing"
    assert manifest["install_action_performed"] is False


def test_cli_writes_manifest_without_touching_target_repo(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    adapter_dir = cloud_root / "models/qwen25_gp4_lora"
    acceptance_report = cloud_root / "reports/acceptance_gate_report_run.json"
    manifest_path = cloud_root / "reports/local_install_manifest_run.json"
    target_repo = tmp_path / "gp4_ws_target"
    _adapter(adapter_dir)
    acceptance_report.parent.mkdir(parents=True)
    acceptance_report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_local_install_manifest.py",
            "--adapter-dir",
            str(adapter_dir),
            "--acceptance-report",
            str(acceptance_report),
            "--target-repo",
            str(target_repo),
            "--cloud-root",
            str(cloud_root),
            "--output",
            str(manifest_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["ready_for_local_install"] is False
    assert manifest["blocked_reason"] == "target repo snapshot is missing"
    assert manifest["install_action_performed"] is False
    assert not target_repo.exists()
