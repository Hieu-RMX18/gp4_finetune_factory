from pathlib import Path
import subprocess
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

def test_cloud_storage_policy_cli_requires_cloud_report_path(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    report = tmp_path / "outside" / "policy_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_cloud_storage_policy.py",
            "--cloud-root",
            str(cloud_root),
            "--path",
            str(cloud_root / "reports/ok.json"),
            "--report",
            str(report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "outside CLOUD_ROOT" in (result.stdout + result.stderr)
    assert not report.exists()
