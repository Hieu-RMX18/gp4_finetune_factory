from pathlib import Path
import subprocess
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

def test_probe_from_environment_blocks_account_creation_automation(
    tmp_path: Path,
) -> None:
    result = probe_from_environment(
        {
            "HOME": "/root",
            "ACCOUNT_CREATION_AUTOMATION": "1",
        },
        cloud_root=str(tmp_path),
    )

    assert result.is_usable is False
    assert result.account_creation_automation is True
    assert result.blocked_reason == "account creation automation is forbidden"

def test_provider_probe_cli_requires_cloud_report_path(tmp_path: Path) -> None:
    cloud_root = tmp_path / "cloud"
    cloud_root.mkdir()
    report = tmp_path / "outside" / "platform_status.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/provider_probe.py",
            "--provider",
            "colab",
            "--cloud-root",
            str(cloud_root),
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
    assert "outside CLOUD_ROOT" in result.stdout
    assert not report.exists()
