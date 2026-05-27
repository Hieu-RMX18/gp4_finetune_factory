import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from launch_cloud_runtime import build_colab_url


def test_build_colab_url_targets_react_notebook_on_current_branch() -> None:
    url = build_colab_url(
        repo_slug="Hieu-RMX18/gp4_finetune_factory",
        branch="codex/gp4-react-ir-cloud-workflow",
        notebook_path=Path("notebooks/colab_gp4_react_qwen25_qlora.ipynb"),
    )

    assert url == (
        "https://colab.research.google.com/github/"
        "Hieu-RMX18/gp4_finetune_factory/blob/"
        "codex/gp4-react-ir-cloud-workflow/"
        "notebooks/colab_gp4_react_qwen25_qlora.ipynb"
    )


def test_launcher_dry_run_prints_browser_command_without_opening() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_cloud_runtime.py",
            "--repo-slug",
            "Hieu-RMX18/gp4_finetune_factory",
            "--branch",
            "codex/gp4-react-ir-cloud-workflow",
            "--browser",
            "brave-browser",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "brave-browser" in result.stdout
    assert "--profile-directory=Profile 26" in result.stdout
    assert "--remote-debugging-address=127.0.0.1" in result.stdout
    assert "--remote-debugging-port=9222" in result.stdout
    assert "colab_gp4_react_qwen25_qlora.ipynb" in result.stdout


def test_launcher_dry_run_allows_overriding_brave_profile_and_cdp_port() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_cloud_runtime.py",
            "--repo-slug",
            "Hieu-RMX18/gp4_finetune_factory",
            "--branch",
            "codex/gp4-react-ir-cloud-workflow",
            "--profile-directory",
            "Default",
            "--remote-debugging-port",
            "9333",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--profile-directory=Default" in result.stdout
    assert "--remote-debugging-port=9333" in result.stdout


def test_launcher_rejects_browser_wrapper_crash_even_with_zero_exit(
    tmp_path: Path,
) -> None:
    fake_browser = tmp_path / "fake-brave"
    fake_browser.write_text(
        "#!/usr/bin/env sh\n"
        "echo 'Trace/breakpoint trap (core dumped)' >&2\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_browser.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_cloud_runtime.py",
            "--repo-slug",
            "Hieu-RMX18/gp4_finetune_factory",
            "--branch",
            "codex/gp4-react-ir-cloud-workflow",
            "--browser",
            str(fake_browser),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "browser launch failed" in result.stdout
