import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from path_config import resolve_gp4_ws


def test_resolve_gp4_ws_uses_cli_value_first(tmp_path: Path) -> None:
    cli_path = tmp_path / "gp4_ws_cli"
    env_path = tmp_path / "gp4_ws_env"

    assert resolve_gp4_ws(cli_value=str(cli_path), env={"GP4_WS": str(env_path)}) == cli_path


def test_resolve_gp4_ws_uses_environment_when_cli_missing(tmp_path: Path) -> None:
    env_path = tmp_path / "gp4_ws_env"

    assert resolve_gp4_ws(cli_value=None, env={"GP4_WS": str(env_path)}) == env_path


def test_resolve_gp4_ws_strips_whitespace_after_selection(tmp_path: Path) -> None:
    env_path = tmp_path / "gp4_ws_env"

    assert resolve_gp4_ws(cli_value=None, env={"GP4_WS": f"  {env_path}  "}) == env_path.resolve()


def test_resolve_gp4_ws_expands_user_and_resolves_relative_paths(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    relative_target = tmp_path / "relative" / "gp4_ws"
    relative_target.parent.mkdir(parents=True)
    expected_home = (home_dir / "gp4_ws").resolve()
    expected_relative = relative_target.resolve()

    original_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home_dir)
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert resolve_gp4_ws(cli_value="~/gp4_ws", env={}) == expected_home
        assert resolve_gp4_ws(cli_value="relative/gp4_ws", env={}) == expected_relative
    finally:
        os.chdir(previous_cwd)
        if original_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = original_home


def test_resolve_gp4_ws_requires_keyword_arguments() -> None:
    try:
        resolve_gp4_ws("/tmp/gp4_ws", {})
    except TypeError:
        pass
    else:
        raise AssertionError("resolve_gp4_ws should reject positional arguments")


def test_resolve_gp4_ws_fails_loud_when_missing() -> None:
    try:
        resolve_gp4_ws(cli_value=None, env={})
    except ValueError as exc:
        assert str(exc) == "GP4_WS is required; set the environment variable or pass --gp4-ws"
    else:
        raise AssertionError("resolve_gp4_ws should fail when no source workspace is configured")


def test_make_contract_requires_gp4_ws_without_default() -> None:
    env = os.environ.copy()
    env.pop("GP4_WS", None)
    result = subprocess.run(
        ["make", "require-gp4-ws"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode in {1, 2}
    assert "GP4_WS is required" in result.stdout
