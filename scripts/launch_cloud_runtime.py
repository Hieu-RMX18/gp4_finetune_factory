#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = Path("notebooks/colab_gp4_react_qwen25_qlora.ipynb")
BROWSER_HANDOFF_TIMEOUT_SECONDS = 3
CRASH_MARKERS = ("Trace/breakpoint trap", "core dumped")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open the approved GP4 ReAct-IR cloud notebook in Brave."
    )
    parser.add_argument("--repo-slug", default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--browser", default="brave-browser")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_slug = args.repo_slug or detect_github_repo_slug()
    branch = args.branch or current_branch()
    notebook_path = args.notebook
    if not (ROOT / notebook_path).is_file():
        print(f"blocked_reason=notebook not found: {notebook_path}")
        return 1

    url = build_colab_url(
        repo_slug=repo_slug,
        branch=branch,
        notebook_path=notebook_path,
    )
    command = [args.browser, "--new-tab", url]
    print(" ".join(command))
    if args.dry_run:
        return 0

    return launch_browser(command)


def build_colab_url(*, repo_slug: str, branch: str, notebook_path: Path) -> str:
    clean_path = notebook_path.as_posix().lstrip("/")
    return (
        "https://colab.research.google.com/github/"
        f"{repo_slug}/blob/{branch}/{clean_path}"
    )


def detect_github_repo_slug() -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("blocked_reason=unable to read git origin remote")
    slug = github_slug_from_remote(result.stdout.strip())
    if not slug:
        raise SystemExit("blocked_reason=origin remote is not a GitHub URL")
    return slug


def github_slug_from_remote(remote_url: str) -> str:
    if remote_url.startswith("git@github.com:"):
        slug = remote_url.removeprefix("git@github.com:")
    else:
        parsed = urlparse(remote_url)
        if parsed.netloc != "github.com":
            return ""
        slug = parsed.path.lstrip("/")
    return slug.removesuffix(".git")


def current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch:
        raise SystemExit("blocked_reason=unable to resolve current git branch")
    return branch


def launch_browser(command: list[str]) -> int:
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        print(f"browser launch failed: {exc}")
        return 1

    try:
        stdout, stderr = process.communicate(timeout=BROWSER_HANDOFF_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return 0

    combined_output = f"{stdout}\n{stderr}"
    crashed = any(marker in combined_output for marker in CRASH_MARKERS)
    if process.returncode != 0 or crashed:
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="")
        print("browser launch failed")
        return process.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
