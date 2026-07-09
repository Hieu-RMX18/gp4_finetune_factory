from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from factory_common import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_LOCAL_INPUT_DIRS = (
    ROOT / "artifact_downloads",
    ROOT / "models",
    ROOT / "reports",
    ROOT / "outputs",
    ROOT / "data/generated",
    ROOT / "data/validated",
    ROOT / "data/splits",
)

ALLOWED_LOCAL_SOURCE_DIRS = (
    ROOT / "scripts",
    ROOT / "configs",
    ROOT / "schemas",
    ROOT / "specs",
    ROOT / "notebooks",
    ROOT / "docs",
    ROOT / "tests",
    ROOT / "data/seed",
)

ALLOWED_LOCAL_SOURCE_FILES = {
    ROOT / "Makefile",
    ROOT / "README.md",
    ROOT / "model_card.md",
    ROOT / "requirements.txt",
    ROOT / "requirements-cloud.txt",
    ROOT / "requirements-local-adapter.txt",
}


class CloudPathError(ValueError):
    pass


@dataclass(frozen=True)
class CopiedSourcePlan:
    path: Path
    sha256: str


def validate_cloud_run_paths(
    *,
    cloud_root: Path | str | None,
    dry_run: bool,
    inputs: Iterable[Path | str],
    outputs: Iterable[Path | str],
    allow_tmp: bool = False,
) -> Path | None:
    if dry_run:
        return _normalize_cloud_root(cloud_root, allow_tmp=allow_tmp) if cloud_root else None

    root = require_cloud_root(cloud_root, allow_tmp=allow_tmp)
    for output in outputs:
        validate_cloud_output_path(output, root, allow_tmp=allow_tmp)
    for input_path in inputs:
        validate_cloud_input_path(input_path, root, allow_tmp=allow_tmp)
    return root


def require_cloud_root(
    cloud_root: Path | str | None,
    *,
    allow_tmp: bool = False,
) -> Path:
    if not cloud_root:
        raise CloudPathError("CLOUD_ROOT is required for non-dry-run cloud phases.")
    root = _normalize_cloud_root(cloud_root, allow_tmp=allow_tmp)
    if _is_under_tmp(root) and not allow_tmp:
        raise CloudPathError("CLOUD_ROOT under /tmp is only allowed with --allow-tmp.")
    return root


def validate_cloud_output_path(
    path: Path | str,
    cloud_root: Path | str,
    *,
    allow_tmp: bool = False,
) -> Path:
    root = require_cloud_root(cloud_root, allow_tmp=allow_tmp)
    resolved = _resolve_path(path)
    _raise_if_symlink_escape(Path(path))
    if not _is_relative_to(resolved, root):
        raise CloudPathError(f"output path is outside CLOUD_ROOT: {path}")
    return resolved


def validate_cloud_input_path(
    path: Path | str,
    cloud_root: Path | str,
    *,
    allow_tmp: bool = False,
) -> Path:
    root = require_cloud_root(cloud_root, allow_tmp=allow_tmp)
    resolved = _resolve_path(path)
    _raise_if_symlink_escape(Path(path))
    if _is_relative_to(resolved, root):
        return resolved
    if _is_forbidden_local_runtime_input(resolved):
        raise CloudPathError(f"forbidden local runtime artifact input: {path}")
    if _is_allowed_local_source_input(resolved):
        return resolved
    raise CloudPathError(f"input path is not a cloud path or allowed source input: {path}")


def configure_cloud_caches(
    *,
    cloud_root: Path | str | None,
    dry_run: bool,
    allow_tmp: bool = False,
) -> dict[str, str]:
    if dry_run:
        return {}
    root = require_cloud_root(cloud_root, allow_tmp=allow_tmp)
    cache_env = {
        "HF_HOME": root / ".cache/huggingface",
        "TRANSFORMERS_CACHE": root / ".cache/huggingface/transformers",
        "HF_DATASETS_CACHE": root / ".cache/huggingface/datasets",
        "TORCH_HOME": root / ".cache/torch",
        "XDG_CACHE_HOME": root / ".cache",
        "WANDB_DIR": root / "wandb",
        "TMPDIR": root / "tmp",
    }
    for key, value in cache_env.items():
        validate_cloud_output_path(value, root, allow_tmp=allow_tmp)
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    return {key: str(value) for key, value in cache_env.items()}


def copy_source_plan_to_cloud(
    *,
    source_plan: Path | str | None,
    cloud_root: Path | str,
    dry_run: bool,
    allow_tmp: bool = False,
) -> CopiedSourcePlan:
    root = require_cloud_root(cloud_root, allow_tmp=allow_tmp)
    destination = root / "manifests" / "source_plan.md"
    validate_cloud_output_path(destination, root, allow_tmp=allow_tmp)

    if dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_text("", encoding="utf-8")
        return CopiedSourcePlan(destination, sha256_file(destination))

    if source_plan:
        source = _resolve_path(source_plan)
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination.resolve(strict=False):
                shutil.copyfile(source, destination)
            return CopiedSourcePlan(destination, sha256_file(destination))

    if destination.exists():
        return CopiedSourcePlan(destination, sha256_file(destination))
    raise CloudPathError("source plan is missing and no cloud copy exists.")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_phase_report(
    *,
    cloud_root: Path | str,
    run_id: str,
    phase: str,
    payload: Mapping[str, object],
    allow_tmp: bool = False,
) -> Path:
    root = require_cloud_root(cloud_root, allow_tmp=allow_tmp)
    report_path = phase_report_path(root, run_id, phase)
    validate_cloud_output_path(report_path, root, allow_tmp=allow_tmp)
    write_json(report_path, dict(payload))
    return report_path


def phase_report_path(cloud_root: Path | str, run_id: str, phase: str) -> Path:
    safe_phase = phase.replace("/", "-")
    if safe_phase == "package":
        safe_phase = "package_report"
    return Path(cloud_root) / "reports" / f"{safe_phase}_{run_id}.json"


def report_passed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = read_json(path)
    except (OSError, ValueError):
        return False
    return report.get("passed") is True


def _normalize_cloud_root(path: Path | str | None, *, allow_tmp: bool) -> Path:
    if path is None:
        raise CloudPathError("CLOUD_ROOT is required for non-dry-run cloud phases.")
    resolved = _resolve_path(path)
    _raise_if_symlink_escape(Path(path))
    if _is_under_tmp(resolved) and not allow_tmp:
        raise CloudPathError("CLOUD_ROOT under /tmp is only allowed with --allow-tmp.")
    return resolved


def _resolve_path(path: Path | str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve(strict=False)


def _is_forbidden_local_runtime_input(path: Path) -> bool:
    return any(_is_relative_to(path, root.resolve(strict=False)) for root in FORBIDDEN_LOCAL_INPUT_DIRS)


def _is_allowed_local_source_input(path: Path) -> bool:
    if path in {item.resolve(strict=False) for item in ALLOWED_LOCAL_SOURCE_FILES}:
        return True
    return any(_is_relative_to(path, root.resolve(strict=False)) for root in ALLOWED_LOCAL_SOURCE_DIRS)


def _is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_under_tmp(path: Path) -> bool:
    return path == Path("/tmp") or Path("/tmp") in path.parents


def _raise_if_symlink_escape(path: Path) -> None:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise CloudPathError(f"path contains a symlink component: {path}")
