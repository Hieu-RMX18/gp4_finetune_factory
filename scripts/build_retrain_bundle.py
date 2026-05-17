#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from cloud_runtime import validate_cloud_run_paths

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_NAME = "gp4_finetune_factory_source_bundle.zip"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o644 << 16

BUNDLE_PATTERNS = (
    "Makefile",
    "README.md",
    "model_card.md",
    "requirements.txt",
    "configs/dataset_spec.yaml",
    "configs/*.yaml",
    "data/seed/*.jsonl",
    "docs/superpowers/plans/*.md",
    "docs/superpowers/specs/*.md",
    "notebooks/colab_gp4_react_qwen25_qlora.ipynb",
    "notebooks/kaggle_gp4_react_qwen25_qlora.ipynb",
    "specs/*.yaml",
    "schemas/*.json",
    "scripts/*.py",
    "tests/*.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic cloud source ZIP for the GP4 retrain run."
    )
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    paths = _collect_bundle_paths(ROOT)
    output = args.output or args.cloud_root / "bundles" / DEFAULT_BUNDLE_NAME
    validate_cloud_run_paths(
        cloud_root=args.cloud_root,
        dry_run=False,
        inputs=paths,
        outputs=[output],
        allow_tmp=args.allow_tmp,
    )
    _write_bundle(output, paths)
    digest = _sha256(output)

    print(f"path={output} bytes={output.stat().st_size} sha256={digest}")
    return 0


def _collect_bundle_paths(root: Path) -> list[Path]:
    paths_by_name: dict[str, Path] = {}
    missing_patterns: list[str] = []

    for pattern in BUNDLE_PATTERNS:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            missing_patterns.append(pattern)
            continue
        for path in matches:
            paths_by_name[path.relative_to(root).as_posix()] = path

    if missing_patterns:
        raise FileNotFoundError(
            "missing bundle input pattern(s): " + ", ".join(missing_patterns)
        )

    return [paths_by_name[name] for name in sorted(paths_by_name)]


def _write_bundle(output_path: Path, paths: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in paths:
            archive_name = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(archive_name, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = ZIP_FILE_MODE
            archive.writestr(info, path.read_bytes())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
