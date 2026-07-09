#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cloud_runtime import CloudPathError, validate_cloud_output_path
from factory_common import load_repo_contract, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the GP4 Semantic IR and primitive contract from gp4_ws."
    )
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/repo_contract.json"))
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    try:
        validate_cloud_output_path(
            args.output,
            args.cloud_root,
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"contract_blocked reason={exc} output={args.output}")
        return 1

    try:
        contract = load_repo_contract(args.repo)
    except ValueError as exc:
        print(f"contract_blocked reason={exc} output={args.output}")
        return 1
    write_json(args.output, contract)
    print(
        "contract extracted: "
        f"intents={len(contract['semantic_intents'])} "
        f"top_level={len(contract['top_level_output_intents'])} "
        f"schema_primitives={len(contract['schema_primitives'])} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
