#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from factory_common import DEFAULT_GP4_WS, load_repo_contract, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the GP4 Semantic IR and primitive contract from gp4_ws."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_GP4_WS)
    parser.add_argument("--output", type=Path, default=Path("reports/repo_contract.json"))
    args = parser.parse_args()

    contract = load_repo_contract(args.repo)
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
