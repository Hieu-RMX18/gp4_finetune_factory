#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping


def resolve_gp4_ws(
    *,
    cli_value: str | None,
    env: Mapping[str, str] = os.environ,
) -> Path:
    raw_value = (cli_value or env.get("GP4_WS") or "").strip()
    if not raw_value:
        raise ValueError(
            "GP4_WS is required; set the environment variable or pass --gp4-ws"
        )
    return Path(raw_value).expanduser().resolve(strict=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the GP4 source workspace path from CLI or environment."
    )
    parser.add_argument("--gp4-ws", default=None)
    parser.add_argument("--print", dest="should_print", action="store_true")
    args = parser.parse_args()

    try:
        gp4_ws = resolve_gp4_ws(cli_value=args.gp4_ws, env=os.environ)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.should_print:
        print(gp4_ws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
