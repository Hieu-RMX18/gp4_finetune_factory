#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from factory_common import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export local master JSONL to Unsloth/TRL chat JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    exported = [{"messages": row["messages"]} for row in rows]
    write_jsonl(args.output, exported)
    print(f"rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
