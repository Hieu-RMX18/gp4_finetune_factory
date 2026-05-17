#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export local master JSONL to Unsloth/TRL chat JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=False,
            inputs=[args.input],
            outputs=[args.output],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"export_blocked reason={exc} output={args.output}")
        return 1

    rows = read_jsonl(args.input)
    exported = [{"messages": row["messages"]} for row in rows]
    write_jsonl(args.output, exported)
    print(f"rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
