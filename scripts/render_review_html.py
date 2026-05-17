#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from cloud_runtime import CloudPathError, validate_cloud_run_paths
from factory_common import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a static HTML review table for GP4 JSONL examples."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/review.html"))
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
        print(f"review_blocked reason={exc} output={args.output}")
        return 1

    rows = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render(rows), encoding="utf-8")
    print(f"rows={len(rows)} output={args.output}")
    return 0


def _render(rows: list[dict]) -> str:
    table_rows = "\n".join(_render_row(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GP4 Fine-Tune Dataset Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccd0d5; padding: 8px; vertical-align: top; }}
    th {{ background: #f2f4f7; text-align: left; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>GP4 Fine-Tune Dataset Review</h1>
  <table>
    <thead>
      <tr><th>ID</th><th>User</th><th>Assistant JSON</th><th>Metadata</th></tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
</body>
</html>
"""


def _render_row(row: dict) -> str:
    user_text = ""
    assistant_text = ""
    for message in row.get("messages", []):
        if message.get("role") == "user":
            user_text = message.get("content", "")
        if message.get("role") == "assistant":
            assistant_text = message.get("content", "")
    metadata = json.dumps(row.get("metadata", {}), ensure_ascii=False, indent=2)
    return (
        "      <tr>"
        f"<td>{html.escape(str(row.get('id', '')))}</td>"
        f"<td><pre>{html.escape(str(user_text))}</pre></td>"
        f"<td><pre>{html.escape(str(assistant_text))}</pre></td>"
        f"<td><pre>{html.escape(metadata)}</pre></td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
