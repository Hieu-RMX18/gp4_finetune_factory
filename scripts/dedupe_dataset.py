#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from factory_common import SEMANTIC_IR_SYSTEM_PROMPT, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deduplicate GP4 JSONL rows by user prompt and assistant target."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for input_path in args.input:
        rows.extend(read_jsonl(input_path))
    kept: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = _dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(_normalize_system_prompt(row))

    write_jsonl(args.output, kept)
    print(f"rows={len(rows)} kept={len(kept)} dropped={len(rows) - len(kept)} output={args.output}")
    return 0


def _dedupe_key(row: dict) -> str:
    messages = row.get("messages", [])
    user_content = ""
    assistant_content = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            user_content = str(message.get("content", "")).strip().lower()
        if message.get("role") == "assistant":
            assistant_content = str(message.get("content", "")).strip()
    normalized_target = json.dumps(
        row.get("expected_json", assistant_content),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{user_content}\n{normalized_target}"


def _normalize_system_prompt(row: dict) -> dict:
    messages = row.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return row
    normalized_messages = [
        (
            {**message, "content": SEMANTIC_IR_SYSTEM_PROMPT}
            if index == 0 and isinstance(message, dict) and message.get("role") == "system"
            else message
        )
        for index, message in enumerate(messages)
    ]
    return {**row, "messages": normalized_messages}


if __name__ == "__main__":
    raise SystemExit(main())
