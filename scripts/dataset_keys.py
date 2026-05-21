from __future__ import annotations

import json
from typing import Any


def dataset_identity_key(row: dict[str, Any]) -> str:
    user_content = ""
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            user_content = str(message.get("content", "")).strip().lower()
            break
    target = json.dumps(
        row.get("expected_json", {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{user_content}\n{target}"
