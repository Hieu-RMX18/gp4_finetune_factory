from __future__ import annotations

from collections import Counter
from typing import Any


REQUIRED_SCENARIO_TAGS = {
    "singularity",
    "wrist_flip",
    "joint_wrap",
    "timeout_abort_recovery",
    "approval_required",
    "collision_limit_edge",
    "dangerous_os_command",
    "unsupported_tool_hallucination",
}
OPTIONAL_SCENARIO_TAGS = {
    "typo_noise",
    "alias_canonicalization",
    "vision_uncertain",
    "normal_motion",
    "status_query",
}
ALLOWED_SCENARIO_TAGS = REQUIRED_SCENARIO_TAGS | OPTIONAL_SCENARIO_TAGS
MISSING_NEW_SCENARIO_TAGS = "<missing-new-scenario-tags>"
NON_LIST_SCENARIO_TAGS = "<non-list-scenario-tags>"


def row_scenario_tags(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return []
    scenario_tags = metadata.get("scenario_tags")
    if scenario_tags is None:
        return []
    if not isinstance(scenario_tags, list):
        return [NON_LIST_SCENARIO_TAGS]
    return [str(tag) for tag in scenario_tags]


def invalid_scenario_tags(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata")
    scenario_tags = row_scenario_tags(row)
    if scenario_tags == [NON_LIST_SCENARIO_TAGS]:
        return scenario_tags
    if isinstance(metadata, dict) and metadata.get("source_dataset") == "new" and not scenario_tags:
        return [MISSING_NEW_SCENARIO_TAGS]
    return sorted(tag for tag in scenario_tags if tag not in ALLOWED_SCENARIO_TAGS)


def distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    legacy_rows = 0
    for row in rows:
        metadata = row.get("metadata")
        scenario_tags = row_scenario_tags(row)
        if scenario_tags == []:
            if not (isinstance(metadata, dict) and metadata.get("source_dataset") == "new"):
                legacy_rows += 1
            continue
        if scenario_tags == [NON_LIST_SCENARIO_TAGS]:
            continue
        counts.update(scenario_tags)

    return {
        "rows": len(rows),
        "legacy_rows_without_scenario_tags": legacy_rows,
        "scenario_tags": dict(sorted(counts.items())),
    }


def quota_failures(rows: list[dict[str, Any]], gates: dict[str, Any]) -> list[dict[str, Any]]:
    distribution = distribution_summary(rows)
    counts = distribution["scenario_tags"]
    minimums = gates.get("scenario_tag_min_counts", {})
    failures: list[dict[str, Any]] = []
    for tag, minimum in sorted(minimums.items()):
        actual = int(counts.get(tag, 0))
        minimum = int(minimum)
        if actual >= minimum:
            continue
        failures.append(
            {
                "tag": tag,
                "actual": actual,
                "minimum": minimum,
                "message": f"scenario tag {tag} has {actual} rows but requires {minimum}",
            }
        )
    if "max_legacy_rows_without_scenario_tags" in gates:
        maximum = int(gates["max_legacy_rows_without_scenario_tags"])
        actual = int(distribution["legacy_rows_without_scenario_tags"])
        if actual > maximum:
            failures.append(
                {
                    "tag": "__legacy_without_scenario_tags__",
                    "actual": actual,
                    "maximum": maximum,
                    "message": (
                        "legacy rows without scenario tags has "
                        f"{actual} rows but allows at most {maximum}"
                    ),
                }
            )
    return failures
