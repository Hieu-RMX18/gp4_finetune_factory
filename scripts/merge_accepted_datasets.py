#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cloud_runtime import CloudPathError, validate_cloud_run_paths
from dataset_keys import dataset_identity_key
from factory_common import read_jsonl, read_yaml, write_json, write_jsonl
from v2_taxonomy import distribution_summary, quota_failures, row_scenario_tags


SourceRow = tuple[str, dict[str, Any]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge old and new accepted GP4 rows.")
    parser.add_argument("--old", action="append", default=[])
    parser.add_argument("--new", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, required=True)
    parser.add_argument("--distribution-spec", type=Path)
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    input_paths = [Path(path) for path in [*args.old, *args.new]]
    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=False,
            inputs=input_paths,
            outputs=[args.output, args.report],
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"merge_blocked reason={exc} report={args.report}")
        return 1

    old_rows = _read_many([Path(path) for path in args.old])
    new_rows = _read_many([Path(path) for path in args.new])
    quota_minimums, max_legacy_rows = _distribution_policy(args.distribution_spec)
    selected, dropped_duplicates, quota_preserved = select_rows(
        old_rows,
        new_rows,
        args.target_rows,
        quota_minimums=quota_minimums,
        max_legacy_rows_without_scenario_tags=max_legacy_rows,
    )
    selected_rows = [row for _, row in selected]
    passed = len(selected_rows) == args.target_rows
    selected_distribution = distribution_summary(selected_rows)
    payload = {
        "passed": passed,
        "blocked_reason": "" if passed else "not enough unique accepted rows",
        "target_rows": args.target_rows,
        "output_rows": len(selected_rows),
        "old_rows_input": len(old_rows),
        "new_rows_input": len(new_rows),
        "old_rows_kept": sum(1 for source, _ in selected if source == "old"),
        "new_rows_kept": sum(1 for source, _ in selected if source == "new"),
        "dropped_duplicates": dropped_duplicates,
        "quota_preserved": quota_preserved,
        "distribution": selected_distribution,
        "quota_failures": quota_failures(
            selected_rows,
            _quota_gates(quota_minimums, max_legacy_rows),
        ),
    }
    write_json(args.report, payload)
    if passed:
        write_jsonl(args.output, selected_rows)
    print(f"passed={passed} output_rows={len(selected_rows)} report={args.report}")
    return 0 if passed else 1


def select_rows(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    target_rows: int,
    *,
    quota_minimums: dict[str, int] | None = None,
    max_legacy_rows_without_scenario_tags: int | None = None,
) -> tuple[list[SourceRow], int, bool]:
    selected: list[SourceRow] = []
    selected_keys: set[str] = set()
    seen: set[str] = set()
    dropped_duplicates = 0
    candidates: list[SourceRow] = [
        *[("old", row) for row in old_rows],
        *[("new", row) for row in new_rows],
    ]
    unique_candidates: list[SourceRow] = []
    for source, row in candidates:
        key = dataset_identity_key(row)
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)
        unique_candidates.append((source, row))

    quota_minimums = quota_minimums or {}
    tag_candidates = _tag_candidates(unique_candidates, quota_minimums)
    tag_offsets = {tag: 0 for tag in quota_minimums}
    for tag, minimum in quota_minimums.items():
        while _selected_tag_count(selected, tag) < int(minimum):
            if len(selected) >= target_rows:
                break
            candidate, tag_offsets[tag] = _next_unselected_with_tag(
                tag_candidates.get(tag, []),
                selected_keys,
                start=tag_offsets[tag],
            )
            if candidate is None:
                break
            _append_selected(selected, selected_keys, candidate)

    if max_legacy_rows_without_scenario_tags is None:
        _fill_selected(selected, selected_keys, unique_candidates, target_rows)
    else:
        _fill_selected(
            selected,
            selected_keys,
            unique_candidates,
            target_rows,
            predicate=lambda candidate: not _is_legacy_without_scenario_tags(
                candidate[1]
            ),
        )
        _fill_selected(
            selected,
            selected_keys,
            unique_candidates,
            target_rows,
            max_legacy_rows_without_scenario_tags=max_legacy_rows_without_scenario_tags,
        )
        _fill_selected(selected, selected_keys, unique_candidates, target_rows)

    quota_preserved = not quota_failures(
        [row for _, row in selected],
        _quota_gates(quota_minimums, max_legacy_rows_without_scenario_tags),
    )
    return selected, dropped_duplicates, quota_preserved


def _fill_selected(
    selected: list[SourceRow],
    selected_keys: set[str],
    candidates: list[SourceRow],
    target_rows: int,
    *,
    predicate: Any | None = None,
    max_legacy_rows_without_scenario_tags: int | None = None,
) -> None:
    legacy_rows_selected = _selected_legacy_without_scenario_tags(selected)
    for candidate in candidates:
        if len(selected) >= target_rows:
            return
        if dataset_identity_key(candidate[1]) in selected_keys:
            continue
        if predicate is not None and not predicate(candidate):
            continue
        if max_legacy_rows_without_scenario_tags is not None:
            if not _is_legacy_without_scenario_tags(candidate[1]):
                continue
            if legacy_rows_selected >= max_legacy_rows_without_scenario_tags:
                return
        _append_selected(selected, selected_keys, candidate)
        if max_legacy_rows_without_scenario_tags is not None:
            legacy_rows_selected += 1


def _append_selected(
    selected: list[SourceRow],
    selected_keys: set[str],
    candidate: SourceRow,
) -> None:
    selected.append(candidate)
    selected_keys.add(dataset_identity_key(candidate[1]))


def _tag_candidates(
    candidates: list[SourceRow],
    quota_minimums: dict[str, int],
) -> dict[str, list[SourceRow]]:
    wanted_tags = set(quota_minimums)
    indexed = {tag: [] for tag in wanted_tags}
    if not wanted_tags:
        return indexed
    for candidate in candidates:
        for tag in row_scenario_tags(candidate[1]):
            if tag in indexed:
                indexed[tag].append(candidate)
    return indexed


def _next_unselected_with_tag(
    candidates: list[SourceRow],
    selected_keys: set[str],
    *,
    start: int,
) -> tuple[SourceRow | None, int]:
    for index in range(start, len(candidates)):
        candidate = candidates[index]
        key = dataset_identity_key(candidate[1])
        if key in selected_keys:
            continue
        return candidate, index + 1
    return None, len(candidates)


def _selected_tag_count(selected: list[SourceRow], tag: str) -> int:
    return sum(1 for _, row in selected if tag in row_scenario_tags(row))


def _selected_legacy_without_scenario_tags(selected: list[SourceRow]) -> int:
    return sum(1 for _, row in selected if _is_legacy_without_scenario_tags(row))


def _is_legacy_without_scenario_tags(row: dict[str, Any]) -> bool:
    if row_scenario_tags(row):
        return False
    metadata = row.get("metadata")
    return not (isinstance(metadata, dict) and metadata.get("source_dataset") == "new")


def _distribution_policy(spec_path: Path | None) -> tuple[dict[str, int], int | None]:
    if spec_path is None:
        return {}, None
    spec = read_yaml(spec_path)
    gates = spec.get("v2_distribution_gates", {})
    if not isinstance(gates, dict):
        return {}, None
    minimums = gates.get("scenario_tag_min_counts", {})
    if not isinstance(minimums, dict):
        minimums = {}
    max_legacy = gates.get("max_legacy_rows_without_scenario_tags")
    return (
        {str(tag): int(value) for tag, value in minimums.items()},
        int(max_legacy) if max_legacy is not None else None,
    )


def _quota_gates(
    quota_minimums: dict[str, int] | None,
    max_legacy_rows_without_scenario_tags: int | None,
) -> dict[str, Any]:
    gates: dict[str, Any] = {"scenario_tag_min_counts": quota_minimums or {}}
    if max_legacy_rows_without_scenario_tags is not None:
        gates["max_legacy_rows_without_scenario_tags"] = (
            max_legacy_rows_without_scenario_tags
        )
    return gates


def _read_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
