#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from cloud_runtime import CloudPathError, validate_cloud_run_paths
from dataset_keys import dataset_identity_key
from factory_common import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic train/val/test splits from validated JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--locked-eval", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    outputs = [
        args.output_dir / "train.jsonl",
        args.output_dir / "val.jsonl",
        args.output_dir / "test.jsonl",
    ]
    if args.report:
        outputs.append(args.report)
    inputs = [args.input]
    if args.locked_eval:
        inputs.append(args.locked_eval)
    try:
        validate_cloud_run_paths(
            cloud_root=args.cloud_root,
            dry_run=False,
            inputs=inputs,
            outputs=outputs,
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(f"blocked_reason={exc}")
        return 1

    rows = read_jsonl(args.input)
    if args.locked_eval:
        contamination = _find_locked_eval_contamination(rows, read_jsonl(args.locked_eval))
        if contamination:
            print(f"locked eval contamination: {contamination[0]}")
            return 1
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must leave room for test rows")

    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    train_count = int(len(shuffled) * args.train_ratio)
    val_count = int(len(shuffled) * args.val_ratio)
    if shuffled and train_count == 0:
        train_count = 1
    if len(shuffled) >= 3 and val_count == 0:
        val_count = 1
    if train_count + val_count >= len(shuffled) and len(shuffled) >= 2:
        train_count = max(1, len(shuffled) - 2)
        val_count = 1

    train_rows = shuffled[:train_count]
    val_rows = shuffled[train_count : train_count + val_count]
    test_rows = shuffled[train_count + val_count :]
    _ensure_train_label_coverage(train_rows, val_rows, test_rows)

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    if args.report:
        from factory_common import write_json

        write_json(
            args.report,
            {
                "rows": len(rows),
                "train": len(train_rows),
                "validation": len(val_rows),
                "test": len(test_rows),
                "locked_eval_contamination": 0,
                "passed": True,
            },
        )
    print(
        f"rows={len(rows)} train={len(train_rows)} val={len(val_rows)} "
        f"test={len(test_rows)} output_dir={args.output_dir}"
    )
    return 0


def _ensure_train_label_coverage(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    labels = {_expected_label(row) for row in [*train_rows, *val_rows, *test_rows]}
    if len(labels) > len(train_rows):
        raise ValueError(
            f"train split has {len(train_rows)} rows but needs {len(labels)} "
            "rows to cover every expected intent/error label"
        )

    train_counts = Counter(_expected_label(row) for row in train_rows)
    for label in sorted(labels):
        if train_counts[label]:
            continue

        source_rows, source_index = _find_row_with_label(val_rows, test_rows, label)
        donor_index = _find_swappable_train_index(train_rows, train_counts)
        donor_label = _expected_label(train_rows[donor_index])

        train_rows[donor_index], source_rows[source_index] = (
            source_rows[source_index],
            train_rows[donor_index],
        )
        train_counts[label] += 1
        train_counts[donor_label] -= 1


def _find_row_with_label(
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    label: tuple[str, str],
) -> tuple[list[dict[str, Any]], int]:
    for rows in (val_rows, test_rows):
        for index, row in enumerate(rows):
            if _expected_label(row) == label:
                return rows, index
    raise ValueError(f"missing expected label after split: {label[0]}={label[1]}")


def _find_swappable_train_index(
    train_rows: list[dict[str, Any]],
    train_counts: Counter[tuple[str, str]],
) -> int:
    for index, row in enumerate(train_rows):
        if train_counts[_expected_label(row)] > 1:
            return index
    raise ValueError("could not preserve train coverage for every expected label")


def _expected_label(row: dict[str, Any]) -> tuple[str, str]:
    expected = row.get("expected_json", {})
    if not isinstance(expected, dict):
        return ("unknown", str(row.get("id", "")))
    intent = expected.get("intent")
    if isinstance(intent, str) and intent:
        return ("intent", intent)
    error = expected.get("error")
    if isinstance(error, str) and error:
        return ("error", error)
    return ("unknown", str(row.get("id", "")))

def _find_locked_eval_contamination(
    rows: list[dict[str, Any]],
    locked_eval_rows: list[dict[str, Any]],
) -> list[str]:
    locked_keys = {_contamination_key(row) for row in locked_eval_rows}
    return [
        str(row.get("id", "<missing-id>"))
        for row in rows
        if _contamination_key(row) in locked_keys
    ]

def _contamination_key(row: dict[str, Any]) -> str:
    return dataset_identity_key(row)


if __name__ == "__main__":
    raise SystemExit(main())
