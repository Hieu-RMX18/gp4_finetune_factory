import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _row(row_id: str, prompt: str, tag: str, source_dataset: str) -> dict:
    expected = {"intent": "stop"}
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": json.dumps(expected, separators=(",", ":")),
            },
        ],
        "expected_json": expected,
        "metadata": {
            "language": "vi",
            "task_type": "normal",
            "source": "synthetic",
            "source_dataset": source_dataset,
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
            "scenario_tags": [tag],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_merge_preserves_valid_old_rows_and_fills_to_target(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    output_path = tmp_path / "accepted_300k.jsonl"
    report_path = tmp_path / "merge_report.json"
    _write_jsonl(
        old_path,
        [_row("gp4_vi_old_000001", "dung robot", "normal_motion", "old")],
    )
    _write_jsonl(
        new_path,
        [
            _row("gp4_vi_new_000001", "dung robot 1", "singularity", "new"),
            _row("gp4_vi_new_000002", "dung robot 2", "wrist_flip", "new"),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_accepted_datasets.py",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--target-rows",
            "3",
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert [row["id"] for row in output_rows] == [
        "gp4_vi_old_000001",
        "gp4_vi_new_000001",
        "gp4_vi_new_000002",
    ]
    assert report["passed"] is True
    assert report["old_rows_kept"] == 1
    assert report["new_rows_kept"] == 2
    assert report["target_rows"] == 3


def test_merge_fails_when_not_enough_unique_rows(tmp_path: Path) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    output_path = tmp_path / "accepted_300k.jsonl"
    report_path = tmp_path / "merge_report.json"
    _write_jsonl(
        old_path,
        [_row("gp4_vi_old_000001", "dung robot", "normal_motion", "old")],
    )
    _write_jsonl(
        new_path,
        [_row("gp4_vi_new_000001", "dung robot", "normal_motion", "new")],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_accepted_datasets.py",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--target-rows",
            "2",
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert report["passed"] is False
    assert report["blocked_reason"] == "not enough unique accepted rows"


def test_merge_counts_old_inputs_by_origin_not_metadata(tmp_path: Path) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    output_path = tmp_path / "accepted_300k.jsonl"
    report_path = tmp_path / "merge_report.json"
    legacy_old_row = _row(
        "gp4_vi_old_000001",
        "dung robot",
        "normal_motion",
        "new",
    )
    legacy_old_row["metadata"].pop("source_dataset")
    _write_jsonl(old_path, [legacy_old_row])
    _write_jsonl(
        new_path,
        [_row("gp4_vi_new_000001", "dung robot 1", "singularity", "new")],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_accepted_datasets.py",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--target-rows",
            "2",
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["old_rows_kept"] == 1
    assert report["new_rows_kept"] == 1


def test_merge_counts_all_duplicates_even_after_target_is_full(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    output_path = tmp_path / "accepted_300k.jsonl"
    report_path = tmp_path / "merge_report.json"
    _write_jsonl(
        old_path,
        [
            _row("gp4_vi_old_000001", "dung robot", "normal_motion", "old"),
            _row("gp4_vi_old_000002", "dung robot", "normal_motion", "old"),
        ],
    )
    _write_jsonl(
        new_path,
        [
            _row("gp4_vi_new_000001", "dung robot 1", "singularity", "new"),
            _row("gp4_vi_new_000002", "dung robot 2", "wrist_flip", "new"),
            _row("gp4_vi_new_000003", "dung robot", "normal_motion", "new"),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_accepted_datasets.py",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--target-rows",
            "2",
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["output_rows"] == 2
    assert report["dropped_duplicates"] == 2


def test_merge_selects_quota_rows_before_stable_fill(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    output_path = tmp_path / "accepted_300k.jsonl"
    report_path = tmp_path / "merge_report.json"
    spec_path = tmp_path / "dataset_spec.yaml"
    _write_jsonl(
        old_path,
        [
            _row("gp4_vi_old_000001", "old normal 1", "normal_motion", "old"),
            _row("gp4_vi_old_000002", "old normal 2", "normal_motion", "old"),
            _row("gp4_vi_old_000003", "old normal 3", "normal_motion", "old"),
        ],
    )
    _write_jsonl(
        new_path,
        [
            _row("gp4_vi_new_000001", "new singularity", "singularity", "new"),
            _row("gp4_vi_new_000002", "new dangerous os", "dangerous_os_command", "new"),
        ],
    )
    spec_path.write_text(
        "v2_distribution_gates:\n"
        "  scenario_tag_min_counts:\n"
        "    singularity: 1\n"
        "    dangerous_os_command: 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_accepted_datasets.py",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--target-rows",
            "3",
            "--distribution-spec",
            str(spec_path),
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert [row["id"] for row in output_rows] == [
        "gp4_vi_new_000001",
        "gp4_vi_new_000002",
        "gp4_vi_old_000001",
    ]
    assert report["quota_preserved"] is True

def test_merge_respects_max_legacy_rows_without_scenario_tags(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    output_path = tmp_path / "accepted_300k.jsonl"
    report_path = tmp_path / "merge_report.json"
    spec_path = tmp_path / "dataset_spec.yaml"
    legacy_old_rows = [
        _row("gp4_vi_old_000001", "old legacy 1", "normal_motion", "old"),
        _row("gp4_vi_old_000002", "old legacy 2", "normal_motion", "old"),
        _row("gp4_vi_old_000003", "old legacy 3", "normal_motion", "old"),
    ]
    for row in legacy_old_rows:
        row["metadata"].pop("scenario_tags")
    _write_jsonl(old_path, legacy_old_rows)
    _write_jsonl(
        new_path,
        [
            _row("gp4_vi_new_000001", "new normal 1", "normal_motion", "new"),
            _row("gp4_vi_new_000002", "new normal 2", "normal_motion", "new"),
        ],
    )
    spec_path.write_text(
        "v2_distribution_gates:\n"
        "  scenario_tag_min_counts: {}\n"
        "  max_legacy_rows_without_scenario_tags: 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_accepted_datasets.py",
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--target-rows",
            "3",
            "--distribution-spec",
            str(spec_path),
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert [row["id"] for row in output_rows] == [
        "gp4_vi_new_000001",
        "gp4_vi_new_000002",
        "gp4_vi_old_000001",
    ]
    assert report["distribution"]["legacy_rows_without_scenario_tags"] == 1
    assert report["quota_preserved"] is True
