import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _row(row_id: str, tag: str) -> dict:
    expected = {"intent": "stop"}
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": f"test command {row_id}"},
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
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
            "scenario_tags": [tag],
        },
    }


def test_validate_react_ir_distribution_gate_reports_missing_v2_tags(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    report = tmp_path / "report.json"
    spec = tmp_path / "dataset_spec.yaml"
    dataset.write_text(
        json.dumps(_row("gp4_vi_stop_000001", "singularity"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    spec.write_text(
        "v2_distribution_gates:\n"
        "  scenario_tag_min_counts:\n"
        "    singularity: 1\n"
        "    wrist_flip: 1\n"
        "  min_total_rows: 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(dataset),
            "--strict",
            "--report",
            str(report),
            "--allow-tmp",
            "--distribution-spec",
            str(spec),
            "--enforce-v2-distribution",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert payload["passed"] is False
    assert payload["distribution"]["scenario_tags"]["singularity"] == 1
    assert payload["quota_failures"][0]["tag"] == "wrist_flip"


def test_validate_react_ir_distribution_gate_limits_untagged_legacy_rows(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    report = tmp_path / "report.json"
    spec = tmp_path / "dataset_spec.yaml"
    row = _row("gp4_vi_stop_000001", "normal_motion")
    row["metadata"].pop("scenario_tags")
    row["metadata"]["source_dataset"] = "old"
    dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    spec.write_text(
        "v2_distribution_gates:\n"
        "  scenario_tag_min_counts: {}\n"
        "  max_legacy_rows_without_scenario_tags: 0\n"
        "  min_total_rows: 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(dataset),
            "--strict",
            "--report",
            str(report),
            "--allow-tmp",
            "--distribution-spec",
            str(spec),
            "--enforce-v2-distribution",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert payload["passed"] is False
    assert payload["distribution"]["legacy_rows_without_scenario_tags"] == 1
    assert payload["quota_failures"][0]["tag"] == "__legacy_without_scenario_tags__"


def test_validate_react_ir_distribution_gate_rejects_malformed_spec(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    report = tmp_path / "report.json"
    spec = tmp_path / "dataset_spec.yaml"
    dataset.write_text(
        json.dumps(_row("gp4_vi_stop_000001", "singularity"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    spec.write_text("[]\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(dataset),
            "--strict",
            "--report",
            str(report),
            "--allow-tmp",
            "--distribution-spec",
            str(spec),
            "--enforce-v2-distribution",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "v2_distribution_gates" in (result.stdout + result.stderr)
    assert not report.exists()


def test_validate_react_ir_distribution_report_counts_parse_errors(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    report = tmp_path / "report.json"
    dataset.write_text(
        "{bad json\n"
        + json.dumps(_row("gp4_vi_stop_000001", "normal_motion"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_react_ir_dataset.py",
            "--input",
            str(dataset),
            "--strict",
            "--report",
            str(report),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert payload["rows"] == 2
    assert payload["valid"] == 1
    assert payload["invalid"] == 1
    assert payload["distribution"]["rows"] == 1
