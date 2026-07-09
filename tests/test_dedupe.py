import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def example(row_id: str, expected_json: dict) -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": "GP4 safety prompt"},
            {"role": "user", "content": "stop"},
            {"role": "assistant", "content": json.dumps(expected_json, separators=(",", ":"))},
        ],
        "expected_json": expected_json,
        "metadata": {
            "language": "en",
            "task_type": "normal",
            "source": "seed",
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
        },
    }


def test_dedupe_dataset_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "deduped.jsonl"
    report_path = tmp_path / "dedupe_report.json"
    write_jsonl(
        input_path,
        [
            example("gp4_en_normal_000001", {"intent": "stop"}),
            example("gp4_en_normal_000002", {"intent": "stop"}),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/dedupe_dataset.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {"dropped": 1, "kept": 1, "rows": 2}
