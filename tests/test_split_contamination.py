import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_splits_rejects_locked_eval_contamination(tmp_path: Path) -> None:
    rows_path = tmp_path / "accepted.jsonl"
    locked_eval = tmp_path / "locked_eval.jsonl"
    output_dir = tmp_path / "splits"
    row = {
        "id": "gp4_en_normal_000001",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "stop"},
            {"role": "assistant", "content": "{\"intent\":\"stop\"}"},
        ],
        "expected_json": {"intent": "stop"},
        "metadata": {
            "language": "en",
            "task_type": "normal",
            "source": "synthetic",
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
        },
    }
    rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    locked_eval.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_splits.py",
            "--input",
            str(rows_path),
            "--locked-eval",
            str(locked_eval),
            "--output-dir",
            str(output_dir),
            "--cloud-root",
            str(tmp_path),
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "locked eval contamination" in result.stdout

def test_build_splits_defaults_to_seed_3526_and_90_5_5(tmp_path: Path) -> None:
    rows_path = tmp_path / "accepted.jsonl"
    rows = []
    for index in range(100):
        rows.append(
            {
                "id": f"gp4_en_normal_{index:06d}",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": f"stop {index}"},
                    {"role": "assistant", "content": "{\"intent\":\"stop\"}"},
                ],
                "expected_json": {"intent": "stop"},
                "metadata": {
                    "language": "en",
                    "task_type": "normal",
                    "source": "synthetic",
                    "safety_class": "safe_motion_plan",
                    "requires_perception": False,
                },
            }
        )
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    default_dir = tmp_path / "splits-default"
    explicit_dir = tmp_path / "splits-explicit"
    base_command = [
        sys.executable,
        "scripts/build_splits.py",
        "--input",
        str(rows_path),
        "--cloud-root",
        str(tmp_path),
        "--allow-tmp",
    ]

    default_result = subprocess.run(
        [*base_command, "--output-dir", str(default_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    explicit_result = subprocess.run(
        [
            *base_command,
            "--output-dir",
            str(explicit_dir),
            "--seed",
            "3526",
            "--train-ratio",
            "0.90",
            "--val-ratio",
            "0.05",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert default_result.returncode == 0, default_result.stderr
    assert explicit_result.returncode == 0, explicit_result.stderr
    assert sum(1 for _ in (default_dir / "train.jsonl").open()) == 90
    assert sum(1 for _ in (default_dir / "val.jsonl").open()) == 5
    assert sum(1 for _ in (default_dir / "test.jsonl").open()) == 5
    assert (default_dir / "train.jsonl").read_text(encoding="utf-8") == (
        explicit_dir / "train.jsonl"
    ).read_text(encoding="utf-8")
