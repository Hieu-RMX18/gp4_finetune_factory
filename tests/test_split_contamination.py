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
