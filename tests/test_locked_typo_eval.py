import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from typo_noise_policy import locked_typo_eval_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_locked_typo_eval_rows_are_separate_from_training_source() -> None:
    rows = locked_typo_eval_rows()

    assert rows
    assert {row["metadata"]["source"] for row in rows} == {"locked_typo_eval"}


def test_build_splits_rejects_locked_typo_eval_contamination(tmp_path: Path) -> None:
    locked_rows = locked_typo_eval_rows()
    accepted_path = tmp_path / "accepted.jsonl"
    locked_eval_path = tmp_path / "locked_typo_eval.jsonl"

    _write_jsonl(accepted_path, [locked_rows[0]])
    _write_jsonl(locked_eval_path, locked_rows)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_splits.py",
            "--input",
            str(accepted_path),
            "--locked-eval",
            str(locked_eval_path),
            "--output-dir",
            str(tmp_path / "splits"),
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


def test_inference_uses_deterministic_typo_policy_for_known_down_typos() -> None:
    from run_adapter_inference import deterministic_typo_model_output

    row = {
        "id": "gp4_vi_typo_policy_001",
        "messages": [
            {"role": "system", "content": "GP4 safety Semantic IR system prompt"},
            {"role": "user", "content": "di xuông 7 cm trong base_link"},
        ],
    }

    output = deterministic_typo_model_output(row)

    assert output == (
        '{"intent":"move_relative","delta":{"x":0.0,"y":0.0,"z":-7.0},'
        '"linear_unit":"cm","reference_frame":"base_link"}'
    )
