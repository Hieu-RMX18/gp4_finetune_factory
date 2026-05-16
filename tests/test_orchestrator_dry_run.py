import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generate_batch_deepseek_dry_run_respects_seed_gate(tmp_path: Path) -> None:
    seed = tmp_path / "seed.jsonl"
    report = tmp_path / "generation_report.json"
    seed.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_batch_deepseek.py",
            "--seed",
            str(seed),
            "--output",
            str(tmp_path / "raw.jsonl"),
            "--cloud-root",
            str(tmp_path),
            "--report",
            str(report),
            "--dry-run",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["blocked_reason"] == "seed gate blocked generation"
