from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from typo_noise_policy import build_typo_noise_row, typo_expected_json


def test_down_typo_with_required_slots_maps_to_down_motion() -> None:
    expected = typo_expected_json("đi xogun 5 cm trong base_link")

    assert expected == {
        "intent": "move_relative",
        "delta": {"x": 0.0, "y": 0.0, "z": -5.0},
        "linear_unit": "cm",
        "reference_frame": "base_link",
    }


def test_down_typo_without_required_slots_returns_missing_slot() -> None:
    expected = typo_expected_json("đi xung")

    assert expected["error"] == "MISSING_SLOT"
    assert expected["missing_slots"] == ["distance", "linear_unit", "reference_frame"]


def test_typo_noise_row_marks_ambiguous_missing_slot_as_locked_eval() -> None:
    row = build_typo_noise_row("locked_typo_001", "di xuông")

    assert row["metadata"]["source"] == "locked_typo_eval"
    assert row["metadata"]["task_type"] == "ambiguous"
    assert row["expected_json"]["error"] == "MISSING_SLOT"
    assert row["messages"][-1]["content"].startswith("{")
