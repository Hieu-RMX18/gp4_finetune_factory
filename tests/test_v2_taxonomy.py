import json
from pathlib import Path
import sys

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v2_taxonomy import (
    ALLOWED_SCENARIO_TAGS,
    REQUIRED_SCENARIO_TAGS,
    distribution_summary,
    invalid_scenario_tags,
    quota_failures,
)


def _row(*scenario_tags: str) -> dict:
    return {"metadata": {"scenario_tags": list(scenario_tags)}}


def test_required_scenario_tags_exact_set() -> None:
    assert REQUIRED_SCENARIO_TAGS == {
        "singularity",
        "wrist_flip",
        "joint_wrap",
        "timeout_abort_recovery",
        "approval_required",
        "collision_limit_edge",
        "dangerous_os_command",
        "unsupported_tool_hallucination",
    }


def test_legacy_rows_without_scenario_tags_are_valid() -> None:
    row = {"metadata": {}}

    assert invalid_scenario_tags(row) == []


def test_new_rows_without_scenario_tags_are_rejected() -> None:
    row = {"metadata": {"source_dataset": "new"}}

    assert invalid_scenario_tags(row) == ["<missing-new-scenario-tags>"]


def test_new_rows_with_empty_scenario_tags_are_rejected() -> None:
    row = {"metadata": {"source_dataset": "new", "scenario_tags": []}}

    assert invalid_scenario_tags(row) == ["<missing-new-scenario-tags>"]


def test_unknown_scenario_tag_is_reported() -> None:
    row = {"metadata": {"scenario_tags": ["random_tag"]}}

    assert invalid_scenario_tags(row) == ["random_tag"]


def test_quota_failures_reports_missing_required_tags() -> None:
    failures = quota_failures(
        rows=[{"metadata": {"scenario_tags": ["singularity"]}}],
        gates={"scenario_tag_min_counts": {"wrist_flip": 1}},
    )

    assert failures == [
        {
            "tag": "wrist_flip",
            "actual": 0,
            "minimum": 1,
            "message": "scenario tag wrist_flip has 0 rows but requires 1",
        }
    ]


def test_distribution_summary_counts_tags_and_legacy_rows() -> None:
    summary = distribution_summary(
        [
            _row("wrist_flip", "singularity"),
            {"metadata": {}},
            _row(),
            {"metadata": {"source_dataset": "new"}},
            _row("wrist_flip"),
        ]
    )

    assert summary == {
        "rows": 5,
        "legacy_rows_without_scenario_tags": 2,
        "scenario_tags": {
            "singularity": 1,
            "wrist_flip": 2,
        },
    }


def test_schema_scenario_tag_enums_match_taxonomy_source() -> None:
    master_schema = json.loads((ROOT / "schemas/master_example.schema.json").read_text())
    react_schema = json.loads((ROOT / "schemas/gp4_react_ir.schema.json").read_text())

    master_enum = master_schema["properties"]["metadata"]["properties"]["scenario_tags"]["items"]["enum"]
    react_enum = react_schema["properties"]["safety"]["properties"]["scenario_tags"]["items"]["enum"]
    master_tags = set(master_enum)
    react_tags = set(react_enum)

    assert len(master_enum) == len(master_tags)
    assert len(react_enum) == len(react_tags)
    assert master_tags == ALLOWED_SCENARIO_TAGS
    assert react_tags == ALLOWED_SCENARIO_TAGS


def test_master_schema_requires_scenario_tags_for_new_rows_only() -> None:
    schema = json.loads((ROOT / "schemas/master_example.schema.json").read_text())
    validator = Draft7Validator(schema)

    def row(metadata: dict) -> dict:
        return {
            "id": "gp4_vi_normal_000001",
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "dung robot"},
                {"role": "assistant", "content": "{\"intent\":\"stop\"}"},
            ],
            "expected_json": {"intent": "stop"},
            "metadata": {
                "language": "vi",
                "task_type": "normal",
                "source": "synthetic",
                "safety_class": "safe_motion_plan",
                "requires_perception": False,
                **metadata,
            },
        }

    legacy_errors = list(validator.iter_errors(row({})))
    old_errors = list(validator.iter_errors(row({"source_dataset": "old"})))
    tagged_new_errors = list(
        validator.iter_errors(row({"source_dataset": "new", "scenario_tags": ["normal_motion"]}))
    )
    missing_new_errors = list(validator.iter_errors(row({"source_dataset": "new"})))
    empty_new_errors = list(
        validator.iter_errors(row({"source_dataset": "new", "scenario_tags": []}))
    )

    assert legacy_errors == []
    assert old_errors == []
    assert tagged_new_errors == []
    assert missing_new_errors
    assert empty_new_errors
