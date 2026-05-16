from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from factory_common import find_forbidden_key, find_forbidden_text


def test_forbidden_key_rejects_nested_primitive_type() -> None:
    payload = {"act": {"intent": "ptp", "primitive_type": "PTP"}}

    assert find_forbidden_key(payload) == "$.act.primitive_type"


def test_forbidden_text_rejects_motoros_and_execution_claim() -> None:
    assert (
        find_forbidden_text({"message": "I sent the command to the robot"})
        == "hardware execution claim"
    )
    assert find_forbidden_text({"act": "call /yaskawa/foo"}) == "ROS or MotoROS2 call"
