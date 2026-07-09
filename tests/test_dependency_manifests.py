from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _lines(path: str) -> set[str]:
    return {
        line.strip()
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_cloud_requirements_declare_training_runtime_packages() -> None:
    lines = _lines("requirements-cloud.txt")

    assert "-r requirements.txt" in lines
    assert any(line.startswith("torch") for line in lines)
    assert any(line.startswith("transformers") for line in lines)
    assert any(line.startswith("datasets") for line in lines)
    assert any(line.startswith("accelerate") for line in lines)
    assert any(line.startswith("peft") for line in lines)
    assert any(line.startswith("trl") for line in lines)
    assert any(line.startswith("unsloth") for line in lines)


def test_local_adapter_requirements_declare_loading_runtime_packages() -> None:
    lines = _lines("requirements-local-adapter.txt")

    assert "-r requirements.txt" in lines
    assert any(line.startswith("torch") for line in lines)
    assert any(line.startswith("transformers") for line in lines)
    assert any(line.startswith("peft") for line in lines)
    assert any(line.startswith("safetensors") for line in lines)
