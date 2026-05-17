import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_can_clone_pushed_branch_without_source_bundle() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup_source = "".join(notebook["cells"][1]["source"])

    assert "git clone --branch codex/gp4-react-ir-cloud-workflow" in setup_source
    assert "if Path(SOURCE_BUNDLE).exists()" in setup_source


def test_kaggle_notebook_can_clone_pushed_branch_without_source_bundle() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/kaggle_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup_source = "".join(notebook["cells"][1]["source"])

    assert "git clone --branch codex/gp4-react-ir-cloud-workflow" in setup_source
    assert "if Path(SOURCE_BUNDLE).exists()" in setup_source
