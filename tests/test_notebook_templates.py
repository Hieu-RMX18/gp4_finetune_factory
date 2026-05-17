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
    assert "os.chdir('/content')" in setup_source


def test_colab_notebook_loads_deepseek_key_from_colab_secrets() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    secret_source = "".join(notebook["cells"][2]["source"])

    assert "from google.colab import userdata" in secret_source
    assert "'DEEPSEEK_API_KEY'" in secret_source
    assert "userdata.get(secret_name)" in secret_source
    assert "'OPENAI_BASE_URL'" in secret_source
    assert "os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_BASE_URL')" in secret_source
    assert "os.environ['OPENAI_MODEL'] = os.environ.get('OPENAI_MODEL', 'gpt-5.4')" in secret_source


def test_colab_notebook_requests_gpu_runtime() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )

    assert notebook["metadata"]["accelerator"] == "GPU"


def test_kaggle_notebook_can_clone_pushed_branch_without_source_bundle() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/kaggle_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup_source = "".join(notebook["cells"][1]["source"])

    assert "git clone --branch codex/gp4-react-ir-cloud-workflow" in setup_source
    assert "if Path(SOURCE_BUNDLE).exists()" in setup_source
    assert "os.chdir('/kaggle/working')" in setup_source
