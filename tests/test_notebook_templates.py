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

    assert "SOURCE_BRANCH = os.environ.get('GP4_SOURCE_BRANCH'" in setup_source
    assert "git clone --branch {SOURCE_BRANCH}" in setup_source
    assert "if Path(SOURCE_BUNDLE).exists()" in setup_source
    assert "os.chdir('/content')" in setup_source
    assert "GP4_WS_REPO_URL = os.environ.get('GP4_WS_REPO_URL'" in setup_source
    assert "GP4_WS_BRANCH = os.environ.get('GP4_WS_BRANCH', 'ws-deep-rebuild-3526')" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT" in setup_source
    assert "if not GP4_WS_EXPECTED_COMMIT:" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT is required" in setup_source
    assert "len(GP4_WS_EXPECTED_COMMIT) < 12" in setup_source
    assert "ACTUAL_GP4_WS_COMMIT = subprocess.run" in setup_source
    assert "GP4_WS commit mismatch" in setup_source
    assert "ACTUAL_GP4_WS_COMMIT.lower().startswith(GP4_WS_EXPECTED_COMMIT.lower())" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT.lower().startswith(ACTUAL_GP4_WS_COMMIT.lower())" not in setup_source
    assert "os.environ['GP4_WS_EXPECTED_COMMIT'] = subprocess.run" not in setup_source
    assert "contract_snapshots/gp4_ws_{GP4_WS_BRANCH}" in setup_source
    assert "GP4_WS must live under CLOUD_ROOT/contract_snapshots" in setup_source
    assert "Path(GP4_WS).resolve(strict=False)" in setup_source
    assert "Path(GP4_WS).parent.mkdir(parents=True, exist_ok=True)" in setup_source
    assert "subprocess.run(['git', 'clone', '--branch', GP4_WS_BRANCH" in setup_source
    assert setup_source.index("if not GP4_WS_EXPECTED_COMMIT:") < setup_source.index(
        "subprocess.run(['git', 'clone', '--branch', GP4_WS_BRANCH"
    )
    assert setup_source.index("Path(GP4_WS).parent.mkdir") < setup_source.index(
        "subprocess.run(['git', 'clone', '--branch', GP4_WS_BRANCH"
    )


def test_colab_notebook_loads_optional_provider_secrets() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    secret_source = "".join(notebook["cells"][2]["source"])

    assert "from google.colab import userdata" in secret_source
    assert "'DEEPSEEK_API_KEY'" in secret_source
    assert "userdata.get(secret_name)" in secret_source
    assert "except Exception" in secret_source
    assert "'OPENAI_BASE_URL'" in secret_source
    assert "Set DEEPSEEK_API_KEY or OPENAI_BASE_URL" not in secret_source
    assert "os.environ['OPENAI_MODEL'] = os.environ.get('OPENAI_MODEL', 'gpt-5.4')" in secret_source


def test_colab_notebook_locks_drive_account_and_reuses_previous_accepted_dataset() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup_source = "".join(notebook["cells"][1]["source"])
    orchestrator_source = "".join(notebook["cells"][5]["source"])

    assert "DRIVE_ACCOUNT_EMAIL = 'johnwickiller4444@gmail.com'" in setup_source
    assert "drive_account_hint.txt" in setup_source
    assert "drive_account_confirmation.json" in setup_source
    assert "GP4_DRIVE_ACCOUNT_CONFIRMED" in setup_source
    assert "operator_input_after_drive_mount" in setup_source
    assert "GP4_PREVIOUS_RUN_ID" in setup_source
    assert "GP4_PREVIOUS_ADAPTER" in setup_source
    assert "GP4_OLD_DATASET" in setup_source
    assert "data/validated/accepted_300k.jsonl" in setup_source
    assert "models/qwen25_gp4_lora" in setup_source
    assert "GP4_OLD_DATASET or GP4_PREVIOUS_RUN_ID" in setup_source
    assert "GP4_PREVIOUS_ADAPTER or GP4_PREVIOUS_RUN_ID" in setup_source
    assert "scripts/colab_readiness_report.py" in orchestrator_source
    assert "colab_readiness_{RUN_ID}.json" in orchestrator_source
    assert orchestrator_source.index("scripts/colab_readiness_report.py") < orchestrator_source.index(
        "scripts/cloud_orchestrator.py"
    )
    assert "--old-dataset" in orchestrator_source
    assert "--previous-adapter" in orchestrator_source


def test_colab_notebook_requests_gpu_runtime() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )

    assert notebook["metadata"]["accelerator"] == "GPU"


def test_cloud_notebooks_name_final_benchmark_and_audit_artifacts() -> None:
    for notebook_name in (
        "colab_gp4_react_qwen25_qlora.ipynb",
        "kaggle_gp4_react_qwen25_qlora.ipynb",
    ):
        notebook = json.loads(
            (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )

        assert "completion_audit_${RUN_ID}.json" in source
        assert "benchmark_report_${RUN_ID}.html" in source
        assert "benchmark_report_${RUN_ID}.md" in source
        assert "Maintenance Reference" in source


def test_kaggle_notebook_can_clone_pushed_branch_without_source_bundle() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/kaggle_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup_source = "".join(notebook["cells"][1]["source"])

    assert "SOURCE_BRANCH = os.environ.get('GP4_SOURCE_BRANCH'" in setup_source
    assert "git clone --branch {SOURCE_BRANCH}" in setup_source
    assert "if Path(SOURCE_BUNDLE).exists()" in setup_source
    assert "os.chdir('/kaggle/working')" in setup_source
    assert "GP4_WS_REPO_URL = os.environ.get('GP4_WS_REPO_URL'" in setup_source
    assert "GP4_WS_BRANCH = os.environ.get('GP4_WS_BRANCH', 'ws-deep-rebuild-3526')" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT" in setup_source
    assert "if not GP4_WS_EXPECTED_COMMIT:" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT is required" in setup_source
    assert "len(GP4_WS_EXPECTED_COMMIT) < 12" in setup_source
    assert "ACTUAL_GP4_WS_COMMIT = subprocess.run" in setup_source
    assert "GP4_WS commit mismatch" in setup_source
    assert "ACTUAL_GP4_WS_COMMIT.lower().startswith(GP4_WS_EXPECTED_COMMIT.lower())" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT.lower().startswith(ACTUAL_GP4_WS_COMMIT.lower())" not in setup_source
    assert "os.environ['GP4_WS_EXPECTED_COMMIT'] = subprocess.run" not in setup_source
    assert "contract_snapshots/gp4_ws_{GP4_WS_BRANCH}" in setup_source
    assert "Path(GP4_WS).parent.mkdir(parents=True, exist_ok=True)" in setup_source
    assert "subprocess.run(['git', 'clone', '--branch', GP4_WS_BRANCH" in setup_source
    assert setup_source.index("if not GP4_WS_EXPECTED_COMMIT:") < setup_source.index(
        "subprocess.run(['git', 'clone', '--branch', GP4_WS_BRANCH"
    )
    assert setup_source.index("Path(GP4_WS).parent.mkdir") < setup_source.index(
        "subprocess.run(['git', 'clone', '--branch', GP4_WS_BRANCH"
    )
