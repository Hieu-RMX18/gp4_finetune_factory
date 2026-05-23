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
    assert "FACTORY_SOURCE_EXPECTED_COMMIT = os.environ.get('GP4_FACTORY_SOURCE_EXPECTED_COMMIT'" in setup_source
    assert "git', 'ls-remote', 'https://github.com/Hieu-RMX18/gp4_finetune_factory.git'" in setup_source
    assert "FACTORY_SOURCE_EXPECTED_COMMIT could not be resolved" in setup_source
    assert "len(FACTORY_SOURCE_EXPECTED_COMMIT) < 12" in setup_source
    assert "!git clone --branch {SOURCE_BRANCH}" not in setup_source
    assert "subprocess.run(['git', 'clone', '--branch', SOURCE_BRANCH" in setup_source
    assert "checkout', '--detach', FACTORY_SOURCE_EXPECTED_COMMIT" in setup_source
    assert "source_revision.json" in setup_source
    assert "factory_source_commit" in setup_source
    assert "Source bundle must include source_revision.json with factory_source_commit" in setup_source
    assert "Factory source commit mismatch" in setup_source
    assert "factory_source_revision.json" in setup_source
    assert "NOTEBOOK_SOURCE_PATH = WORK_DIR / 'notebooks/colab_gp4_react_qwen25_qlora.ipynb'" in setup_source
    assert "NOTEBOOK_DRIVE_COPY = NOTEBOOK_DRIVE_DIR / 'colab_gp4_react_qwen25_qlora.ipynb'" in setup_source
    assert "colab_notebook_copy.json" in setup_source
    assert "if Path(SOURCE_BUNDLE).exists()" in setup_source
    assert "os.chdir('/content')" in setup_source
    assert "GP4_WS_REPO_URL = os.environ.get('GP4_WS_REPO_URL'" in setup_source
    assert "GP4_WS_BRANCH = os.environ.get('GP4_WS_BRANCH', 'ws-deep-rebuild-3526')" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT" in setup_source
    assert "GP4_WS_EXPECTED_COMMIT', '3bbcb0726a4c3305c086c93e2b1e4a320471090b'" in setup_source
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
    assert "def gp4_ws_snapshot_needs_refresh(gp4_ws_path):" in setup_source
    assert "def configure_gp4_ws_git_for_drive(gp4_ws_path):" in setup_source
    assert "'core.fileMode'" in setup_source
    assert "'update-index', '--refresh'" in setup_source
    assert "dirty_gp4_ws_snapshots" in setup_source
    assert "Moved dirty/stale GP4_WS snapshot to:" in setup_source
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
    assert "'HF_TOKEN'" in secret_source
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
    assert "operator_or_explicit_requested_account_after_drive_mount" in setup_source
    assert "GP4_PREVIOUS_RUN_ID" in setup_source
    assert "DATASET_PREVIOUS_RUN_ID = PREVIOUS_RUN_ID" in setup_source
    assert "ADAPTER_PREVIOUS_RUN_ID = PREVIOUS_RUN_ID" in setup_source
    assert "GP4_PREVIOUS_ADAPTER" in setup_source
    assert "GP4_OLD_DATASET" in setup_source
    assert "data/validated/accepted_300k.jsonl" in setup_source
    assert "qwen25_gp4_lora" in setup_source
    assert "Path(previous_dir) / 'models' / adapter_name" in setup_source
    assert "qwen25_gp4_lora_pilot" in setup_source
    assert "LEGACY_DRIVE_ROOT_RUN_ID = 'legacy-drive-root'" in setup_source
    assert "def _adapter_candidates_for_run(previous_dir):" in setup_source
    assert "def _adapter_candidates_for_drive_root(drive_root):" in setup_source
    assert "checkpoint-*" in setup_source
    assert "if previous_dir.name == RUN_ID:" in setup_source
    assert "continue" in setup_source
    assert "def _is_under(path, root):" in setup_source
    assert "Ignoring stale GP4_OLD_DATASET from the current run:" in setup_source
    assert "Ignoring stale GP4_PREVIOUS_ADAPTER from the current run:" in setup_source
    assert "os.environ.pop('GP4_OLD_DATASET', None)" in setup_source
    assert "os.environ.pop('GP4_PREVIOUS_ADAPTER', None)" in setup_source
    assert setup_source.index("if previous_dir.name == RUN_ID:") < setup_source.index(
        "accepted_path = previous_dir / 'data/validated/accepted_300k.jsonl'"
    )
    assert "if not DATASET_PREVIOUS_RUN_ID and not OLD_DATASET:" in setup_source
    assert "if not ADAPTER_PREVIOUS_RUN_ID and not PREVIOUS_ADAPTER:" in setup_source
    assert "DATASET_PREVIOUS_RUN_ID = max(previous_dataset_candidates)[1]" in setup_source
    assert "_, ADAPTER_PREVIOUS_RUN_ID, PREVIOUS_ADAPTER = max(previous_adapter_candidates)" in setup_source
    assert "previous_candidate = Path(DRIVE_ROOT) / DATASET_PREVIOUS_RUN_ID / 'data/validated/accepted_300k.jsonl'" in setup_source
    assert "previous_adapter_candidates.append((adapter_candidate.stat().st_mtime, previous_dir.name, str(adapter_candidate)))" in setup_source
    assert "Path(DRIVE_ROOT) / ADAPTER_PREVIOUS_RUN_ID" in setup_source
    assert "<adapter-only reuse; generate full 300k>" in setup_source
    assert "GP4_PREVIOUS_ADAPTER or GP4_PREVIOUS_RUN_ID" in setup_source
    assert "GP4_OLD_DATASET must live under the configured Google Drive root" in setup_source
    assert "scripts/colab_readiness_report.py" in orchestrator_source
    assert "colab_readiness_{RUN_ID}.json" in orchestrator_source
    assert orchestrator_source.index("scripts/colab_readiness_report.py") < orchestrator_source.index(
        "scripts/cloud_orchestrator.py"
    )
    assert "--old-dataset" in orchestrator_source
    assert "--allow-adapter-only-reuse" in orchestrator_source
    assert "--allow-mixed-prior-artifacts" in orchestrator_source
    assert "--previous-adapter" in orchestrator_source


def test_colab_notebook_explains_drive_mount_consent_failure() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup_source = "".join(notebook["cells"][1]["source"])

    assert "def mount_google_drive_or_explain():" in setup_source
    assert "drive.mount('/content/drive')" in setup_source
    assert "except ValueError as exc:" in setup_source
    assert "Approve the Google Drive permission prompt" in setup_source
    assert "rerun this setup cell" in setup_source
    assert "mount_google_drive_or_explain()" in setup_source


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
    ):
        notebook = json.loads(
            (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )

        assert "completion_audit_{RUN_ID}.json" in source
        assert "benchmark_report_${RUN_ID}.html" in source
        assert "benchmark_report_${RUN_ID}.md" in source
    assert "Maintenance Reference" in source


def test_colab_notebook_command_surface_excludes_dangerous_os_commands() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "rm -rf" not in source
    assert "sudo" not in source
    assert "apt-get" not in source
    assert "WORK_DIR = Path('/content/gp4_finetune_factory_source')" in source
    assert "EXPECTED_WORK_DIR = Path('/content/gp4_finetune_factory_source')" in source
    assert "WORK_DIR.resolve(strict=False) != EXPECTED_WORK_DIR" in source
    assert "raise RuntimeError(f'Unsafe WORK_DIR cleanup path: {WORK_DIR}')" in source
    assert "archive.extractall(" not in source
    assert "for member in archive.infolist():" in source
    assert "Unsafe source bundle path" in source
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in cell.get("source", []):
            assert not line.lstrip().startswith("!")


def test_colab_notebook_writes_failure_reports_for_subprocess_steps() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/colab_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup_source = "".join(notebook["cells"][1]["source"])
    pip_source = "".join(notebook["cells"][3]["source"])
    provider_source = "".join(notebook["cells"][4]["source"])
    orchestrator_source = "".join(notebook["cells"][5]["source"])
    audit_source = "".join(notebook["cells"][6]["source"])

    assert "def run_checked(label, command):" in setup_source
    assert "capture_output=True" in setup_source
    assert "check=False" in setup_source
    assert "stdout_tail" in setup_source
    assert "stderr_tail" in setup_source
    assert "failure_{RUN_ID}.json" in setup_source
    assert "subprocess.CalledProcessError" in setup_source
    assert "run_checked('pip-install'" in pip_source
    assert "run_checked('provider-probe'" in provider_source
    assert "run_checked('colab-readiness', readiness_command)" in orchestrator_source
    assert "run_checked('cloud-orchestrator', orchestrator_command)" in orchestrator_source
    assert "run_checked('completion-audit'" in audit_source

def test_kaggle_react_notebook_is_disabled_in_drive_only_workflow() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/kaggle_gp4_react_qwen25_qlora.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "Deprecated notebook disabled" in source
    assert "Google Drive only" in source
    assert "/kaggle/" not in source
    assert "git clone --branch" not in source
