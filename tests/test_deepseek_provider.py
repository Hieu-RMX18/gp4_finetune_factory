import json
from pathlib import Path
import sys
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_batch_deepseek import (
    DeepSeekProviderError,
    _build_generation_payload,
    _expand_rows_to_count,
    _generate_rows,
    _parse_generated_rows,
    resolve_deepseek_config,
)


def _policy(model: str = "deepseek-v4-flash") -> dict:
    return {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": model,
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.4,
            "max_tokens": 7000,
        }
    }


def test_deepseek_requires_deepseek_api_key_not_openai_key() -> None:
    env = {
        "OPENAI_API_KEY": "sk-openai-placeholder",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    }

    with pytest.raises(DeepSeekProviderError, match="DEEPSEEK_API_KEY"):
        resolve_deepseek_config(_policy(), env=env, dry_run=False)


def test_deepseek_non_dry_run_rejects_localhost_base_url() -> None:
    env = {
        "DEEPSEEK_API_KEY": "sk-deepseek-placeholder",
        "DEEPSEEK_BASE_URL": "http://localhost:20128/v1",
    }

    with pytest.raises(DeepSeekProviderError, match="localhost"):
        resolve_deepseek_config(_policy(), env=env, dry_run=False)


def test_deepseek_defaults_to_flash_model() -> None:
    env = {"DEEPSEEK_API_KEY": "sk-deepseek-placeholder"}

    config = resolve_deepseek_config(_policy(), env=env, dry_run=False)

    assert config.model == "deepseek-v4-flash"
    assert config.base_url == "https://api.deepseek.com"


def test_deepseek_rejects_unapproved_model() -> None:
    env = {"DEEPSEEK_API_KEY": "sk-deepseek-placeholder"}

    with pytest.raises(DeepSeekProviderError, match="unsupported DeepSeek model"):
        resolve_deepseek_config(_policy("deepseek-chat"), env=env, dry_run=False)


def test_deepseek_parser_accepts_fenced_json_object() -> None:
    content = '```json\n{"examples": [{"id": "gp4_vi_synthetic_000001"}]}\n```'

    rows = _parse_generated_rows(content)

    assert rows == [{"id": "gp4_vi_synthetic_000001"}]


def test_deepseek_payload_requests_json_object() -> None:
    payload = _build_generation_payload(
        model="deepseek-v4-flash",
        seed_rows=[{"id": "seed"}],
        count=10,
        temperature=0.4,
        max_tokens=7000,
    )

    assert payload["response_format"] == {"type": "json_object"}
    assert "examples" in payload["messages"][0]["content"]

def test_deepseek_prompt_requires_perception_rows_to_be_safe_errors() -> None:
    payload = _build_generation_payload(
        model="deepseek-v4-flash",
        seed_rows=[{"id": "seed"}],
        count=10,
        temperature=0.4,
        max_tokens=7000,
    )

    prompt_text = "\n".join(message["content"] for message in payload["messages"])

    assert "PERCEPTION_REQUIRED" in prompt_text
    assert "requires_perception" in prompt_text
    assert "safe error" in prompt_text

def test_deepseek_generation_batches_until_requested_count(monkeypatch) -> None:
    calls: list[int] = []

    class FakeResponse:
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps({"examples": self.rows})}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout: int):
        payload = json.loads(request.data.decode("utf-8"))
        user_payload = json.loads(payload["messages"][1]["content"])
        requested_rows = int(user_payload["requested_rows"])
        calls.append(requested_rows)
        returned_rows = min(2, requested_rows)
        rows = [{"id": f"gp4_vi_synthetic_{len(calls)}_{index}"} for index in range(returned_rows)]
        return FakeResponse(rows)

    monkeypatch.setattr("generate_batch_deepseek.urllib.request.urlopen", fake_urlopen)

    rows = _generate_rows(
        api_key="sk-test",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        seed_rows=[{"id": "seed"}],
        count=5,
        temperature=0.4,
        max_tokens=7000,
        batch_size=3,
        retry_limit=3,
    )

    assert len(rows) == 5
    assert calls == [3, 3, 1]

def test_deepseek_generation_falls_back_to_9router_gpt54(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            row = {"id": "gp4_vi_synthetic_000001"}
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps({"examples": [row]})}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout: int):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append((request.full_url, payload["model"]))
        if len(calls) == 1:
            raise RuntimeError("deepseek quota exhausted")
        return FakeResponse()

    monkeypatch.setattr("generate_batch_deepseek.urllib.request.urlopen", fake_urlopen)

    rows = _generate_rows(
        api_key="sk-deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        seed_rows=[{"id": "seed"}],
        count=1,
        temperature=0.4,
        max_tokens=7000,
        batch_size=1,
        retry_limit=1,
        fallback_api_key="sk-9router",
        fallback_base_url="http://localhost:20128/v1",
        fallback_model="gpt-5.4",
    )

    assert rows == [{"id": "gp4_vi_synthetic_000001"}]
    assert calls == [
        ("https://api.deepseek.com/chat/completions", "deepseek-v4-flash"),
        ("http://localhost:20128/v1/chat/completions", "gpt-5.4"),
    ]

def test_generation_can_use_9router_without_deepseek_provider(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            row = {"id": "gp4_vi_synthetic_000001"}
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps({"examples": [row]})}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout: int):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append((request.full_url, payload["model"]))
        return FakeResponse()

    monkeypatch.setattr("generate_batch_deepseek.urllib.request.urlopen", fake_urlopen)

    rows = _generate_rows(
        api_key="",
        base_url="",
        model="",
        seed_rows=[{"id": "seed"}],
        count=1,
        temperature=0.4,
        max_tokens=7000,
        batch_size=1,
        retry_limit=1,
        fallback_base_url="http://localhost:20128/v1",
        fallback_model="gpt-5.4",
    )

    assert rows == [{"id": "gp4_vi_synthetic_000001"}]
    assert calls == [("http://localhost:20128/v1/chat/completions", "gpt-5.4")]

def test_provider_seed_rows_can_be_expanded_to_requested_count() -> None:
    seed_rows = [
        {
            "id": "gp4_vi_normal_000001",
            "messages": [
                {"role": "system", "content": "GP4 safety Semantic IR system prompt"},
                {"role": "user", "content": "move up"},
                {"role": "assistant", "content": "{\"intent\":\"stop\"}"},
            ],
            "expected_json": {"intent": "stop"},
            "metadata": {
                "language": "vi",
                "task_type": "normal",
                "source": "seed",
                "safety_class": "safe_motion_plan",
                "requires_perception": False,
            },
        }
    ]

    rows = _expand_rows_to_count(seed_rows, count=3, id_prefix="gp4_vi_synthetic")

    assert [row["id"] for row in rows] == [
        "gp4_vi_synthetic_000001",
        "gp4_vi_synthetic_000002",
        "gp4_vi_synthetic_000003",
    ]
    assert len({row["messages"][1]["content"] for row in rows}) == 3
    assert all(row["metadata"]["source"] == "synthetic" for row in rows)

def test_seed_expansion_needs_no_provider_when_enabled() -> None:
    seed_rows = [
        {
            "id": "gp4_vi_normal_000001",
            "messages": [
                {"role": "system", "content": "GP4 safety Semantic IR system prompt"},
                {"role": "user", "content": "dừng robot"},
                {"role": "assistant", "content": "{\"intent\":\"stop\"}"},
            ],
            "expected_json": {"intent": "stop"},
            "metadata": {
                "language": "vi",
                "task_type": "normal",
                "source": "seed",
                "safety_class": "safe_motion_plan",
                "requires_perception": False,
            },
        }
    ]

    rows = _generate_rows(
        api_key="",
        base_url="",
        model="",
        seed_rows=seed_rows,
        count=2,
        temperature=0.4,
        max_tokens=7000,
        batch_size=50,
        retry_limit=1,
        expand_from_provider=True,
    )

    assert len(rows) == 2
    assert rows[0]["expected_json"] == {"intent": "stop"}

def test_generator_cli_seed_expansion_runs_without_provider_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed = tmp_path / "seed.jsonl"
    output = tmp_path / "raw.jsonl"
    report = tmp_path / "report.json"
    row = {
        "id": "gp4_vi_normal_000001",
        "messages": [
            {"role": "system", "content": "GP4 safety Semantic IR system prompt"},
            {"role": "user", "content": "dừng robot"},
            {"role": "assistant", "content": "{\"intent\":\"stop\"}"},
        ],
        "expected_json": {"intent": "stop"},
        "metadata": {
            "language": "vi",
            "task_type": "normal",
            "source": "seed",
            "safety_class": "safe_motion_plan",
            "requires_perception": False,
        },
    }
    seed.write_text(
        "".join(json.dumps({**row, "id": f"gp4_vi_normal_{index:06d}"}) + "\n" for index in range(1, 51)),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_batch_deepseek.py",
            "--seed",
            str(seed),
            "--output",
            str(output),
            "--cloud-root",
            str(tmp_path),
            "--report",
            str(report),
            "--count",
            "3",
            "--allow-tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
    assert json.loads(report.read_text(encoding="utf-8"))["generated"] == 3
