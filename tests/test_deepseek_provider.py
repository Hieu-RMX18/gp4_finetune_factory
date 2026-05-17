from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_batch_deepseek import (
    DeepSeekProviderError,
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
