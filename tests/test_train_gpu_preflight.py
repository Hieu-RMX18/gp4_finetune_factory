import builtins
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_unsloth_qlora


def test_train_checks_cuda_before_importing_training_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    for module_name in ("datasets", "trl", "unsloth"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in {"datasets", "trl", "unsloth"}:
            raise AssertionError(f"imported {name} before CUDA preflight")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        train_unsloth_qlora._train(
            train_path=tmp_path / "train.jsonl",
            val_path=tmp_path / "val.jsonl",
            output_dir=tmp_path / "adapter",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            training={},
            report_path=tmp_path / "train_report.json",
            report={},
            resume_from_adapter=None,
        )
