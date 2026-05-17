#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from cloud_runtime import (
    CloudPathError,
    configure_cloud_caches,
    validate_cloud_run_paths,
)
from factory_common import read_jsonl, read_yaml, write_json
from package_adapter import adapter_artifact_exists


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/dataset_spec.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "models/qwen25_gp4_lora_pilot"
DEFAULT_REPORT = ROOT / "reports/training_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a Qwen2.5-7B GP4 Semantic IR QLoRA adapter with Unsloth."
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/splits/train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data/splits/val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        try:
            validate_cloud_run_paths(
                cloud_root=args.cloud_root,
                dry_run=False,
                inputs=[args.train, args.val],
                outputs=[args.output_dir, args.report],
                allow_tmp=args.allow_tmp,
            )
            configure_cloud_caches(
                cloud_root=args.cloud_root,
                dry_run=False,
                allow_tmp=args.allow_tmp,
            )
        except CloudPathError as exc:
            if _can_write_cloud_report(args.report, args.cloud_root, args.allow_tmp):
                write_json(args.report, {"status": "blocked", "reason": str(exc)})
            print(f"training_blocked reason={exc} report={args.report}")
            return 1

    spec = read_yaml(SPEC_PATH)
    training = _training_config(spec, max_steps=args.max_steps)
    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)

    report = {
        "passed": bool(args.dry_run),
        "status": "dry_run_ok" if args.dry_run else "started",
        "model_name": args.model_name,
        "output_dir": str(args.output_dir),
        "train_path": str(args.train),
        "val_path": str(args.val),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "training": training,
    }
    if args.dry_run:
        write_json(args.report, report)
        print(
            f"dry_run_ok train_rows={len(train_rows)} val_rows={len(val_rows)} "
            f"output_dir={args.output_dir} report={args.report}"
        )
        return 0

    try:
        _train(
            train_path=args.train,
            val_path=args.val,
            output_dir=args.output_dir,
            model_name=args.model_name,
            training=training,
            report_path=args.report,
            report=report,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        report["status"] = "blocked"
        report["passed"] = False
        report["reason"] = str(exc)
        write_json(args.report, report)
        print(f"training_blocked reason={exc} report={args.report}")
        return 1
    if not adapter_artifact_exists(args.output_dir):
        report["status"] = "blocked"
        report["passed"] = False
        report["reason"] = "adapter artifact files are missing"
        write_json(args.report, report)
        print(f"training_blocked reason={report['reason']} report={args.report}")
        return 1
    return 0


def _training_config(spec: dict[str, Any], *, max_steps: int | None) -> dict[str, Any]:
    lora = spec["training"]["lora"]
    return {
        "load_in_4bit": bool(lora["load_in_4bit"]),
        "r": int(lora["r"]),
        "alpha": int(lora["alpha"]),
        "dropout": float(lora["dropout"]),
        "max_seq_length": int(lora["max_seq_length"]),
        "max_steps": int(max_steps or lora["pilot_max_steps"]),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "logging_steps": 5,
        "save_steps": 25,
        "seed": 3407,
    }


def _train(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    model_name: str,
    training: dict[str, Any],
    report_path: Path,
    report: dict[str, Any],
) -> None:
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen2.5-7B 4-bit QLoRA training.")

    token = os.getenv("HF_TOKEN") or None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=training["max_seq_length"],
        dtype=None,
        load_in_4bit=training["load_in_4bit"],
        token=token,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=training["r"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=training["alpha"],
        lora_dropout=training["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training["seed"],
    )

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )

    def format_row(row: dict[str, Any]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
            )
        }

    train_dataset = dataset["train"].map(format_row, remove_columns=dataset["train"].column_names)
    val_dataset = dataset["validation"].map(
        format_row,
        remove_columns=dataset["validation"].column_names,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": _sft_config(output_dir=output_dir, training=training),
    }
    trainer = _build_trainer(SFTTrainer, trainer_kwargs, tokenizer)

    train_result = trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(str(output_dir), tokenizer, save_method="lora")
    else:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    report.update(
        {
            "status": "completed",
            "passed": True,
            "train_result": getattr(train_result, "metrics", {}),
        }
    )
    write_json(report_path, report)
    print(f"trained_adapter={output_dir} report={report_path}")


def _build_trainer(trainer_cls: Any, trainer_kwargs: dict[str, Any], tokenizer: Any) -> Any:
    attempts = [
        {"processing_class": tokenizer, "dataset_text_field": "text"},
        {"processing_class": tokenizer},
        {"tokenizer": tokenizer, "dataset_text_field": "text"},
        {"tokenizer": tokenizer},
    ]
    last_error: TypeError | None = None
    for extra_kwargs in attempts:
        try:
            return trainer_cls(**trainer_kwargs, **extra_kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError("SFTTrainer initialization failed without an exception.")
    raise last_error


def _sft_config(*, output_dir: Path, training: dict[str, Any]) -> Any:
    from trl import SFTConfig

    common_kwargs = {
        "output_dir": str(output_dir),
        "max_steps": training["max_steps"],
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "logging_steps": training["logging_steps"],
        "save_steps": training["save_steps"],
        "seed": training["seed"],
        "max_seq_length": training["max_seq_length"],
        "optim": "adamw_8bit",
    }
    try:
        return SFTConfig(**common_kwargs, dataset_text_field="text")
    except TypeError:
        return SFTConfig(**common_kwargs)

def _can_write_cloud_report(
    report: Path,
    cloud_root: Path | None,
    allow_tmp: bool,
) -> bool:
    if cloud_root is None:
        return False
    try:
        validate_cloud_run_paths(
            cloud_root=cloud_root,
            dry_run=False,
            inputs=[],
            outputs=[report],
            allow_tmp=allow_tmp,
        )
    except CloudPathError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
