#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

from cloud_runtime import (
    CloudPathError,
    configure_cloud_caches,
    validate_cloud_run_paths,
)
from check_cloud_storage_policy import CloudStoragePolicy, is_allowed_cloud_path
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
    parser.add_argument("--resume-from-adapter", type=Path)
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
            _validate_resume_adapter(
                args.resume_from_adapter,
                cloud_root=args.cloud_root,
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
    resume_metadata = _resume_adapter_metadata(
        args.resume_from_adapter,
        cloud_root=args.cloud_root,
        allow_tmp=args.allow_tmp,
    )

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
        **resume_metadata,
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
            resume_from_adapter=args.resume_from_adapter,
        )
    except Exception as exc:
        report["status"] = "blocked"
        report["passed"] = False
        report["reason"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
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
    resume_from_adapter: Path | None,
) -> None:
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen2.5-7B 4-bit QLoRA training.")

    token = os.getenv("HF_TOKEN") or None
    load_source = str(resume_from_adapter) if resume_from_adapter is not None else model_name
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_source,
        max_seq_length=training["max_seq_length"],
        dtype=None,
        load_in_4bit=training["load_in_4bit"],
        token=token,
    )
    if resume_from_adapter is None:
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

    train_dataset = Dataset.from_list(
        _tokenize_training_rows(read_jsonl(train_path), tokenizer, training["max_seq_length"])
    )
    val_dataset = Dataset.from_list(
        _tokenize_training_rows(read_jsonl(val_path), tokenizer, training["max_seq_length"])
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=_training_args(TrainingArguments, output_dir=output_dir, training=training),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )

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


def _format_training_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    return [
        {
            "text": tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
            )
        }
        for row in rows
    ]


def _tokenize_training_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int
) -> list[dict[str, list[int]]]:
    texts = [row["text"] for row in _format_training_rows(rows, tokenizer)]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    return [
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        for input_ids, attention_mask in zip(
            encoded["input_ids"],
            encoded["attention_mask"],
            strict=True,
        )
    ]


def _training_args(
    training_args_cls: Any, *, output_dir: Path, training: dict[str, Any]
) -> Any:
    return training_args_cls(
        output_dir=str(output_dir),
        max_steps=training["max_steps"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        logging_steps=training["logging_steps"],
        save_steps=training["save_steps"],
        seed=training["seed"],
        optim="adamw_8bit",
        dataloader_num_workers=0,
        report_to=[],
    )

def _validate_resume_adapter(
    resume_from_adapter: Path | None,
    *,
    cloud_root: Path | None,
    allow_tmp: bool,
) -> None:
    if resume_from_adapter is None:
        return
    if cloud_root is None:
        raise CloudPathError("CLOUD_ROOT is required when reusing a previous adapter.")
    policy = CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=allow_tmp)
    if not is_allowed_cloud_path(resume_from_adapter, policy):
        raise CloudPathError(
            f"resume adapter path is outside approved cloud storage: {resume_from_adapter}"
        )
    if not adapter_artifact_exists(resume_from_adapter):
        raise CloudPathError(
            f"resume adapter artifact files are missing: {resume_from_adapter}"
        )

def _resume_adapter_metadata(
    resume_from_adapter: Path | None,
    *,
    cloud_root: Path | None,
    allow_tmp: bool,
) -> dict[str, Any]:
    if resume_from_adapter is None:
        return {
            "resume_from_adapter": "",
            "resume_from_adapter_allowed_cloud_path": False,
            "resume_from_adapter_artifact_exists": False,
        }
    policy = (
        CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=allow_tmp)
        if cloud_root is not None
        else CloudStoragePolicy((), allow_tmp=allow_tmp)
    )
    return {
        "resume_from_adapter": str(resume_from_adapter),
        "resume_from_adapter_allowed_cloud_path": is_allowed_cloud_path(
            resume_from_adapter,
            policy,
        ),
        "resume_from_adapter_artifact_exists": adapter_artifact_exists(
            resume_from_adapter
        ),
    }

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
