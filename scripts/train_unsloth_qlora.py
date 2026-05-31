#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import inspect
import multiprocessing
import os
import traceback
from pathlib import Path
from typing import Any, Iterator

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
DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
QWEN_EOS_TOKEN_CANDIDATES = ("<|im_end|>", "<|endoftext|>")
QWEN_PAD_TOKEN_CANDIDATES = ("<|PAD_TOKEN|>", "<|endoftext|>")
NON_INTERACTIVE_TRAINING_ENV = {
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "WANDB_SILENT": "true",
    "WANDB_CONSOLE": "off",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "BITSANDBYTES_NOWELCOME": "1",
    "TQDM_DISABLE": "1",
}


def main() -> int:
    parser = build_arg_parser()
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
            _validate_resume_checkpoint(
                args.resume_from_checkpoint,
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
    resume_metadata.update(
        _resume_checkpoint_metadata(
            args.resume_from_checkpoint,
            cloud_root=args.cloud_root,
            allow_tmp=args.allow_tmp,
        )
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
        _configure_non_interactive_training_env()
        with _non_interactive_input_guard():
            _train(
                train_path=args.train,
                val_path=args.val,
                output_dir=args.output_dir,
                model_name=args.model_name,
                training=training,
                report_path=args.report,
                report=report,
                resume_from_adapter=args.resume_from_adapter,
                resume_from_checkpoint=args.resume_from_checkpoint,
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

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Qwen2.5-7B GP4 Semantic IR QLoRA adapter with Unsloth."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from-adapter", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _training_config(spec: dict[str, Any], *, max_steps: int | None) -> dict[str, Any]:
    lora = spec["training"]["lora"]
    return {
        "load_in_4bit": bool(lora["load_in_4bit"]),
        "r": int(lora["r"]),
        "alpha": int(lora["alpha"]),
        "dropout": float(lora["dropout"]),
        "max_seq_length": int(lora["max_seq_length"]),
        "max_steps": int(max_steps) if max_steps is not None else -1,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "eval_steps": 250,
        "save_steps": 250,
        "save_total_limit": 3,
        "seed": 3526,
    }


def _configure_non_interactive_training_env() -> dict[str, str]:
    for key, value in NON_INTERACTIVE_TRAINING_ENV.items():
        os.environ[key] = value
    return dict(NON_INTERACTIVE_TRAINING_ENV)


@contextmanager
def _non_interactive_input_guard() -> Iterator[None]:
    original_input = builtins.input

    def blocked_input(prompt: object = "") -> str:
        prompt_text = str(prompt).strip()
        reason = "interactive prompt blocked during non-interactive training"
        if prompt_text:
            reason = f"{reason}: {prompt_text}"
        raise RuntimeError(reason)

    builtins.input = blocked_input
    try:
        yield
    finally:
        builtins.input = original_input


def _configure_tokenizer_special_tokens(tokenizer: Any) -> None:
    if not _token_exists(tokenizer, getattr(tokenizer, "eos_token", None)):
        tokenizer.eos_token = _first_existing_token(
            tokenizer,
            QWEN_EOS_TOKEN_CANDIDATES,
            "EOS",
        )
    if not _token_exists(tokenizer, getattr(tokenizer, "pad_token", None)):
        tokenizer.pad_token = _first_existing_token(
            tokenizer,
            (getattr(tokenizer, "eos_token", None), *QWEN_PAD_TOKEN_CANDIDATES),
            "padding",
        )


def _configure_single_process_dataset_map() -> None:
    multiprocessing.set_start_method("spawn", force=True)


def _first_existing_token(
    tokenizer: Any,
    candidates: tuple[str | None, ...],
    label: str,
) -> str:
    for token in candidates:
        if _token_exists(tokenizer, token):
            return str(token)
    raise RuntimeError(f"Unable to find a valid {label} token in tokenizer vocabulary.")


def _token_exists(tokenizer: Any, token: str | None) -> bool:
    if not token:
        return False
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return True
    if token_id is None or token_id == -1:
        return False
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    return not (unk_token_id is not None and token_id == unk_token_id and token != unk_token)


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
    resume_from_checkpoint: Path | None,
) -> None:
    _configure_non_interactive_training_env()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen2.5-7B 4-bit QLoRA training.")

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    _configure_single_process_dataset_map()
    token = os.getenv("HF_TOKEN") or None
    load_source = str(resume_from_adapter) if resume_from_adapter is not None else model_name
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_source,
        max_seq_length=training["max_seq_length"],
        dtype=None,
        load_in_4bit=training["load_in_4bit"],
        token=token,
    )
    _configure_tokenizer_special_tokens(tokenizer)
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
        _format_training_rows(read_jsonl(train_path), tokenizer)
    )
    val_dataset = Dataset.from_list(
        _format_training_rows(read_jsonl(val_path), tokenizer)
    )
    trainer = _build_sft_trainer(
        sft_trainer_cls=SFTTrainer,
        sft_config_cls=SFTConfig,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        training=training,
    )

    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
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


def _build_sft_trainer(
    *,
    sft_trainer_cls: Any,
    sft_config_cls: Any,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: Path,
    training: dict[str, Any],
) -> Any:
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": _build_sft_config(
            sft_config_cls,
            output_dir=output_dir,
            training=training,
            tokenizer=tokenizer,
        ),
    }
    if _constructor_accepts(sft_trainer_cls, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if _constructor_accepts(sft_trainer_cls, "dataset_text_field"):
        trainer_kwargs["dataset_text_field"] = "text"
    if _constructor_accepts(sft_trainer_cls, "max_seq_length"):
        trainer_kwargs["max_seq_length"] = training["max_seq_length"]
    if _constructor_accepts(sft_trainer_cls, "packing"):
        trainer_kwargs["packing"] = False
    return sft_trainer_cls(**trainer_kwargs)


def _build_sft_config(
    sft_config_cls: Any,
    *,
    output_dir: Path,
    training: dict[str, Any],
    tokenizer: Any | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": training.get("max_steps", -1),
        "num_train_epochs": training.get("num_train_epochs", 1.0),
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "warmup_ratio": training.get("warmup_ratio", 0.03),
        "weight_decay": training.get("weight_decay", 0.01),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "logging_steps": training["logging_steps"],
        "save_steps": training["save_steps"],
        "eval_strategy": "steps",
        "eval_steps": training.get("eval_steps", training["save_steps"]),
        "save_strategy": "steps",
        "save_total_limit": training.get("save_total_limit", 3),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": training["seed"],
        "optim": "adamw_8bit",
        "fp16": True,
        "bf16": False,
        "dataloader_num_workers": 0,
        "report_to": [],
    }
    if _constructor_accepts(sft_config_cls, "max_length"):
        kwargs["max_length"] = training["max_seq_length"]
    elif _constructor_accepts(sft_config_cls, "max_seq_length"):
        kwargs["max_seq_length"] = training["max_seq_length"]
    if _constructor_accepts(sft_config_cls, "packing"):
        kwargs["packing"] = False
    if _constructor_accepts(sft_config_cls, "dataset_text_field"):
        kwargs["dataset_text_field"] = "text"
    if _constructor_accepts(sft_config_cls, "dataset_num_proc"):
        kwargs["dataset_num_proc"] = 1
    if tokenizer is not None and _constructor_accepts(sft_config_cls, "eos_token"):
        kwargs["eos_token"] = getattr(tokenizer, "eos_token", None)
    if tokenizer is not None and _constructor_accepts(sft_config_cls, "pad_token"):
        kwargs["pad_token"] = getattr(tokenizer, "pad_token", None)
    return sft_config_cls(**kwargs)


def _constructor_accepts(cls: Any, parameter: str) -> bool:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    return parameter in parameters or any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
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

def _validate_resume_checkpoint(
    resume_from_checkpoint: Path | None,
    *,
    cloud_root: Path | None,
    allow_tmp: bool,
) -> None:
    if resume_from_checkpoint is None:
        return
    if cloud_root is None:
        raise CloudPathError("CLOUD_ROOT is required when resuming a checkpoint.")
    policy = CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=allow_tmp)
    if not is_allowed_cloud_path(resume_from_checkpoint, policy):
        raise CloudPathError(
            "resume checkpoint path is outside approved cloud storage: "
            f"{resume_from_checkpoint}"
        )
    if not resume_from_checkpoint.is_dir():
        raise CloudPathError(
            f"resume checkpoint directory is missing: {resume_from_checkpoint}"
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

def _resume_checkpoint_metadata(
    resume_from_checkpoint: Path | None,
    *,
    cloud_root: Path | None,
    allow_tmp: bool,
) -> dict[str, Any]:
    if resume_from_checkpoint is None:
        return {
            "resume_from_checkpoint": "",
            "resume_from_checkpoint_allowed_cloud_path": False,
            "resume_from_checkpoint_exists": False,
        }
    policy = (
        CloudStoragePolicy((cloud_root, cloud_root.parent), allow_tmp=allow_tmp)
        if cloud_root is not None
        else CloudStoragePolicy((), allow_tmp=allow_tmp)
    )
    return {
        "resume_from_checkpoint": str(resume_from_checkpoint),
        "resume_from_checkpoint_allowed_cloud_path": is_allowed_cloud_path(
            resume_from_checkpoint,
            policy,
        ),
        "resume_from_checkpoint_exists": resume_from_checkpoint.is_dir(),
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
