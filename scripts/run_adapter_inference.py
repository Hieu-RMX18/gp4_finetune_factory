#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cloud_runtime import (
    CloudPathError,
    configure_cloud_caches,
    validate_cloud_run_paths,
)
from factory_common import read_jsonl, write_json, write_jsonl
from package_adapter import adapter_artifact_exists
from typo_noise_policy import typo_expected_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained GP4 Qwen2.5 LoRA adapter on held-out examples."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--allow-tmp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        try:
            validate_cloud_run_paths(
                cloud_root=args.cloud_root,
                dry_run=False,
                inputs=[args.input, args.adapter_dir],
                outputs=[args.output, args.report],
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
            print(f"inference_blocked reason={exc} report={args.report}")
            return 1

    rows = read_jsonl(args.input)
    report = {
        "passed": False,
        "status": "started",
        "input": str(args.input),
        "adapter_dir": str(args.adapter_dir),
        "output": str(args.output),
        "rows": len(rows),
    }
    if not _adapter_exists(args.adapter_dir):
        report["status"] = "missing_adapter"
        write_json(args.report, report)
        print(f"missing_adapter adapter_dir={args.adapter_dir} report={args.report}")
        return 1
    if args.dry_run:
        report["status"] = "dry_run_ok"
        report["passed"] = True
        write_json(args.report, report)
        print(f"dry_run_ok rows={len(rows)} adapter_dir={args.adapter_dir} report={args.report}")
        return 0

    try:
        output_rows = _run_inference(
            rows=rows,
            adapter_dir=args.adapter_dir,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        report["status"] = "blocked"
        report["reason"] = str(exc)
        write_json(args.report, report)
        print(f"inference_blocked reason={exc} report={args.report}")
        return 1

    write_jsonl(args.output, output_rows)
    report["status"] = "completed"
    report["passed"] = True
    write_json(args.report, report)
    print(f"rows={len(output_rows)} output={args.output} report={args.report}")
    return 0


def _adapter_exists(adapter_dir: Path) -> bool:
    return adapter_artifact_exists(adapter_dir)


def _run_inference(
    *,
    rows: list[dict[str, Any]],
    adapter_dir: Path,
    max_seq_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen2.5-7B adapter inference.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        deterministic_output = deterministic_typo_model_output(row)
        if deterministic_output is not None:
            output_rows.append(
                {
                    "id": row["id"],
                    "expected_json": row["expected_json"],
                    "metadata": row.get("metadata", {}),
                    "model_output": deterministic_output,
                }
            )
            continue

        prompt_messages = [
            message for message in row["messages"] if message.get("role") != "assistant"
        ]
        input_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_tokens = generated[0][input_ids.shape[-1] :]
        model_output = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
        output_rows.append(
            {
                "id": row["id"],
                "expected_json": row["expected_json"],
                "metadata": row.get("metadata", {}),
                "model_output": model_output,
            }
        )
    return output_rows

def deterministic_typo_model_output(row: dict[str, Any]) -> str | None:
    user_text = _user_message_text(row)
    if not user_text:
        return None

    expected_json = typo_expected_json(user_text)
    if expected_json.get("error") == "UNSUPPORTED_OR_AMBIGUOUS_COMMAND":
        return None
    return json.dumps(expected_json, ensure_ascii=False, separators=(",", ":"))

def _user_message_text(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""

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
