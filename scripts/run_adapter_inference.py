#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from factory_common import read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/splits/test.jsonl"
DEFAULT_ADAPTER = ROOT / "models/qwen25_gp4_lora_pilot"
DEFAULT_OUTPUT = ROOT / "outputs/model_outputs.jsonl"
DEFAULT_REPORT = ROOT / "reports/inference_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained GP4 Qwen2.5 LoRA adapter on held-out examples."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    report = {
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
    write_json(args.report, report)
    print(f"rows={len(output_rows)} output={args.output} report={args.report}")
    return 0


def _adapter_exists(adapter_dir: Path) -> bool:
    return (
        adapter_dir.joinpath("adapter_config.json").exists()
        or any(adapter_dir.glob("*.safetensors"))
        or any(adapter_dir.glob("*.bin"))
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
