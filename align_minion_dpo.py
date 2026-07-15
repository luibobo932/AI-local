"""Căn chỉnh adapter Minion bằng DPO sau khi SFT vượt cổng an toàn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", default="data/minion_dpo/train.jsonl")
    parser.add_argument("--eval-data", default="data/minion_dpo/validation.jsonl")
    parser.add_argument("--out", default="models/minion-v4-dpo")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--gate-report", required=True, help="Báo cáo eval SFT phải không còn critical failure")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    data_path, eval_path, gate_path = Path(args.data), Path(args.eval_data), Path(args.gate_report)
    for path in (data_path, eval_path, gate_path, Path(args.adapter)):
        if not path.exists():
            print(f"Không tìm thấy: {path}")
            return 1
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    result = gate.get("adapter_result") or {}
    critical = result.get("critical_failures")
    if critical is None:
        print("Báo cáo gate chưa có adapter_result/critical_failures.")
        return 1
    if critical:
        print(f"Không chạy DPO: SFT còn {len(critical)} critical failure.")
        return 2
    plan = {
        "base": args.base, "adapter": args.adapter, "train_pairs": count_jsonl(data_path),
        "eval_pairs": count_jsonl(eval_path), "out": args.out, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "max_length": args.max_length,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        print("Không có CUDA.")
        return 1
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quantization, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    training = DPOConfig(
        output_dir=args.out, max_length=args.max_length, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, num_train_epochs=args.epochs, logging_steps=5,
        eval_strategy="steps", eval_steps=25, save_steps=25, save_total_limit=2,
        bf16=True, gradient_checkpointing=True, report_to="none",
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=training, processing_class=tokenizer,
        train_dataset=load_dataset("json", data_files=str(data_path), split="train"),
        eval_dataset=load_dataset("json", data_files=str(eval_path), split="train"),
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Đã lưu adapter DPO: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
