"""Fine-tune bộ não Minion bằng SFT + QLoRA trên dữ liệu hội thoại chuẩn."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def validate_dataset(path: Path) -> tuple[int, list[str]]:
    """Kiểm tra schema trước khi tải model để tránh tốn thời gian và VRAM."""
    errors: list[str] = []
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Dòng {line_number}: JSON không hợp lệ ({exc})")
                continue
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"Dòng {line_number}: cần mảng messages có ít nhất 2 phần tử")
                continue
            roles = [message.get("role") for message in messages if isinstance(message, dict)]
            if "user" not in roles or "assistant" not in roles:
                errors.append(f"Dòng {line_number}: cần có cả role user và assistant")
            for index, message in enumerate(messages, start=1):
                if not isinstance(message, dict) or not message.get("role"):
                    errors.append(f"Dòng {line_number}, message {index}: thiếu role")
                if not isinstance(message, dict) or (
                    "content" not in message and "tool_calls" not in message
                ):
                    errors.append(f"Dòng {line_number}, message {index}: thiếu content hoặc tool_calls")
    return count, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT QLoRA cho Minion")
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", required=True, help="File JSONL có cột messages và tools tùy chọn")
    parser.add_argument("--out", default="models/minion-sft")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="", help="Nơi lưu báo cáo train JSON (tùy chọn)")
    parser.add_argument("--resume-from-checkpoint", default="", help="Checkpoint Trainer để tiếp tục một lượt train")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Không tìm thấy dữ liệu: {data_path}")
        return 1

    count, errors = validate_dataset(data_path)
    if errors:
        print("Dữ liệu chưa đạt:")
        for error in errors[:50]:
            print(f"- {error}")
        return 1
    if count < 10 and not args.dry_run:
        print(f"Chỉ có {count} ví dụ. Cần tối thiểu 10 ví dụ để tránh train nhầm dữ liệu mẫu.")
        return 1

    plan = {
        "base": args.base,
        "data": str(data_path),
        "examples": count,
        "output": args.out,
        "qlora": "4-bit NF4 + double quant",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "resume_from_checkpoint": args.resume_from_checkpoint or None,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry-run đạt: schema và cấu hình hợp lệ, chưa tải model hoặc train.")
        return 0

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("PyTorch chưa nhận CUDA. Hãy chạy setup_minion_gpu.ps1 trước.")
        return 1

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        use_rslora=True,
    )
    training = SFTConfig(
        output_dir=args.out,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=args.base,
        args=training,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        quantization_config=quantization,
        peft_config=lora,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    eval_metrics = trainer.evaluate()
    trainer.save_model(args.out)
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(args.out)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": args.base,
            "adapter_path": str(Path(args.out)),
            "dataset": str(data_path),
            "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "examples": count,
            "gpu": torch.cuda.get_device_name(0),
            "configuration": plan,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu báo cáo train: {report_path}")
    print(f"Đã lưu LoRA adapter của Minion tại: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
