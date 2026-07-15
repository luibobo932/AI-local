"""Căn chỉnh adapter Minion bằng DPO với khóa triển khai fail-closed."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def assess_training_gate(gate: dict, repair_mode: bool = False) -> dict:
    """Quyết định có được train DPO hay không; không quyết định triển khai model."""
    result = gate.get("adapter_result") or {}
    critical = result.get("critical_failures")
    if critical is None:
        return {
            "allowed": False,
            "exit_code": 1,
            "reason": "Báo cáo gate chưa có adapter_result/critical_failures.",
            "critical_failures": [],
            "deployment_locked": True,
        }
    if critical and not repair_mode:
        return {
            "allowed": False,
            "exit_code": 2,
            "reason": f"Không chạy DPO chuẩn: SFT còn {len(critical)} critical failure.",
            "critical_failures": critical,
            "deployment_locked": True,
        }
    reason = (
        f"Cho phép DPO repair với {len(critical)} critical failure; bắt buộc đánh giá lại trước khi triển khai."
        if critical
        else "SFT đã vượt điều kiện critical của gate; cho phép DPO chuẩn."
    )
    return {
        "allowed": True,
        "exit_code": 0,
        "reason": reason,
        "critical_failures": critical,
        "deployment_locked": True,
    }


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
    parser.add_argument("--gate-report", required=True, help="Báo cáo eval của adapter đầu vào")
    parser.add_argument(
        "--repair-mode",
        action="store_true",
        help="Cho phép train sửa critical failure nhưng vẫn khóa triển khai",
    )
    parser.add_argument("--report", default="", help="Nơi lưu báo cáo train JSON")
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    data_path = Path(args.data)
    eval_path = Path(args.eval_data)
    gate_path = Path(args.gate_report)
    adapter_path = Path(args.adapter).resolve()
    output_path = Path(args.out).resolve()
    for path in (data_path, eval_path, gate_path, adapter_path):
        if not path.exists():
            print(f"Không tìm thấy: {path}")
            return 1
    if adapter_path == output_path:
        print("Không được ghi DPO đè lên adapter đầu vào.")
        return 1

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_decision = assess_training_gate(gate, args.repair_mode)
    print(gate_decision["reason"])
    if not gate_decision["allowed"]:
        return gate_decision["exit_code"]

    plan = {
        "base": args.base,
        "adapter": str(adapter_path),
        "train_pairs": count_jsonl(data_path),
        "eval_pairs": count_jsonl(eval_path),
        "out": str(output_path),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "repair_mode": args.repair_mode,
        "deployment_locked": True,
        "source_critical_failures": len(gate_decision["critical_failures"]),
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
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=quantization,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    training = DPOConfig(
        output_dir=str(output_path),
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training,
        processing_class=tokenizer,
        train_dataset=load_dataset("json", data_files=str(data_path), split="train"),
        eval_dataset=load_dataset("json", data_files=str(eval_path), split="train"),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "trained_candidate_deployment_locked",
            "configuration": plan,
            "gate_decision": gate_decision,
            "gpu": torch.cuda.get_device_name(0),
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "required_next_step": "Chạy frozen eval và gate; không tự động tích hợp vào Minion.",
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu báo cáo train: {report_path}")
    print(f"Đã lưu adapter DPO candidate: {output_path}")
    print("Adapter vẫn bị khóa triển khai cho đến khi frozen eval vượt gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
