"""
Fine-tune một model PRE-TRAINED (GPT-2, Llama, model tiếng Việt...) trên
dữ liệu của riêng bạn — "lấy về để làm của riêng".

Đây là cách thực tế nhất để có một LLM nói chuyện tốt: thay vì train từ đầu
(tốn hàng triệu USD), ta lấy model người ta đã train sẵn rồi dạy thêm cho nó
phong cách / kiến thức / nhân cách riêng.

Hai chế độ:
  • Full fine-tune  — cập nhật toàn bộ weights (cho model nhỏ: gpt2, distilgpt2)
  • LoRA            — chỉ train vài % weights (cho model lớn, máy yếu) — cần `peft`

Usage:
    # Fine-tune GPT-2 trên dữ liệu instruction có sẵn
    python finetune_hf.py --base gpt2 --out my-gpt2

    # Fine-tune trên dữ liệu của bạn (JSONL: {"instruction":..., "response":...})
    python finetune_hf.py --base gpt2 --data data/my_data.jsonl --out my-assistant

    # LoRA fine-tune model lớn (cần: pip install peft)
    python finetune_hf.py --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --data data/my_data.jsonl --out my-tinyllama --lora

Sau đó:
    python cli.py run my-assistant        # chat với model của riêng bạn
    python cli.py serve                   # serve qua REST API
"""

import argparse
import json
import math
import os

import torch
from torch.nn import functional as F


PROMPT_TEMPLATE = "Human: {instruction}\n\nAssistant: {response}"
PROMPT_PREFIX   = "Human: {instruction}\n\nAssistant: "

MODELS_DIR = "models"


# ─── Dữ liệu ──────────────────────────────────────────────────────────────────

def load_dataset(data_path: str | None) -> list[dict]:
    """Đọc dữ liệu fine-tune. Nếu không có file, dùng dataset có sẵn từ prepare_sft."""
    if data_path and os.path.exists(data_path):
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        print(f"Đã đọc {len(examples)} examples từ {data_path}")
        return examples

    # Fallback: dùng built-in dataset từ data/prepare_sft.py
    print("Không có --data, dùng dataset instruction có sẵn (BUILTIN_SFT_DATA)")
    import sys
    sys.path.insert(0, "data")
    from prepare_sft import BUILTIN_SFT_DATA
    return BUILTIN_SFT_DATA


def build_batches(examples, tokenizer, block_size, batch_size, device):
    """Token hóa với prompt masking (labels=-100 cho phần prompt)."""
    tokenized = []
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for ex in examples:
        prompt = PROMPT_PREFIX.format(instruction=ex["instruction"])
        full   = PROMPT_TEMPLATE.format(instruction=ex["instruction"], response=ex["response"])

        prompt_ids = tokenizer.encode(prompt)
        full_ids   = tokenizer.encode(full) + [eos]
        full_ids   = full_ids[:block_size]

        labels = list(full_ids)
        # Mask phần prompt — chỉ học cách trả lời, không học cách hỏi
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        tokenized.append((full_ids, labels))

    # Tạo batch (pad tới độ dài lớn nhất trong batch)
    batches = []
    for i in range(0, len(tokenized), batch_size):
        chunk = tokenized[i:i + batch_size]
        maxlen = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), maxlen), eos, dtype=torch.long)
        y = torch.full((len(chunk), maxlen), -100, dtype=torch.long)
        for j, (ids, labels) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            y[j, :len(labels)] = torch.tensor(labels, dtype=torch.long)
        batches.append((x.to(device), y.to(device)))
    return batches


# ─── Fine-tune ────────────────────────────────────────────────────────────────

def finetune(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Base model: {cfg.base}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.base)
    model.to(device)

    # LoRA — chỉ train một phần nhỏ weights
    if cfg.lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            print("LoRA cần thư viện peft. Chạy: pip install peft")
            return
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M "
          f"({100*trainable/total:.1f}%)")

    examples = load_dataset(cfg.data)
    batches = build_batches(examples, tokenizer, cfg.block_size, cfg.batch_size, device)
    print(f"Số batch mỗi epoch: {len(batches)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=0.01,
    )

    total_steps = cfg.epochs * len(batches)
    step = 0
    print(f"\n--- Fine-tuning {cfg.epochs} epochs ({total_steps} steps) ---")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for x, y in batches:
            # Cosine LR với warmup
            if step < cfg.warmup_steps:
                lr = cfg.learning_rate * (step + 1) / cfg.warmup_steps
            else:
                ratio = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
                lr = cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * ratio))
            for g in optimizer.param_groups:
                g["lr"] = lr

            out = model(input_ids=x, labels=y)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

        avg = epoch_loss / max(len(batches), 1)
        print(f"  epoch {epoch+1:3d}/{cfg.epochs} | loss {avg:.4f} | lr {lr:.2e}")

    # ─── Lưu model ───
    out_dir = os.path.join(MODELS_DIR, cfg.out)
    os.makedirs(out_dir, exist_ok=True)

    if cfg.lora and cfg.merge:
        print("Merge LoRA weights vào base model...")
        model = model.merge_and_unload()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Ghi metadata để cli/server nhận diện
    with open(os.path.join(out_dir, "ai_local.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base": cfg.base,
            "stage": "finetuned-lora" if cfg.lora else "finetuned",
            "examples": len(examples),
            "epochs": cfg.epochs,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Model của bạn đã lưu tại: {out_dir}/")
    print(f"\nChạy thử:")
    print(f"  python cli.py run {cfg.out}")
    print(f"  python cli.py serve   →  dùng model '{cfg.out}' trong API")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune pre-trained model thành của riêng bạn")
    p.add_argument("--base", default="gpt2",
                   help="Model pre-trained làm gốc (gpt2, distilgpt2, repo/model-id, hoặc models/<dir>)")
    p.add_argument("--data", default="",
                   help="File JSONL dữ liệu của bạn ({\"instruction\":..,\"response\":..}). Bỏ trống = dùng data có sẵn")
    p.add_argument("--out", default="my-model", help="Tên model output (lưu vào models/<out>/)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--lora", action="store_true", help="Dùng LoRA (tiết kiệm RAM, cho model lớn)")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--merge", action="store_true", help="Merge LoRA vào base khi lưu (chạy độc lập, không cần peft)")
    return p.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
