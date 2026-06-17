"""
Giai đoạn 2 — SFT (Supervised Fine-Tuning).

Load model đã pretrain và fine-tune trên dữ liệu instruction-following.
Chỉ tính loss trên phần response (prompt bị mask bằng -1).

Usage:
    python data/prepare_sft.py                    # chuẩn bị dữ liệu trước
    python finetune_sft.py                         # fine-tune với default settings
    python finetune_sft.py --max_iters 2000        # nhiều bước hơn
    python finetune_sft.py --pretrain_checkpoint checkpoints/ckpt.pt
"""

import argparse
import contextlib
import math
import os
import pickle
import random
import time

import torch
import torch.nn.functional as F

from config import GPTConfig
from model.gpt import GPT


# ─── Hyperparameters SFT ─────────────────────────────────────────────────────

class SFTConfig:
    pretrain_checkpoint: str = "checkpoints/ckpt.pt"
    data_dir: str = "data"
    out_dir: str = "checkpoints"
    out_name: str = "sft_ckpt.pt"

    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 50

    batch_size: int = 8
    block_size: int = 256

    learning_rate: float = 1e-4   # lower than pretraining
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    warmup_iters: int = 50
    decay_lr: bool = True
    min_lr: float = 1e-5

    device: str = "auto"


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def cosine_lr(step: int, cfg: SFTConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(cfg.warmup_iters, 1)
    ratio = (step - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def load_sft_data(data_dir: str, split: str) -> list[dict]:
    path = os.path.join(data_dir, f"sft_{split}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SFT data not found: {path}\nRun: python data/prepare_sft.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_sequence(ids: list[int], length: int, pad_val: int = 0) -> list[int]:
    return ids[:length] + [pad_val] * max(0, length - len(ids))


def get_batch(
    examples: list[dict], batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input_ids, labels) with padding."""
    batch = random.sample(examples, min(batch_size, len(examples)))

    xs, ys = [], []
    for ex in batch:
        ids = pad_sequence(ex["input_ids"], block_size + 1)
        lbl = pad_sequence(ex["labels"], block_size + 1, pad_val=-1)

        xs.append(ids[:block_size])
        ys.append(lbl[1 : block_size + 1])   # shift labels by 1

    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def estimate_val_loss(
    model: GPT,
    val_data: list[dict],
    cfg: SFTConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(min(cfg.eval_iters, len(val_data))):
        x, y = get_batch(val_data, cfg.batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def finetune_sft(cfg: SFTConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = get_device(cfg.device)
    print(f"Device: {device}")

    # Load data
    train_data = load_sft_data(cfg.data_dir, "train")
    val_data   = load_sft_data(cfg.data_dir, "val")
    print(f"SFT data: {len(train_data)} train  |  {len(val_data)} val examples")

    # Load pretrained model
    if not os.path.exists(cfg.pretrain_checkpoint):
        raise FileNotFoundError(
            f"Pretrain checkpoint not found: {cfg.pretrain_checkpoint}\n"
            "Run: python data/prepare.py && python train.py  first."
        )
    print(f"Loading pretrained model: {cfg.pretrain_checkpoint}")
    ckpt = torch.load(cfg.pretrain_checkpoint, map_location=device, weights_only=False)
    model_cfg: GPTConfig = ckpt["model_cfg"]
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Model: {model.num_parameters() / 1e6:.2f}M params | block_size={model_cfg.block_size}")

    # Optimizer — lower LR than pretraining
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )

    ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    best_val_loss = float("inf")
    t0 = time.time()
    model.train()

    print("\n--- SFT Training ---")
    for step in range(cfg.max_iters + 1):
        lr = cosine_lr(step, cfg) if cfg.decay_lr else cfg.learning_rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step % cfg.eval_interval == 0:
            val_loss = estimate_val_loss(model, val_data, cfg, device)
            print(f"Step {step:5d} | val loss {val_loss:.4f} | lr {lr:.2e}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_out = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cfg": model_cfg,
                    "iter_num": step,
                    "best_val_loss": best_val_loss,
                    "stage": "sft",
                }
                path = os.path.join(cfg.out_dir, cfg.out_name)
                torch.save(ckpt_out, path)
                print(f"  Checkpoint saved → {path}")

        if step == cfg.max_iters:
            break

        x, y = get_batch(train_data, cfg.batch_size, model_cfg.block_size, device)
        with ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(f"  step {step:5d} | loss {loss.item():.4f} | {dt*1000:.0f}ms", end="\r")
            t0 = time.time()

    print(f"\nSFT complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {os.path.join(cfg.out_dir, cfg.out_name)}")
    print("\nNext: python data/prepare_dpo.py && python align_dpo.py")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> SFTConfig:
    cfg = SFTConfig()
    parser = argparse.ArgumentParser()
    fields = {k: v for k, v in vars(SFTConfig).items() if not k.startswith("_")}
    for field, val in fields.items():
        parser.add_argument(f"--{field}", type=type(val) if val != "" else str, default=val)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    finetune_sft(parse_args())
