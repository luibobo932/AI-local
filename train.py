"""
Training script for the GPT language model.

Quick start:
    python data/prepare.py          # prepare Shakespeare data
    python train.py                 # train with default settings (~5min on GPU)
    python train.py --max_iters 500 # quick smoke test on CPU

Resume from checkpoint:
    python train.py --resume_from checkpoints/ckpt.pt
"""

import argparse
import contextlib
import math
import os
import pickle
import time

import numpy as np
import torch

from config import GPTConfig, TrainConfig
from model.gpt import GPT


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def get_dtype(cfg: TrainConfig, device: torch.device):
    if cfg.dtype == "bfloat16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if cfg.dtype == "float16" and device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def load_data(path: str, device: torch.device) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    return data


def get_batch(
    data: np.ndarray, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / cfg.warmup_iters
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (step - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, model.cfg.block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = get_device(cfg)
    dtype = get_dtype(cfg, device)
    print(f"Device: {device}  |  dtype: {dtype}")

    # Data
    train_data = load_data(os.path.join(cfg.data_dir, "train.bin"), device)
    val_data = load_data(os.path.join(cfg.data_dir, "val.bin"), device)

    # Vocabulary size: from meta.pkl (char) or fixed 50257 (bpe)
    meta_path = os.path.join(cfg.data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        print(f"Char-level vocab: {vocab_size} tokens")
    else:
        vocab_size = 50304   # BPE padded
        print(f"BPE vocab: {vocab_size} tokens")

    # Model
    model_cfg = GPTConfig(vocab_size=vocab_size)
    iter_num = 0
    best_val_loss = float("inf")

    if cfg.resume_from:
        print(f"Resuming from {cfg.resume_from}")
        checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model_cfg = checkpoint["model_cfg"]
        model = GPT(model_cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        model = GPT(model_cfg).to(device)

    if cfg.compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    print(f"Parameters: {model.num_parameters() / 1e6:.2f}M")

    # Optimizer — separate weight decay from non-decay params
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )
    if cfg.resume_from:
        optimizer.load_state_dict(checkpoint["optimizer"])

    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type != "cpu"
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == torch.float16 and device.type == "cuda"))
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_amp else contextlib.nullcontext()

    # Training loop
    x, y = get_batch(train_data, cfg.batch_size, model_cfg.block_size, device)
    t0 = time.time()

    print("\n--- Training ---")
    while iter_num < cfg.max_iters:
        # Learning rate schedule
        lr = cosine_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation
        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            print(
                f"Step {iter_num:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | lr {lr:.2e}"
            )

            if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cfg": model_cfg,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                path = os.path.join(cfg.out_dir, "ckpt.pt")
                torch.save(ckpt, path)
                print(f"  Checkpoint saved → {path}")

        # Forward + backward with gradient accumulation
        for micro_step in range(cfg.gradient_accumulation_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            x, y = get_batch(train_data, cfg.batch_size, model_cfg.block_size, device)
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            loss_val = loss.item() * cfg.gradient_accumulation_steps
            print(f"  iter {iter_num:5d} | loss {loss_val:.4f} | {dt*1000:.0f}ms/iter", end="\r")
            t0 = time.time()

        iter_num += 1

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(cfg.out_dir, 'ckpt.pt')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    cfg = TrainConfig()
    for field, value in vars(cfg).items():
        parser.add_argument(f"--{field}", type=type(value) if value != "" else str, default=value)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
