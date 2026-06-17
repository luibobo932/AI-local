"""
Giai đoạn 3 — DPO (Direct Preference Optimization).

Căn chỉnh model SFT bằng dữ liệu preference mà không cần reward model riêng biệt.
Implement DPO loss từ paper: Rafailov et al., "Direct Preference Optimization" (2023).

DPO Loss:
    L = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                   - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))]

    y_w = chosen response,  y_l = rejected response
    π_θ = policy (được train),  π_ref = reference / SFT model (frozen)
    β = temperature (thường 0.1)

Usage:
    python data/prepare_dpo.py                       # chuẩn bị dữ liệu trước
    python align_dpo.py                               # align với default settings
    python align_dpo.py --sft_checkpoint checkpoints/sft_ckpt.pt
"""

import argparse
import contextlib
import copy
import math
import os
import pickle
import random
import time

import torch
import torch.nn.functional as F

from config import GPTConfig
from model.gpt import GPT


# ─── Hyperparameters DPO ─────────────────────────────────────────────────────

class DPOConfig:
    sft_checkpoint: str = "checkpoints/sft_ckpt.pt"
    data_dir: str = "data"
    out_dir: str = "checkpoints"
    out_name: str = "dpo_ckpt.pt"

    max_iters: int = 500
    eval_interval: int = 50
    eval_iters: int = 10
    log_interval: int = 25

    batch_size: int = 4    # DPO cần ít hơn vì mỗi example xử lý 2 lần (chosen + rejected)

    beta: float = 0.1       # DPO temperature — điều chỉnh độ lệch so với reference

    learning_rate: float = 5e-5   # rất nhỏ để tránh catastrophic forgetting
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    warmup_iters: int = 25
    min_lr: float = 5e-6

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


def cosine_lr(step: int, cfg: DPOConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(cfg.warmup_iters, 1)
    ratio = (step - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def load_dpo_data(data_dir: str, split: str) -> list[dict]:
    path = os.path.join(data_dir, f"dpo_{split}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DPO data not found: {path}\nRun: python data/prepare_dpo.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_to(ids: list[int], length: int, pad_val: int = 0) -> list[int]:
    return ids[:length] + [pad_val] * max(0, length - len(ids))


def get_batch(
    examples: list[dict], batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (chosen_ids, chosen_labels, rejected_ids, rejected_labels)."""
    batch = random.sample(examples, min(batch_size, len(examples)))

    c_ids_list, c_lbl_list, r_ids_list, r_lbl_list = [], [], [], []
    for ex in batch:
        c_ids_list.append(pad_to(ex["chosen_ids"],    block_size))
        c_lbl_list.append(pad_to(ex["chosen_labels"], block_size, -1))
        r_ids_list.append(pad_to(ex["rejected_ids"],    block_size))
        r_lbl_list.append(pad_to(ex["rejected_labels"], block_size, -1))

    to_t = lambda lst: torch.tensor(lst, dtype=torch.long, device=device)
    return to_t(c_ids_list), to_t(c_lbl_list), to_t(r_ids_list), to_t(r_lbl_list)


def compute_log_probs(
    model: GPT, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute the sum of log-probabilities for response tokens (where labels != -1).

    input_ids: (B, T)
    labels:    (B, T)  — -1 for prompt/padding tokens
    Returns:   (B,)    — scalar per example
    """
    logits = model.forward_full(input_ids)          # (B, T, V)
    # Shift: predict token t+1 from token t
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()       # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # Gather log prob at the correct token position
    # clamp labels to avoid indexing error on -1 positions
    safe_labels = shift_labels.clamp(min=0)
    token_log_probs = log_probs.gather(
        2, safe_labels.unsqueeze(2)
    ).squeeze(2)                                     # (B, T-1)

    # Mask out prompt / padding positions
    mask = (shift_labels != -1).float()
    return (token_log_probs * mask).sum(dim=-1)      # (B,)


def dpo_loss(
    policy_log_probs_chosen:   torch.Tensor,
    policy_log_probs_rejected: torch.Tensor,
    ref_log_probs_chosen:      torch.Tensor,
    ref_log_probs_rejected:    torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DPO loss (Rafailov et al., 2023).
    Returns: (loss, reward_margin) where reward_margin > 0 means chosen > rejected.
    """
    log_ratio_chosen   = policy_log_probs_chosen   - ref_log_probs_chosen
    log_ratio_rejected = policy_log_probs_rejected - ref_log_probs_rejected

    reward_margin = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -F.logsigmoid(reward_margin).mean()
    return loss, reward_margin.mean()


# ─── Main ─────────────────────────────────────────────────────────────────────

def align_dpo(cfg: DPOConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = get_device(cfg.device)
    print(f"Device: {device}")

    # Load data
    train_data = load_dpo_data(cfg.data_dir, "train")
    val_data   = load_dpo_data(cfg.data_dir, "val")
    print(f"DPO data: {len(train_data)} train  |  {len(val_data)} val preference pairs")

    # Load SFT model
    if not os.path.exists(cfg.sft_checkpoint):
        raise FileNotFoundError(
            f"SFT checkpoint not found: {cfg.sft_checkpoint}\n"
            "Run: python data/prepare_sft.py && python finetune_sft.py  first."
        )
    print(f"Loading SFT model: {cfg.sft_checkpoint}")
    ckpt = torch.load(cfg.sft_checkpoint, map_location=device, weights_only=False)
    model_cfg: GPTConfig = ckpt["model_cfg"]
    block_size = model_cfg.block_size

    # Policy model: will be trained
    policy = GPT(model_cfg).to(device)
    policy.load_state_dict(ckpt["model"])

    # Reference model: frozen copy of SFT model
    ref_model = GPT(model_cfg).to(device)
    ref_model.load_state_dict(ckpt["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"Policy + Reference model: {policy.num_parameters() / 1e6:.2f}M params each")
    print(f"β (DPO temperature): {cfg.beta}")

    # Optimizer
    decay_params   = [p for n, p in policy.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in policy.named_parameters() if p.dim() < 2]
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
    policy.train()

    print("\n--- DPO Alignment ---")
    for step in range(cfg.max_iters + 1):
        lr = cosine_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            policy.eval()
            val_losses, val_margins = [], []
            with torch.no_grad():
                for _ in range(min(cfg.eval_iters, len(val_data))):
                    c_ids, c_lbl, r_ids, r_lbl = get_batch(
                        val_data, cfg.batch_size, block_size, device
                    )
                    with ctx:
                        pi_lp_c  = compute_log_probs(policy,    c_ids, c_lbl)
                        pi_lp_r  = compute_log_probs(policy,    r_ids, r_lbl)
                        ref_lp_c = compute_log_probs(ref_model, c_ids, c_lbl)
                        ref_lp_r = compute_log_probs(ref_model, r_ids, r_lbl)
                    loss, margin = dpo_loss(pi_lp_c, pi_lp_r, ref_lp_c, ref_lp_r, cfg.beta)
                    val_losses.append(loss.item())
                    val_margins.append(margin.item())

            val_loss   = sum(val_losses)  / max(len(val_losses), 1)
            val_margin = sum(val_margins) / max(len(val_margins), 1)
            print(
                f"Step {step:5d} | val loss {val_loss:.4f} | "
                f"reward margin {val_margin:+.4f} | lr {lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                dpo_ckpt = {
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cfg": model_cfg,
                    "iter_num": step,
                    "best_val_loss": best_val_loss,
                    "stage": "dpo",
                    "dpo_beta": cfg.beta,
                }
                path = os.path.join(cfg.out_dir, cfg.out_name)
                torch.save(dpo_ckpt, path)
                print(f"  Checkpoint saved → {path}")

            policy.train()

        if step == cfg.max_iters:
            break

        # ── Training step ──────────────────────────────────────────────────────
        c_ids, c_lbl, r_ids, r_lbl = get_batch(
            train_data, cfg.batch_size, block_size, device
        )
        with ctx:
            pi_lp_c  = compute_log_probs(policy,    c_ids, c_lbl)
            pi_lp_r  = compute_log_probs(policy,    r_ids, r_lbl)
            with torch.no_grad():
                ref_lp_c = compute_log_probs(ref_model, c_ids, c_lbl)
                ref_lp_r = compute_log_probs(ref_model, r_ids, r_lbl)

            loss, margin = dpo_loss(pi_lp_c, pi_lp_r, ref_lp_c, ref_lp_r, cfg.beta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(
                f"  step {step:5d} | loss {loss.item():.4f} | margin {margin.item():+.4f} | {dt*1000:.0f}ms",
                end="\r",
            )
            t0 = time.time()

    print(f"\nDPO alignment complete. Best val loss: {best_val_loss:.4f}")
    print(f"Aligned model: {os.path.join(cfg.out_dir, cfg.out_name)}")
    print("\nChạy inference:")
    print('  python generate.py --checkpoint checkpoints/dpo_ckpt.pt --chat')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> DPOConfig:
    cfg = DPOConfig()
    parser = argparse.ArgumentParser()
    fields = {k: v for k, v in vars(DPOConfig).items() if not k.startswith("_")}
    for field, val in fields.items():
        parser.add_argument(f"--{field}", type=type(val) if val != "" else str, default=val)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    align_dpo(parse_args())
