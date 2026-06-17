"""
Inspect a trained checkpoint: show config, parameter count, and a sample generation.

Usage:
    python inspect_model.py
    python inspect_model.py --checkpoint checkpoints/ckpt.pt
"""

import argparse
import os
import pickle

import torch

from model.gpt import GPT, GPTConfig


def inspect(checkpoint: str = "checkpoints/ckpt.pt", data_dir: str = "data"):
    if not os.path.exists(checkpoint):
        print(f"No checkpoint found at {checkpoint}")
        print("Run: python data/prepare.py && python train.py")
        return

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg: GPTConfig = ckpt["model_cfg"]

    print("=" * 50)
    print("Model Configuration")
    print("=" * 50)
    for k, v in vars(cfg).items():
        print(f"  {k:<20} {v}")

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("Parameter Count")
    print("=" * 50)
    print(f"  Total:        {total:>12,}")
    print(f"  Trainable:    {trainable:>12,}")
    print(f"  Non-emb:      {model.num_parameters():>12,}")

    print("\n" + "=" * 50)
    print("Training State")
    print("=" * 50)
    print(f"  Iterations:   {ckpt.get('iter_num', '?')}")
    print(f"  Best val loss:{ckpt.get('best_val_loss', '?'):.4f}" if 'best_val_loss' in ckpt else "  Best val loss: ?")

    print("\n" + "=" * 50)
    print("Layer breakdown")
    print("=" * 50)
    for name, p in model.named_parameters():
        print(f"  {name:<45} {str(tuple(p.shape)):<20} {p.numel():>10,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    inspect(args.checkpoint, args.data_dir)
