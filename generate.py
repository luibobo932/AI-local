"""
Generate text from a trained GPT checkpoint.

Usage:
    python generate.py                                       # from ckpt.pt with default prompt
    python generate.py --prompt "To be or not to be"
    python generate.py --temperature 0.8 --top_k 50 --num_tokens 300
    python generate.py --checkpoint checkpoints/ckpt.pt
"""

import argparse
import os
import pickle

import torch

from model.gpt import GPT, GPTConfig


def load_checkpoint(path: str, device: torch.device) -> tuple[GPT, dict | None]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg: GPTConfig = ckpt["model_cfg"]
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, ckpt.get("meta")


def load_meta(data_dir: str) -> dict | None:
    path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def encode_prompt(prompt: str, meta: dict | None) -> list[int]:
    if meta is not None:
        # Character-level encoding
        stoi = meta["stoi"]
        return [stoi[c] for c in prompt if c in stoi]
    else:
        # BPE encoding
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            return enc.encode(prompt)
        except ImportError:
            raise SystemExit("Install tiktoken: pip install tiktoken")


def decode_tokens(ids: list[int], meta: dict | None) -> str:
    if meta is not None:
        itos = meta["itos"]
        return "".join(itos[i] for i in ids)
    else:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            return enc.decode(ids)
        except ImportError:
            raise SystemExit("Install tiktoken: pip install tiktoken")


def generate(
    checkpoint: str = "checkpoints/ckpt.pt",
    prompt: str = "\n",
    num_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 50,
    num_samples: int = 1,
    data_dir: str = "data",
    device_str: str = "auto",
    seed: int = 42,
):
    # Device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    torch.manual_seed(seed)

    # Load model
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run: python data/prepare.py && python train.py"
        )

    print(f"Loading checkpoint: {checkpoint}")
    model, _ = load_checkpoint(checkpoint, device)
    print(f"Model: {model.num_parameters() / 1e6:.2f}M parameters")

    # Load vocab metadata
    meta = load_meta(data_dir)

    # Encode prompt
    tokens = encode_prompt(prompt, meta)
    if not tokens:
        # Start from newline if prompt is empty or unrecognized
        tokens = [0]

    print(f"\n--- Prompt ---\n{prompt}\n--- Generated ---")

    for i in range(num_samples):
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        y = model.generate(x, max_new_tokens=num_tokens, temperature=temperature, top_k=top_k)
        output_tokens = y[0].tolist()
        text = decode_tokens(output_tokens, meta)
        print(text)
        if num_samples > 1:
            print(f"\n{'─' * 60} [{i+1}/{num_samples}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--num_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        num_samples=args.num_samples,
        data_dir=args.data_dir,
        device_str=args.device,
        seed=args.seed,
    )
