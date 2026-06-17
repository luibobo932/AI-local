"""
Data preparation: downloads Shakespeare dataset (or any text file)
and tokenizes it into binary .bin files for fast training.

Usage:
    python data/prepare.py                          # Shakespeare (default)
    python data/prepare.py --input my_text.txt     # custom text file
    python data/prepare.py --tokenizer bpe          # use BPE (tiktoken gpt2)
"""

import argparse
import os
import struct
import urllib.request
import numpy as np


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_shakespeare(dest: str) -> str:
    path = os.path.join(dest, "shakespeare.txt")
    if not os.path.exists(path):
        print("Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        print(f"Saved to {path}")
    return path


def char_tokenizer(text: str) -> tuple[np.ndarray, dict, dict]:
    """Character-level tokenizer. Returns (token_ids, stoi, itos)."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    return ids, stoi, itos


def bpe_tokenizer(text: str) -> tuple[np.ndarray, None, None]:
    """BPE tokenizer using tiktoken (GPT-2 vocab, 50257 tokens)."""
    try:
        import tiktoken
    except ImportError:
        raise SystemExit("Run: pip install tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    return ids, None, None


def save_bin(ids: np.ndarray, path: str):
    """Save token ids as a flat binary file (uint16)."""
    ids.tofile(path)
    print(f"Saved {len(ids):,} tokens → {path}")


def save_meta(vocab_size: int, stoi: dict, itos: dict, path: str):
    import pickle
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved vocab metadata → {path}  (vocab_size={vocab_size})")


def prepare(
    input_path: str | None = None,
    out_dir: str = "data",
    tokenizer: str = "char",
    train_split: float = 0.9,
):
    os.makedirs(out_dir, exist_ok=True)

    # Acquire text
    if input_path is None:
        input_path = download_shakespeare(out_dir)

    print(f"Reading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,} characters")

    # Tokenize
    if tokenizer == "char":
        ids, stoi, itos = char_tokenizer(text)
        vocab_size = len(stoi)
        save_meta(vocab_size, stoi, itos, os.path.join(out_dir, "meta.pkl"))
    elif tokenizer == "bpe":
        ids, _, _ = bpe_tokenizer(text)
        vocab_size = 50257
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    print(f"  {len(ids):,} tokens  |  vocab_size={vocab_size}")

    # Train / val split
    n = int(train_split * len(ids))
    train_ids = ids[:n]
    val_ids = ids[n:]

    save_bin(train_ids, os.path.join(out_dir, "train.bin"))
    save_bin(val_ids, os.path.join(out_dir, "val.bin"))

    print("\nDone. Run `python train.py` to start training.")
    return vocab_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to custom text file")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--train_split", type=float, default=0.9)
    args = parser.parse_args()

    prepare(
        input_path=args.input,
        out_dir=args.out_dir,
        tokenizer=args.tokenizer,
        train_split=args.train_split,
    )
