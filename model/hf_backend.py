"""
HuggingFace backend — wraps any HF text-generation model in the same
interface as our custom GPT so server.py can serve both transparently.

Supported model IDs (examples):
    gpt2, gpt2-medium, gpt2-large, gpt2-xl
    distilgpt2
    openai-community/gpt2
    microsoft/phi-2
    TinyLlama/TinyLlama-1.1B-Chat-v1.0

Usage:
    from model.hf_backend import HFModel
    model = HFModel("gpt2")
    for token in model.generate_iter(idx, 200, temperature=0.8, top_k=50):
        ...
"""

from __future__ import annotations
import os
import torch


MODELS_DIR = "models"

# HF model IDs we treat as "known" for auto-detect
_HF_ALIASES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "distilgpt2": "distilgpt2",
}


def local_model_dir(name: str) -> str | None:
    """Nếu `name` là một model fine-tune lưu trong models/<name>/, trả về đường dẫn."""
    d = os.path.join(MODELS_DIR, name)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
        return d
    # Đường dẫn trực tiếp tới một thư mục HF model
    if os.path.isdir(name) and os.path.exists(os.path.join(name, "config.json")):
        return name
    return None


def is_hf_model(name: str) -> bool:
    """Return True if name looks like a HuggingFace model id or a local HF dir."""
    if not name:
        return False
    if name in _HF_ALIASES:
        return True
    if local_model_dir(name):
        return True
    # repo/model-name format (e.g. microsoft/phi-2)
    if "/" in name and not name.startswith("/") and not name.endswith(".pt"):
        return True
    return False


class HFModel:
    """
    Thin wrapper around a HuggingFace AutoModelForCausalLM that exposes
    the same generate_iter() interface as our GPT class.
    """

    def __init__(self, model_id: str, device: torch.device | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Model fine-tune local trong models/<name>/ — load từ thư mục đó
        resolved = local_model_dir(model_id) or model_id

        self.model_id = model_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        print(f"Loading HuggingFace model: {resolved} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.hf_model.eval()

        # Expose a GPTConfig-like object so server.py can read fields
        self.cfg = _FakeConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=getattr(self.hf_model.config, "n_positions",
                               getattr(self.hf_model.config, "max_position_embeddings", 1024)),
            n_layer=getattr(self.hf_model.config, "n_layer",
                            getattr(self.hf_model.config, "num_hidden_layers", 0)),
            n_head=getattr(self.hf_model.config, "n_head",
                           getattr(self.hf_model.config, "num_attention_heads", 0)),
            n_embd=getattr(self.hf_model.config, "n_embd",
                           getattr(self.hf_model.config, "hidden_size", 0)),
        )

    def parameters(self):
        return self.hf_model.parameters()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_iter(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """Yield one token id at a time — same API as GPT.generate_iter()."""
        from torch.nn import functional as F

        input_ids = idx.to(self.device)
        past = None

        for _ in range(max_new_tokens):
            out = self.hf_model(input_ids=input_ids, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :] / max(temperature, 1e-8)
            past = out.past_key_values

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            yield next_token.item()

            # Feed only the new token (KV cache handles the rest)
            input_ids = next_token


class _FakeConfig:
    """Mimics GPTConfig fields so server.py introspection works."""
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = 0.0
        self.bias = False
