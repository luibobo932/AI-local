from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 50304      # GPT-2 vocab size (padded to nearest multiple of 64)
    block_size: int = 256        # context / sequence length
    n_layer: int = 6             # number of transformer blocks
    n_head: int = 6              # number of attention heads
    n_embd: int = 384            # embedding dimension
    dropout: float = 0.1
    bias: bool = False           # use bias in linear layers and layer norms


@dataclass
class TrainConfig:
    # Data
    dataset: str = "shakespeare"           # "shakespeare" | "custom"
    data_dir: str = "data"
    out_dir: str = "checkpoints"

    # Training
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    log_interval: int = 100

    # Batch
    batch_size: int = 64
    gradient_accumulation_steps: int = 1  # simulate larger batch

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 6e-5

    # System
    device: str = "auto"         # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "bfloat16"      # "float32" | "bfloat16" | "float16"
    compile: bool = False        # torch.compile (requires PyTorch 2.0+)

    # Checkpointing
    resume_from: str = ""        # path to checkpoint to resume from
    always_save_checkpoint: bool = False
