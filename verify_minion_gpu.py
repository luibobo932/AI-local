"""Kiểm tra môi trường CUDA dành riêng cho train Minion."""

from __future__ import annotations

import json
import sys

import torch


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    if torch.cuda.is_available():
        # Phép tính nhỏ xác nhận tensor thực sự chạy trên GPU.
        left = torch.randn((512, 512), device="cuda")
        right = torch.randn((512, 512), device="cuda")
        result = left @ right
        info["gpu_test"] = bool(torch.isfinite(result).all().item())
        info["allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
    else:
        info["gpu_test"] = False
        info["error"] = "PyTorch chưa nhận CUDA. Không nên bắt đầu fine-tune."
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info["gpu_test"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
