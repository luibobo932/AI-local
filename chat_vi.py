"""
Chat tiếng Việt — chạy ngay, không cần train.

Model đã được train sẵn (checkpoints/chat_vi.pt) và đưa vào repo, nên bạn
chỉ cần:

    pip install torch numpy
    python chat_vi.py

Rồi gõ câu hỏi. Gõ /bye để thoát.

Lưu ý: đây là model nhỏ (1.8M tham số) học từ một corpus mẫu, nên nó trả lời
tốt các chủ đề đã học (chào hỏi, Việt Nam, món ăn, học tập, tục ngữ...).
Hỏi câu hoàn toàn mới có thể chưa trả lời được — cần corpus lớn hơn.
"""

import os
import pickle
import torch

from model.gpt import GPT

CKPT = "checkpoints/chat_vi.pt"
META = "data/meta.pkl"

GREEN, YELLOW, CYAN, RESET = "\033[92m", "\033[93m", "\033[96m", "\033[0m"


def main():
    if not os.path.exists(CKPT) or not os.path.exists(META):
        print("Thiếu file model. Cần checkpoints/chat_vi.pt và data/meta.pkl")
        return

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GPT(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(META, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{GREEN}AI-Local 🇻🇳{RESET} — model tiếng Việt ({params/1e6:.1f}M tham số)")
    print("Gõ câu hỏi, /bye để thoát.\n")

    while True:
        try:
            q = input(f"{CYAN}>>> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break
        if not q:
            continue
        if q in ("/bye", "/exit", "/quit"):
            print("Tạm biệt!")
            break

        prompt = f"Human: {q}\n\nAssistant: "
        ids = [stoi.get(c, stoi.get(" ", 0)) for c in prompt]
        idx = torch.tensor([ids], dtype=torch.long)

        print(f"{YELLOW}", end="", flush=True)
        with torch.no_grad():
            for tid in model.generate_iter(idx, 200, temperature=0.4, top_k=10):
                c = itos.get(tid, "")
                if "■" in c:          # EOS — model báo kết thúc
                    break
                print(c, end="", flush=True)
        print(f"{RESET}\n")


if __name__ == "__main__":
    main()
