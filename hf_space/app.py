"""
HuggingFace Space — Chat tiếng Việt AI-Local.

File này tự tải model + code từ GitHub khi Space khởi động, nên Space chỉ cần
3 file nhỏ (app.py, requirements.txt, README.md). Không phải upload model nặng.
"""

import os
import subprocess
import sys
import pickle

import torch
import gradio as gr

# ─── Tự tải code + model từ GitHub khi chạy trên Space ────────────────────────
REPO = "https://github.com/luibobo932/AI-local"
BRANCH = "claude/local-language-model-3g7ije"
DIR = "AI-local"

if not os.path.exists(DIR):
    print("Đang tải model + code từ GitHub...")
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, DIR], check=True)

sys.path.insert(0, DIR)
from model.gpt import GPT  # noqa: E402

CKPT = os.path.join(DIR, "checkpoints", "chat_vi.pt")
META = os.path.join(DIR, "data", "meta.pkl")

# ─── Tải model một lần ────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = GPT(ckpt["model_cfg"])
model.load_state_dict(ckpt["model"])
model.eval()

with open(META, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
print(f"Model tiếng Việt sẵn sàng ({sum(p.numel() for p in model.parameters())/1e6:.1f}M tham số)")


def respond(message, history, temperature, top_k):
    parts = []
    for turn in history:
        role = turn.get("role")
        if role == "user":
            parts.append(f"Human: {turn['content']}")
        elif role == "assistant":
            parts.append(f"Assistant: {turn['content']}")
    parts.append(f"Human: {message}")
    parts.append("Assistant: ")
    prompt = "\n\n".join(parts)

    ids = [stoi.get(c, stoi.get(" ", 0)) for c in prompt]
    idx = torch.tensor([ids], dtype=torch.long)

    answer = ""
    with torch.no_grad():
        for tid in model.generate_iter(idx, 200, temperature=temperature, top_k=int(top_k)):
            c = itos.get(tid, "")
            if "■" in c or "\n\nHuman" in (answer + c):
                break
            answer += c
            yield answer


SAMPLES = ["Xin chào", "Bạn tên là gì?", "Thủ đô của Việt Nam là gì?",
           "Món phở là gì?", "Làm sao để học tốt?", "Kể một câu tục ngữ"]

demo = gr.ChatInterface(
    fn=respond,
    title="AI-Local 🧠 — Chat tiếng Việt",
    description="Mô hình ngôn ngữ train từ đầu (1.8M tham số). Trả lời tốt các chủ đề đã học.",
    examples=[[s, 0.4, 10] for s in SAMPLES],
    additional_inputs=[
        gr.Slider(0.1, 1.2, value=0.4, step=0.1, label="Temperature"),
        gr.Slider(1, 50, value=10, step=1, label="Top-k"),
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.queue().launch()
