"""
Phần thân thật của HuggingFace Space — được app.py (bootstrap) tải về từ
GitHub rồi exec(). Không chạy trực tiếp file này.
"""

import os
import pickle
import torch
import gradio as gr

# Tìm thư mục repo (bootstrap clone vào ./repo; fallback về thư mục hiện tại)
BASE = "repo" if os.path.exists(os.path.join("repo", "checkpoints", "chat_vi.pt")) else "."
import sys
sys.path.insert(0, BASE)

from model.gpt import GPT  # noqa: E402

ckpt = torch.load(os.path.join(BASE, "checkpoints", "chat_vi.pt"),
                  map_location="cpu", weights_only=False)
model = GPT(ckpt["model_cfg"])
model.load_state_dict(ckpt["model"])
model.eval()

with open(os.path.join(BASE, "data", "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
print(f"Model tiếng Việt sẵn sàng ({sum(p.numel() for p in model.parameters())/1e6:.1f}M tham số)")


def respond(message, history, temperature, top_k):
    # KHÔNG dùng streaming (yield) — trả về cả câu một lần để tránh lỗi rớt chữ
    # trên một số phiên bản giao diện.
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
    return answer.strip()


SAMPLES = ["Xin chào", "Bạn tên là gì?", "Thủ đô của Việt Nam là gì?",
           "Món phở là gì?", "Làm sao để học tốt?", "Kể một câu tục ngữ"]

demo = gr.ChatInterface(
    fn=respond,
    title="AI-Local 🧠 — Chat tiếng Việt",
    description="Mô hình ngôn ngữ train từ đầu, chạy local. Trả lời tốt các chủ đề đã học.",
    # top_k=1 = greedy = luôn chọn chữ chắc chắn nhất → ổn định, không rớt chữ
    examples=[[s, 0.3, 1] for s in SAMPLES],
    additional_inputs=[
        gr.Slider(0.1, 1.2, value=0.3, step=0.1, label="Temperature (độ sáng tạo)"),
        gr.Slider(1, 50, value=1, step=1, label="Top-k (1 = chắc chắn nhất)"),
    ],
    cache_examples=False,
)

demo.queue().launch()
