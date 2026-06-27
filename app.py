"""
Giao diện chat Gradio — chạy được trên Google Colab, tự sinh LINK CÔNG KHAI.

Dùng trên điện thoại (qua Google Colab):
    1. Mở https://colab.research.google.com trên điện thoại
    2. Tạo notebook mới, dán đoạn code này vào một ô (cell):

        !git clone -b claude/local-language-model-3g7ije https://github.com/luibobo932/AI-local
        %cd AI-local
        !pip install -q torch numpy gradio
        !python app.py

    3. Chạy ô đó. Sau ~1 phút sẽ hiện link dạng:
           Running on public URL: https://xxxxx.gradio.live
    4. Bấm vào link đó — chat ngay trên điện thoại!

Chạy trên máy tính (local):
    pip install gradio
    python app.py            # mở http://localhost:7860
"""

import pickle
import torch
import gradio as gr

from model.gpt import GPT

CKPT = "checkpoints/chat_vi.pt"
META = "data/meta.pkl"

# ─── Tải model một lần ────────────────────────────────────────────────────────
print("Đang tải model...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = GPT(ckpt["model_cfg"])
model.load_state_dict(ckpt["model"])
model.eval()

with open(META, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
params = sum(p.numel() for p in model.parameters())
print(f"Đã tải model tiếng Việt ({params/1e6:.1f}M tham số)")


def respond(message, history, temperature, top_k):
    """Sinh câu trả lời, stream từng chữ. history là list {'role','content'}."""
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
            if "■" in c:          # EOS — model báo kết thúc
                break
            if "\n\nHuman" in (answer + c):
                break
            answer += c
            yield answer


SAMPLES = ["Xin chào", "Bạn tên là gì?", "Thủ đô của Việt Nam là gì?",
           "Món phở là gì?", "Làm sao để học tốt?", "Kể một câu tục ngữ"]

demo = gr.ChatInterface(
    fn=respond,
    title="AI-Local 🧠 — Chat tiếng Việt",
    description="Mô hình ngôn ngữ train từ đầu, chạy local. Model nhỏ (1.8M tham số) — "
                "trả lời tốt các chủ đề đã học.",
    examples=[[s, 0.4, 10] for s in SAMPLES],
    additional_inputs=[
        gr.Slider(0.1, 1.2, value=0.4, step=0.1, label="Temperature"),
        gr.Slider(1, 50, value=10, step=1, label="Top-k"),
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    # share=True → tạo link công khai .gradio.live (mở được trên điện thoại)
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)
