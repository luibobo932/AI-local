# AI-Local 🧠

Một bộ công cụ **mô hình ngôn ngữ lớn (LLM) chạy hoàn toàn trên máy của bạn** — xây từ đầu theo phong cách [nanoGPT](https://github.com/karpathy/nanoGPT) của Andrej Karpathy, kèm REST API và CLI tương thích [Ollama](https://ollama.com).

> Riêng tư, miễn phí, chạy offline. Không gửi dữ liệu lên cloud.

---

## ✨ Có gì trong này

| Thành phần | Mô tả |
|---|---|
| **Kiến trúc GPT** | Transformer decoder-only (GPT-2 style), Flash Attention, weight tying |
| **Train 3 giai đoạn** | Pre-training → SFT (instruction) → DPO (alignment) |
| **Backend HuggingFace** | Tải & chạy GPT-2, TinyLlama, model tiếng Việt... |
| **Fine-tune** | Lấy model pre-trained về, dạy thành của riêng bạn (full hoặc LoRA) |
| **REST API server** | Tương thích Ollama + OpenAI (streaming) |
| **CLI** | `serve · run · list · show · ps · pull · finetune · rm` |
| **Tiếng Việt** | Data prep char-level hỗ trợ đầy đủ dấu tiếng Việt |

---

## 🚀 Bắt đầu nhanh

```bash
git clone https://github.com/luibobo932/AI-local
cd AI-local
pip install -r requirements.txt
```

### Cách 1 — Dùng model có sẵn (nhanh nhất, chat được ngay)
```bash
python cli.py pull gpt2      # tải GPT-2 của OpenAI (~500MB)
python cli.py run gpt2       # chat!
```

### Cách 2 — Fine-tune model thành của riêng bạn ⭐
```bash
python cli.py pull gpt2
python cli.py finetune --base gpt2 --data data/sample_finetune.jsonl --out my-assistant
python cli.py run my-assistant
```

### Cách 3 — Train từ đầu (hiểu cách LLM hoạt động)
```bash
python cli.py pull all       # pretrain → SFT → DPO (cần GPU để nhanh)
python cli.py run dpo_ckpt
```

---

## 🇻🇳 Train tiếng Việt

```bash
# Dùng corpus có sẵn hoặc corpus của bạn
python data/prepare_vi.py                          # corpus tiếng Việt nhúng sẵn
python data/prepare_vi.py --input corpus.txt       # corpus riêng của bạn

# Train (tokenizer char-level tự bao gồm dấu à/ạ/ê/ộ/ữ...)
python train.py --n_layer 4 --n_head 4 --n_embd 192 \
    --block_size 256 --batch_size 12 --max_iters 2000 --learning_rate 3e-3

# SFT hỏi-đáp tiếng Việt
python data/prepare_sft.py --input data/sft_vi.jsonl --block_size 256
python finetune_sft.py --block_size 256

python cli.py run sft_ckpt
```

Muốn chất lượng cao hơn → fine-tune model Việt có sẵn:
```bash
python cli.py finetune --base vinai/PhoGPT-4B-Chat --data data/sft_vi.jsonl --out tro-ly-viet --lora
```

---

## 🌐 REST API Server (tương thích Ollama)

```bash
python cli.py serve          # hoặc: python server.py --port 11434
```

```bash
# Ollama-style
curl http://localhost:11434/api/chat -d '{
  "model": "gpt2",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# OpenAI-compatible — dùng được với openai SDK
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "gpt2",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

Endpoints: `/api/version` `/api/health` `/api/tags` `/api/ps` `/api/show` `/api/copy`
`/api/generate` `/api/chat` `/api/embeddings` `/api/embed` `/api/delete`
+ OpenAI: `/v1/models` `/v1/chat/completions` `/v1/completions` `/v1/embeddings`

---

## 📂 Cấu trúc dự án

```
config.py            GPTConfig + TrainConfig
model/
  gpt.py             Kiến trúc GPT (attention, MLP, block, generate)
  hf_backend.py      Wrapper cho model HuggingFace
data/
  prepare.py         Chuẩn bị data Shakespeare (tiếng Anh)
  prepare_vi.py      Chuẩn bị data tiếng Việt (char-level có dấu)
  prepare_sft.py     Chuẩn bị data instruction (SFT)
  prepare_dpo.py     Chuẩn bị data preference (DPO)
  sft_vi.jsonl       Bộ hỏi-đáp tiếng Việt mẫu
train.py             Pre-training
finetune_sft.py      SFT trên model tự train
finetune_hf.py       Fine-tune model pre-trained (full / LoRA)
align_dpo.py         DPO alignment
generate.py          Sinh văn bản / chat trực tiếp
server.py            REST API server
cli.py               Command line interface
```

---

## 🎓 Ba giai đoạn train là gì?

1. **Pre-training** — model đọc lượng lớn văn bản, học dự đoán từ tiếp theo. Học ngữ pháp, từ vựng, kiến thức.
2. **SFT (Supervised Fine-Tuning)** — dạy model trả lời theo định dạng hỏi-đáp bằng các cặp instruction-response.
3. **DPO (Direct Preference Optimization)** — căn chỉnh model theo sở thích con người bằng cặp câu trả lời tốt/xấu, không cần reward model.

---

## ⚙️ Yêu cầu

- Python 3.10+
- PyTorch 2.1+
- (Tùy chọn) GPU NVIDIA hoặc Apple Silicon để train nhanh
- (Tùy chọn) `peft` để fine-tune LoRA

## 📜 License

MIT — dùng tự do cho học tập và dự án cá nhân.
