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
| **🤖 Agent framework** | Tool use, ReAct loop, function calling — giống Claude Code / Cowork |
| **🔧 16 built-in tools** | file · shell · web · git · repo_map · memory · skills |
| **🔌 MCP client** | Kết nối MCP servers ngoài (như Claude Code) |
| **🧠 Memory & Skills** | Project memory (AILOCAL.md), auto-memory, workflow đóng gói |
| **💾 Sessions** | Lưu & resume các phiên làm việc |

---

## ⚡ MVP nhanh nhất — Minion + Ollama (KHÔNG cần torch/fastapi)

Chạy được ngay trên máy cá nhân với **Ollama / LM Studio / OpenAI-compatible**, chỉ cần Python 3.10+:

```bash
# 1. Cài Ollama (https://ollama.com) và pull model code:
ollama pull qwen2.5-coder           # hoặc deepseek-coder-v2

# 2. Cấu hình (tùy chọn — mặc định đã trỏ Ollama + qwen2.5-coder):
cp .env.example .env

# 3. Chạy — không cần cài thêm thư viện nào:
python minion.py providers                       # xem cấu hình
python minion.py chat                            # chat tương tác
python minion.py agent "tóm tắt repo này" -v     # agent coding với tool use
python minion.py agent "đề xuất cải thiện X" --plan   # chế độ chỉ-đọc (lập kế hoạch)
python minion.py serve --port 8000               # HTTP API: POST /chat, /agent/run

# Hoặc dùng script:
./run_local.sh serve        # Linux/Mac
run_local.bat serve         # Windows
```

Dùng **LM Studio**: đặt `LLM_PROVIDER=lmstudio` trong `.env`.
Dùng **OpenAI**: `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, đặt `OPENAI_API_KEY` trong môi trường.

API:
```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "liệt kê file Python và tóm tắt", "max_steps": 6}'
```

> `minion.py` dùng **stdlib thuần** (urllib) — không cần torch/fastapi/httpx.
> Phần `server.py` (REST API đầy đủ + model built-in) vẫn dùng được nếu cài `requirements.txt`.

---

## 🤖 Agent đa năng (như Claude Code / Cowork)

Ngoài chat, AI-local còn là **agent** biết tự dùng công cụ để hoàn thành nhiệm vụ:

```bash
python cli.py serve                                    # khởi động server
python cli.py agent "đọc README.md và tóm tắt" -v      # chạy agent, hiện từng bước
python cli.py agent "tìm và sửa bug X" --skill fix-bug # áp dụng skill
python cli.py agent "đề xuất cải thiện server.py" --plan  # chế độ chỉ-đọc (lập kế hoạch)
python cli.py tools                                    # 16 tools có sẵn
python cli.py skills list                              # workflow đóng gói
python cli.py repomap                                  # bản đồ codebase (kiểu Aider)
python cli.py memory show                              # bộ nhớ dài hạn
python cli.py sessions list                            # phiên đã lưu
python cli.py mcp add fs http://localhost:3001         # kết nối MCP server
```

Hoặc bật toggle **🤖 Agent** ngay trên giao diện web (kèm **📋 Plan** cho chế độ chỉ-đọc). Agent tự động:
- Đọc **project memory** (`AILOCAL.md`) + **auto-memory** vào ngữ cảnh
- Sinh **repo map** để hiểu cấu trúc codebase
- Chọn & gọi tools theo vòng lặp (đọc/ghi file, chạy shell, search web, git, MCP...)
- Tự **ghi nhớ** điều học được cho phiên sau (`remember` tool)
- **Permission modes** kiểu Claude Code: `auto` (cho phép tất cả) · `plan` (chỉ đọc, lập kế hoạch) · `approve` (ghi cần allowlist)

**API:** `POST /v1/agent` · `POST /v1/agent/stream` (SSE, hiện từng bước live) · `GET /v1/tools` · `GET /v1/skills` · `GET /v1/memory` · `GET /v1/sessions` · `GET /v1/repomap` · `GET/POST /v1/mcp/servers`

> 💡 Agent mạnh nhất với model HF lớn (Llama, Mistral...) ở chế độ `function_calling`.
> Model tiếng Việt nhỏ `chat_vi` chạy chế độ `react` — phù hợp demo, kết quả hạn chế.

---

## 🚀 Bắt đầu nhanh

```bash
git clone https://github.com/luibobo932/AI-local
cd AI-local
pip install -r requirements.txt
```

### 📱 Chat trên điện thoại (Google Colab — có link công khai)
Không cần máy tính. Mở [Google Colab](https://colab.research.google.com), tạo notebook mới, dán và chạy:
```python
!git clone -b claude/local-language-model-3g7ije https://github.com/luibobo932/AI-local
%cd AI-local
!pip install -q torch numpy gradio
!python app.py
```
Sau ~1 phút sẽ hiện link `https://xxxxx.gradio.live` → bấm vào là chat ngay trên điện thoại.
(Hoặc mở sẵn file `AI_Local_Chat.ipynb` trong repo bằng Colab.)

### 🖥️ Giao diện web (chạy local)
```bash
python cli.py serve          # rồi mở http://localhost:11434 trên trình duyệt
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

Sau khi train xong, model hỏi-đáp tiếng Việt được ngay (dừng sạch nhờ EOS):
```
>>> Xin chào
AI-Local: Xin chào bạn! Rất vui được gặp bạn. Tôi có thể giúp gì cho bạn hôm nay?
>>> Thủ đô của Việt Nam là gì?
AI-Local: Thủ đô của Việt Nam là Hà Nội, một thành phố nghìn năm văn hiến với bề dày lịch sử.
>>> Món phở là gì?
AI-Local: Phở là món ăn nổi tiếng của Hà Nội, với nước dùng ngọt thanh, bánh phở mềm và thịt bò thơm ngon.
```
> Model nhỏ (1.8M params) học từ corpus mẫu → trả lời tốt trong phạm vi đã học.
> Corpus càng lớn, model càng khái quát hóa và trả lời được câu ngoài tập huấn luyện.

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
