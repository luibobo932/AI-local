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

---

## Minion Desktop Cowork

Chạy app local:

```bash
python server.py --port 11435
```

Mở UI: `http://localhost:11435`

Minion hiện có các phần chính:

- Chat tiếng Việt với persona rõ: Minion là trợ lý local do Duy phát triển.
- Computer-use local: chụp màn hình, đọc UI tree, click/gõ/cuộn chuột, focus cửa sổ, resize/minimize/maximize, mở app/link.
- Agent nhiều bước: `POST /api/agent/run`, xem trạng thái tại `/api/agent/runs/{id}`, có stop/pause/resume.
- Workspace/code agent API: status, diagnostics, review findings, current diff, list files, search, read, diff/patch exact replace, run command trong workspace.
- Quyền an toàn mặc định: `ask_when_risky`. Shell command, sửa file, ghi clipboard, đóng cửa sổ phải xác nhận một lần; lệnh nguy hiểm như `git reset --hard`, `rm -rf`, `Remove-Item -Recurse` bị chặn.
- Cấu hình nằm ở `minion.config.json`.

Endpoint mới:

```text
POST /api/computer-use/command
POST /api/agent/run
GET  /api/agent/runs/{id}
POST /api/agent/runs/{id}/stop
POST /api/agent/runs/{id}/pause
POST /api/agent/runs/{id}/resume
GET  /api/workspace/status
GET  /api/workspace/diagnostics
GET  /api/workspace/review
GET  /api/workspace/diff
POST /api/workspace/files
POST /api/workspace/search
POST /api/workspace/read
POST /api/workspace/patch
POST /api/workspace/run
```

---

## Minion Brain v1

Minion đã có lớp bộ não mới nhưng vẫn tương thích các API cũ:

- `minion_core.py`: persona và định tuyến Qwen3/Qwen2.5-Coder theo nhiệm vụ.
- `minion_memory.py`: trí nhớ SQLite local, tìm theo từ khóa và embedding Ollama.
- `minion_agent.py`: vòng lặp tool-calling nhiều bước; thao tác rủi ro vẫn dùng approval hiện có.
- `evals/`: bộ bài thi router, safety và chống nhận nhầm câu hỏi nhà thành lệnh máy.
- `train_minion_qlora.py`: pipeline SFT + QLoRA 4-bit cho dữ liệu hội thoại/tool-calling.

### Trí nhớ local

```text
GET    /api/memory
POST   /api/memory/remember
POST   /api/memory/search
DELETE /api/memory/{id}
POST   /api/knowledge/ingest
GET    /api/minion/config
GET    /api/minion/route?text=...
```

Trên giao diện có nút `Ghi nhớ`. Database mặc định là `data/minion_memory.db` và không được commit.

### Kiểm tra bộ não

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m unittest discover -s tests -p "test_*.py" -v
python evals/run_minion_eval.py --base-url http://127.0.0.1:11435
```

### Môi trường GPU riêng

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_minion_gpu.ps1
.\.venv-cuda\Scripts\python.exe verify_minion_gpu.py
```

### Chuẩn bị dữ liệu và QLoRA

Dữ liệu phải là JSONL hội thoại có trường `messages`; bộ seed hiện tại được tạo từ dữ liệu cũ và các mẫu đã biên soạn riêng cho Minion.

```powershell
# Tạo và kiểm định bộ seed (142 ví dụ, gồm hội thoại, an toàn và tool-calling)
python data\build_minion_sft.py
python data\validate_minion_sft.py

# Chỉ kiểm tra cấu hình, chưa train
.\.venv-cuda\Scripts\python.exe train_minion_qlora.py `
  --data data\minion_sft_seed.jsonl --dry-run

# Train adapter seed và lưu báo cáo đo lường
.\.venv-cuda\Scripts\python.exe train_minion_qlora.py `
  --base Qwen/Qwen3-0.6B `
  --data data\minion_sft_seed.jsonl `
  --out models\minion-sft-seed-v1 `
  --max-length 512 `
  --report reports\minion_sft_seed_v1.json
```

Thư mục `models/` không đẩy lên GitHub vì adapter có dung lượng lớn. GitHub lưu mã nguồn, dữ liệu seed và báo cáo train để có thể tái tạo model.

Không train bản chính bằng file sample. File sample chỉ dùng kiểm tra schema.
