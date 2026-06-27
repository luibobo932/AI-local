# 📋 Tài liệu bàn giao dự án AI-Local

> Tài liệu này dành cho người/AI tiếp nhận dự án (Codex). Tóm tắt trạng thái
> hiện tại, kiến trúc, những gì đã hoàn thành, hạn chế đã biết, và việc nên làm
> tiếp theo.

---

## 1. Tổng quan

AI-Local là bộ công cụ **LLM tự xây từ đầu** (phong cách nanoGPT) chạy hoàn toàn
trên máy, kèm REST API + CLI tương thích Ollama, backend HuggingFace, pipeline
fine-tune, và một model chat tiếng Việt train-from-scratch đã deploy.

- **Repo:** https://github.com/luibobo932/AI-local
- **Branch đang phát triển:** `claude/local-language-model-3g7ije`
- **Pull Request:** #1
- **Demo công khai (HF Space):** https://luibobo-ai-local-chat.hf.space
- **Ngôn ngữ:** Python 3.10+, PyTorch 2.x

---

## 2. Kiến trúc & bản đồ file

```
config.py              GPTConfig (kiến trúc) + TrainConfig (siêu tham số train)
model/
  gpt.py               GPT decoder-only: CausalSelfAttention, MLP, Block, GPT
                       - forward(), forward_full() (cho DPO)
                       - generate(), generate_iter() (streaming, yield 1 token)
  hf_backend.py        HFModel: bọc model HuggingFace cùng interface generate_iter
                       - is_hf_model(): nhận diện gpt2 / repo-id / models/<dir>
data/
  prepare.py           Dữ liệu Shakespeare (tiếng Anh, char-level)
  prepare_vi.py        Dữ liệu TIẾNG VIỆT char-level (corpus nhúng + CHAR_COVERAGE
                       + EOS "■"). --input để dùng corpus riêng
  prepare_sft.py       Token hóa SFT (prompt masking, thêm EOS "■" nếu vocab có)
  prepare_dpo.py       Dữ liệu preference cho DPO
  sft_vi.jsonl         67 cặp hỏi-đáp tiếng Việt (nhiều biến thể cách hỏi)
  sample_finetune.jsonl  Dữ liệu mẫu cho finetune_hf.py
  meta.pkl             [committed] vocab char-level tiếng Việt (210 token, có EOS)
train.py               Pre-training (cosine LR, grad accum, AMP, CLI override arch)
finetune_sft.py        SFT trên model tự train (masked cross-entropy)
finetune_hf.py         Fine-tune model PRE-TRAINED (full / LoRA) -> models/<out>/
align_dpo.py           DPO alignment (Rafailov 2023, ref model đóng băng)
generate.py            Sinh văn bản / chat CLI trực tiếp
inspect_model.py       Xem thông tin checkpoint
server.py              REST API (Ollama + OpenAI compatible) + web UI tại "/"
cli.py                 CLI: serve/run/list/show/ps/pull/finetune/rm
web/index.html         Giao diện chat web (vanilla JS, gọi /api/chat)
app.py                 Giao diện Gradio (cho Colab / link công khai)
chat_vi.py             Script chat tiếng Việt 1 lệnh (dùng chat_vi.pt)
checkpoints/chat_vi.pt [committed] model chat tiếng Việt đã train (8.5MB)
hf_space/              Bộ file deploy lên HuggingFace Space
  app.py               Bootstrap: pip install + git clone + exec _chat_app.py
  _chat_app.py         Thân thật của Space (non-streaming, greedy decoding)
  requirements.txt, README.md
AI_Local_Chat.ipynb    Notebook Colab 1 ô để chạy nhanh
README.md              Hướng dẫn tổng (tiếng Việt)
```

---

## 3. Ba con đường tạo model (đều hoạt động)

1. **Train từ đầu:** `python cli.py pull all` (pretrain → SFT → DPO)
2. **Dùng model có sẵn:** `python cli.py pull gpt2 && python cli.py run gpt2`
3. **Fine-tune thành của riêng:** `python cli.py finetune --base gpt2 --data <jsonl> --out <name>`

### Pipeline tiếng Việt (đã chứng minh chạy được trên CPU)
```bash
python data/prepare_vi.py                                  # vocab 210, có EOS
python train.py --n_layer 4 --n_head 4 --n_embd 192 \
    --block_size 256 --batch_size 12 --max_iters 2000 --learning_rate 3e-3
python data/prepare_sft.py --input data/sft_vi.jsonl --block_size 256
python finetune_sft.py --block_size 256 --max_iters 800
python chat_vi.py                                          # chat
```

---

## 4. Trạng thái hiện tại

| Thành phần | Trạng thái |
|---|---|
| Kiến trúc GPT (model/gpt.py) | ✅ Hoàn chỉnh, có test thủ công |
| Pre-training (train.py) | ✅ Chạy tốt CPU/GPU |
| SFT (finetune_sft.py) | ✅ Hoạt động, có EOS |
| DPO (align_dpo.py) | ✅ Code xong (chạy demo 20 iter) |
| HF backend + fine-tune | ✅ Code xong, CHƯA test với weight thật (môi trường chặn HF) |
| REST API server | ✅ Test: tags/chat/generate/embed OK |
| Web UI (web/index.html) | ✅ Test cục bộ OK |
| Gradio app + Colab | ✅ Test boot + sinh chữ OK |
| Model chat tiếng Việt | ✅ Train xong, trả lời đúng nhiều biến thể (greedy) |
| HF Space deploy | ✅ Đã deploy & chạy; vừa vá lỗi rớt chữ (chờ user factory reboot) |

---

## 5. ⚠️ Hạn chế & vấn đề đã biết

1. **Model tiếng Việt rất nhỏ (1.8M tham số), corpus viết tay nhỏ.** Nó *memorize*
   các cặp hỏi-đáp + corpus, generalize kém. Hỏi ngoài phạm vi → trả lời sai/lung
   tung. Đây là giới hạn cơ bản về scale, không phải bug.

2. **Streaming gây rớt chữ trên một số phiên bản Gradio.** Quan sát: output local
   hoàn hảo nhưng trên HF Space bị rớt ký tự giữa từ ("Xin"→"Xi"). Đã xử lý bằng
   cách dùng **non-streaming + greedy (top_k=1)** trong `hf_space/_chat_app.py`.
   Nghi do khác phiên bản torch/gradio. *Chưa truy được root cause chính xác* —
   nếu vẫn lỗi, thử ghim phiên bản torch/gradio trong hf_space/requirements.txt.

3. **Môi trường phát triển (cloud sandbox) chặn mạng tới HuggingFace/PyPI một phần.**
   Vì vậy backend HF (`pull gpt2`, fine-tune model thật) **chưa chạy end-to-end**
   ở đây — code đã viết và test phần logic, cần chạy thật trên máy có mạng.

4. **CPU-only ở môi trường này** → train chậm (~0.5-1s/iter cho model nhỏ). Model
   lớn cần GPU.

5. **chat_vi.pt được force-add** dù `checkpoints/` nằm trong .gitignore. Các
   checkpoint khác (ckpt.pt, sft_ckpt.pt, dpo_ckpt.pt) KHÔNG được commit (tái tạo
   bằng pipeline).

---

## 6. 🎯 Việc nên làm tiếp (gợi ý cho Codex)

**Ưu tiên cao — chất lượng model tiếng Việt:**
- [ ] Thay corpus viết tay bằng **corpus tiếng Việt lớn** (Wikipedia VN, sách,
      tin tức) để model học ngữ pháp/từ vựng thật, generalize tốt.
- [ ] Tăng kích thước model (n_embd 384-512, n_layer 6-8) khi có GPU.
- [ ] Cân nhắc **subword tokenizer** (BPE/SentencePiece) thay char-level cho
      tiếng Việt — hiệu quả hơn nhiều.
- [ ] Hoặc thực dụng hơn: **fine-tune PhoGPT/Vistral/SeaLLM** qua `finetune_hf.py`
      (`--lora`) để có chất lượng cao ngay.

**Kỹ thuật:**
- [ ] Truy root cause lỗi rớt chữ streaming trên Gradio; khôi phục streaming nếu
      sửa được (UX tốt hơn).
- [ ] Test end-to-end backend HF (`pull gpt2`, fine-tune) trên máy có mạng.
- [ ] Thêm test tự động (pytest) cho model forward, generate, server endpoints.
- [ ] DPO: train đủ iteration với data thật, đánh giá reward margin.
- [ ] Thêm chat template native cho model HF (hiện hardcode "Human:/Assistant:").

**Sản phẩm:**
- [ ] Lưu lịch sử hội thoại nhiều lượt đúng cách (hiện server có context cơ bản).
- [ ] Auth/rate-limit cho server nếu deploy public.
- [ ] CI/CD: GitHub Action build & test.

---

## 7. Cách chạy nhanh để kiểm tra

```bash
git clone -b claude/local-language-model-3g7ije https://github.com/luibobo932/AI-local
cd AI-local
pip install -r requirements.txt

python chat_vi.py                 # chat tiếng Việt ngay (model đã kèm sẵn)
python cli.py serve               # API + web UI tại http://localhost:11434
python app.py                     # Gradio UI (share=True cho link công khai)
```

---

## 8. Quy ước & lưu ý kỹ thuật

- `torch.load(..., weights_only=False)` ở mọi nơi (checkpoint chứa class GPTConfig).
- EOS token là ký tự `■` (U+25A0); generate dừng khi gặp, không in ra.
- Char-level vocab build từ corpus + `CHAR_COVERAGE` (đảm bảo phủ đủ ký tự cho SFT).
- Template chat: `"Human: {q}\n\nAssistant: {r}"`.
- Model nhỏ → nên decode **greedy (top_k=1)** để ổn định; sampling dễ lạc.
