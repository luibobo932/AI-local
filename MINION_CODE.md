# Minion Code — coding agent local (kiểu Claude Code)

Tài liệu phần "trợ lý lập trình" của minion. Phần bất động sản xem `server.py`.

## 5 năng lực (theo yêu cầu) — hiện trạng

| Năng lực | Thực hiện ở đâu |
|---|---|
| 1. Tool Use (đọc/ghi/liệt kê file + JSON function-calling) | `tools/` + `agent.py` (đã có sẵn) |
| 2. **RAG ngữ nghĩa (embeddings + vector store)** | `rag.py` + tool `rag_search` ⭐ MỚI |
| 3. Agentic loop tự sửa lỗi (chạy code → đọc Traceback → sửa → lặp) | `agent.py` + tool `run_command`, ép trong system prompt ⭐ |
| 4. System 2 — `<thinking>` Chain-of-Thought | `CODING_AGENT_SYSTEM` trong `agent.py` ⭐ |
| 5. Song ngữ: suy luận EN trong `<thinking>`, trả lời VI | `CODING_AGENT_SYSTEM` ⭐ |

## RAG — bộ nhớ ngoài (`rag.py`)

Tự chia nhỏ mã nguồn → tạo embedding qua chính LLM provider của bạn (Ollama/LM Studio/OpenAI)
→ lưu vector vào `.ai-local/rag_index.json` → tra cứu đoạn code liên quan theo NGỮ NGHĨA.
Nhẹ, chỉ stdlib (+ numpy nếu có). Không cần langchain/llama-index.

### Cài đặt
```bash
# 1. Cài model embedding (nếu dùng Ollama)
ollama pull nomic-embed-text
# 2. Khai báo trong .env
echo "LLM_EMBED_MODEL=nomic-embed-text" >> .env
```

### Dùng
```bash
python rag.py index            # lập chỉ mục toàn dự án (tái dùng cache theo hash)
python rag.py search "xử lý đăng nhập tạo token" -k 5
python rag.py status
```

Trong agent, minion tự gọi tool `rag_search` khi cần hiểu code theo ý nghĩa
(khác `search_files` chỉ khớp text).

## Chạy coding agent
```bash
# Cấu hình provider trong .env (xem .env.example): LLM_PROVIDER, LLM_MODEL...
python minion.py agent "Thêm hàm tính giai thừa vào utils.py rồi viết test" -v
python minion.py agent "..." --plan      # chế độ chỉ-đọc (không ghi file)
python minion.py chat                      # chat thường
```

## Cơ chế `<thinking>` + tự sửa lỗi

System prompt ép minion:
- Mở đầu bằng `<thinking>...</thinking>` để lập kế hoạch + tự kiểm tra giả định
  (được suy luận bằng tiếng Anh để tận dụng kiến thức pre-training).
- Sau `</thinking>` trả lời người dùng bằng tiếng Việt kỹ thuật.
- Sau khi sửa code, CHẠY bằng `run_command`; nếu Traceback thì đọc lỗi, sửa, chạy lại
  tới khi đúng (trong giới hạn `max_steps`). Tính toán thì chạy code thay vì tính nhẩm.

## Test
```bash
python tests/test_smoke.py     # agent, tools, permission (10 test)
python tests/test_rag.py       # RAG index/retrieve/cache (4 test, embedder giả lập)
```

## Lưu ý
- RAG cần model embedding của provider. Nếu chưa cấu hình, `rag_search` báo nhẹ nhàng
  (không crash) và agent vẫn dùng được `search_files`/`repo_map`.
- `.ai-local/` (index, sessions, memory) đã được gitignore — là trạng thái cục bộ per-máy.
