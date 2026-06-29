# TODO_CLAUDE.md — Kế hoạch hoàn thiện MVP

> Mục tiêu: AI coding agent chạy trên máy cá nhân, dùng được với **Ollama / LM Studio / OpenAI-compatible API**.
> Ưu tiên: đơn giản, chạy được trước, không over-engineering, tôn trọng code hiện có.

## 1. Phân tích hiện trạng

**Đã có (tốt):**
- `agent.py` — agent loop (ReAct + function calling), permission modes, memory/skills
- `tools/` — 21 tools (file, shell, web, git, glob, multi_edit, todo, repo_map...)
- `server.py` — REST API Ollama + OpenAI compatible (FastAPI)
- `repo_map.py`, `memory.py`, `sessions.py`, `skills.py`, `mcp_client.py`

**Thiếu / cần sửa (chặn MVP):**
- ❌ Toàn bộ phụ thuộc `torch` + `fastapi` + `httpx` → khó "chạy trên máy cá nhân"
- ❌ Agent chỉ gọi model 1.8M built-in, **không kết nối Ollama/LM Studio/OpenAI** (model code mạnh)
- ❌ Không có `.env.example`, không có logging tập trung
- ❌ Không có test / smoke test
- ❌ Không có script chạy local gọn (chỉ có .bat cho Windows desktop)

## 2. Việc cần làm (theo thứ tự)

- [x] **llm.py** — LLM client provider-agnostic, **chỉ dùng stdlib (urllib)**
      - Hỗ trợ: ollama, lmstudio, openai, ai-local
      - chat() + chat_stream(), đọc cấu hình từ `.env`/env
      - Không hard-code key (đọc từ env)
- [x] **Refactor agent.py** — dùng llm.py, bỏ phụ thuộc cứng httpx
      - Thêm tham số provider/base_url/api_key; giữ tương thích server_url cũ
      - Giữ giới hạn vòng lặp (max_steps) — tránh chạy vô hạn
- [x] **minion.py** — entrypoint MVP **không cần torch/fastapi**
      - CLI: `chat`, `agent`, `serve`, `providers`, `models`
      - HTTP server stdlib (`http.server`): `POST /chat`, `POST /agent/run`
- [x] **Logging cơ bản** — `logging` ghi ra `.ai-local/minion.log` + console
- [x] **.env.example** — cấu hình provider/model/key (không chứa secret thật)
- [x] **tests/test_smoke.py** — smoke test stdlib (fake LLM server, agent loop, tools, permission)
- [x] **Script chạy** — `run_local.sh` (Linux/Mac) + `run_local.bat` (Windows)
- [x] **README** — cập nhật phần "MVP với Ollama/LM Studio"
- [x] **requirements** — tách deps nặng (torch/fastapi) thành optional
- [x] **CLAUDE_FINAL_REPORT.md** — báo cáo cuối

## 3. Ràng buộc an toàn (đã đảm bảo)
- Không hard-code API key/secret → đọc từ env/.env
- `run_command` chặn lệnh phá hoại (rm -rf, format, shutdown, sudo...)
- Permission mode `plan` cho agent chỉ-đọc
- Giới hạn `max_steps` chống loop vô hạn

## 4. Ngoài phạm vi MVP (phát triển sau)
- RAG ngữ nghĩa cho codebase (embeddings)
- Subagents, hooks, custom slash commands
- Tích hợp VS Code (kiểu Cline)
