# AILOCAL.md — Hướng dẫn dự án cho AI-local agent

> Agent đọc file này tự động ở đầu mỗi phiên (`use_memory=True`).
> Viết vào đây những gì bạn không muốn giải thích lại mỗi lần.

## Dự án này là gì

AI-local là bộ công cụ LLM chạy hoàn toàn offline, kèm **agent framework** đa năng
(giống Claude Code / Cowork): tool use, MCP client, repo map, memory, skills, sessions.
Server tương thích Ollama + OpenAI API.

## Lệnh thường dùng

- Khởi động server: `python cli.py serve` (mặc định cổng 11434)
- Chat CLI: `python cli.py run <model>`
- Chạy agent: `python cli.py agent "nhiệm vụ" --verbose`
- Xem tools: `python cli.py tools`
- Quản lý skills: `python cli.py skills list|init`
- Quản lý memory: `python cli.py memory show|add "..."`
- Quản lý sessions: `python cli.py sessions list`

## Cấu trúc chính

- `model/` — kiến trúc GPT + HF backend
- `tools/` — 16 built-in tools (file, shell, web, git, repo_map, memory, skills)
- `agent.py` — agent executor (ReAct + function calling)
- `mcp_client.py` — kết nối MCP servers ngoài
- `repo_map.py` / `memory.py` / `sessions.py` / `skills.py` — tầng "trí nhớ" của agent
- `server.py` — REST API (Ollama + OpenAI + /v1/agent, /v1/tools, /v1/mcp, /v1/skills...)

## Quy ước code

- Python 3.10+, comment và message tiếng Việt
- 4 space indent, dùng type hints
- Tools trả về string (kể cả lỗi: prefix `[ERROR]`/`[OK]`/`[BLOCKED]`)
- Không thêm dependency nặng nếu stdlib đủ dùng

## Lưu ý quan trọng

- Model Vietnamese local (`chat_vi`) chỉ 1.8M params → dùng `mode=react`, kết quả hạn chế.
  Để agent mạnh, dùng model HF lớn (`mode=function_calling`).
- `.ai-local/` chứa sessions + auto-memory, đã gitignore (runtime state).
- `run_command` chặn lệnh phá hoại (rm, shutdown, sudo...).
