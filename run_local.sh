#!/usr/bin/env bash
# Chạy Minion MVP trên Linux/Mac. KHÔNG cần torch/fastapi.
# Yêu cầu: Python 3.10+ và một LLM endpoint (Ollama / LM Studio).
set -e

cd "$(dirname "$0")"

# Tạo .env từ mẫu nếu chưa có
if [ ! -f .env ]; then
  echo "→ Tạo .env từ .env.example (hãy chỉnh model/provider nếu cần)"
  cp .env.example .env
fi

# Gợi ý kiểm tra Ollama
if command -v ollama >/dev/null 2>&1; then
  echo "→ Phát hiện Ollama. Đảm bảo model đã pull, ví dụ:"
  echo "    ollama pull qwen2.5-coder"
else
  echo "⚠ Chưa thấy 'ollama'. Cài tại https://ollama.com hoặc dùng LM Studio/OpenAI."
fi

CMD="${1:-serve}"
shift || true

case "$CMD" in
  serve)  exec python3 minion.py serve "$@" ;;
  chat)   exec python3 minion.py chat "$@" ;;
  agent)  exec python3 minion.py agent "$@" ;;
  *)      exec python3 minion.py "$CMD" "$@" ;;
esac
