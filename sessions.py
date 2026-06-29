"""
Session persistence cho AI-local agent.

Lưu lịch sử các phiên agent xuống đĩa (JSON + markdown log) để:
- Resume một phiên dở dang
- Xem lại lịch sử làm việc
- Chia sẻ transcript

Lưu trong .ai-local/sessions/<session_id>.json
Markdown log trong .ai-local/sessions/<session_id>.md

Sử dụng:
    from sessions import SessionStore
    store = SessionStore()
    sid = store.create("Sửa bug trong server.py")
    store.append_message(sid, "user", "...")
    store.save_agent_result(sid, result)
"""

import json
import os
import time
import uuid
from datetime import datetime

_SESSIONS_DIR = os.path.join(".ai-local", "sessions")


class SessionStore:
    """Quản lý lưu trữ các phiên agent/chat."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.dir = os.path.join(self.root, _SESSIONS_DIR)
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.dir, f"{session_id}.json")

    def _md_path(self, session_id: str) -> str:
        return os.path.join(self.dir, f"{session_id}.md")

    def create(self, title: str = "", model: str = "") -> str:
        """Tạo phiên mới, trả về session_id."""
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        data = {
            "id": session_id,
            "title": title or "Phiên không tên",
            "model": model,
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
            "agent_runs": [],
        }
        self._write(session_id, data)
        return session_id

    def _write(self, session_id: str, data: dict):
        data["updated_at"] = time.time()
        with open(self._path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._write_markdown(session_id, data)

    def load(self, session_id: str) -> dict | None:
        """Đọc một phiên."""
        path = self._path(session_id)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def append_message(self, session_id: str, role: str, content: str):
        """Thêm một message vào phiên."""
        data = self.load(session_id)
        if data is None:
            return
        data["messages"].append({
            "role": role,
            "content": content,
            "ts": time.time(),
        })
        # Tự đặt title từ message đầu nếu chưa có
        if data["title"] == "Phiên không tên" and role == "user":
            data["title"] = content[:60]
        self._write(session_id, data)

    def save_agent_run(self, session_id: str, result: dict):
        """Lưu kết quả một lần chạy agent (dict từ AgentResult)."""
        data = self.load(session_id)
        if data is None:
            return
        data["agent_runs"].append({
            "answer": result.get("answer", ""),
            "model": result.get("model", ""),
            "elapsed": result.get("elapsed", 0),
            "steps": result.get("steps", []),
            "ts": time.time(),
        })
        self._write(session_id, data)

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """Liệt kê các phiên, mới nhất trước."""
        sessions = []
        if not os.path.isdir(self.dir):
            return []
        for fname in os.listdir(self.dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "id": data["id"],
                    "title": data.get("title", ""),
                    "model": data.get("model", ""),
                    "updated_at": data.get("updated_at", 0),
                    "message_count": len(data.get("messages", [])),
                    "agent_run_count": len(data.get("agent_runs", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions[:limit]

    def delete(self, session_id: str) -> bool:
        """Xóa một phiên."""
        deleted = False
        for path in (self._path(session_id), self._md_path(session_id)):
            if os.path.isfile(path):
                os.remove(path)
                deleted = True
        return deleted

    def get_messages_for_resume(self, session_id: str) -> list[dict]:
        """Lấy messages (role/content) để tiếp tục phiên trong model."""
        data = self.load(session_id)
        if data is None:
            return []
        return [{"role": m["role"], "content": m["content"]} for m in data.get("messages", [])]

    def _write_markdown(self, session_id: str, data: dict):
        """Ghi transcript markdown để dễ đọc/chia sẻ."""
        lines = [
            f"# {data.get('title', 'Phiên')}",
            f"",
            f"- ID: `{session_id}`",
            f"- Model: {data.get('model', 'N/A')}",
            f"- Tạo: {datetime.fromtimestamp(data.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')}",
            f"",
            "---",
            "",
        ]
        for msg in data.get("messages", []):
            role = {"user": "👤 User", "assistant": "🤖 Assistant", "system": "⚙️ System"}.get(
                msg["role"], msg["role"]
            )
            lines.append(f"### {role}")
            lines.append(msg["content"])
            lines.append("")

        for i, run in enumerate(data.get("agent_runs", []), 1):
            lines.append(f"### 🤖 Agent run #{i} ({run.get('model', '')}, {run.get('elapsed', 0):.1f}s)")
            for step in run.get("steps", []):
                lines.append(f"**Bước {step.get('step', '?')}**")
                if step.get("thought"):
                    lines.append(f"> {step['thought']}")
                for tc in step.get("tool_calls", []):
                    lines.append(f"- 🔧 `{tc.get('name', '')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})`")
            lines.append(f"\n**Kết quả:** {run.get('answer', '')}")
            lines.append("")

        with open(self._md_path(session_id), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    import sys
    store = SessionStore()
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for s in store.list_sessions():
            ts = datetime.fromtimestamp(s["updated_at"]).strftime("%Y-%m-%d %H:%M")
            print(f"{s['id']}  [{ts}]  {s['title']}  ({s['message_count']} msg)")
    else:
        print("Dùng: python sessions.py list")
