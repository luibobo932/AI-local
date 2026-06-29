"""
Memory system cho AI-local agent (lấy cảm hứng từ Claude Code).

Hai tầng bộ nhớ:
1. AILOCAL.md — project memory người dùng viết (hướng dẫn, quy ước, lệnh build).
   Tự động load vào system prompt mỗi phiên agent.
2. MEMORY.md — auto-memory agent tự ghi lại điều học được qua các phiên.

Cả hai đều là file markdown trong thư mục dự án (hoặc .ai-local/).

Sử dụng:
    from memory import load_project_context, remember, get_memory
"""

import os
from datetime import datetime

# Tên file project memory được tìm kiếm (ưu tiên theo thứ tự)
_PROJECT_MEMORY_NAMES = ["AILOCAL.md", "AI_LOCAL.md", "CLAUDE.md", "AGENTS.md"]
_AUTO_MEMORY_NAME = "MEMORY.md"
_MEMORY_DIR = ".ai-local"

# Giới hạn load (giống Claude Code: ~25KB / 200 dòng đầu)
_MAX_MEMORY_BYTES = 25 * 1024
_MAX_MEMORY_LINES = 200


def find_project_memory(root: str = ".") -> str | None:
    """Tìm file project memory (AILOCAL.md / CLAUDE.md...) trong dự án."""
    root = os.path.abspath(os.path.expanduser(root))
    # Tìm ở root và trong .ai-local/
    search_dirs = [root, os.path.join(root, _MEMORY_DIR)]
    for d in search_dirs:
        for name in _PROJECT_MEMORY_NAMES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    return None


def _auto_memory_path(root: str = ".") -> str:
    """Đường dẫn file auto-memory (tạo thư mục .ai-local/ nếu cần)."""
    root = os.path.abspath(os.path.expanduser(root))
    mem_dir = os.path.join(root, _MEMORY_DIR)
    os.makedirs(mem_dir, exist_ok=True)
    return os.path.join(mem_dir, _AUTO_MEMORY_NAME)


def _truncate(text: str) -> str:
    """Cắt theo giới hạn dòng/byte."""
    lines = text.splitlines()
    if len(lines) > _MAX_MEMORY_LINES:
        lines = lines[:_MAX_MEMORY_LINES]
        lines.append(f"… (cắt bớt, hiển thị {_MAX_MEMORY_LINES} dòng đầu)")
    out = "\n".join(lines)
    if len(out.encode("utf-8")) > _MAX_MEMORY_BYTES:
        out = out.encode("utf-8")[:_MAX_MEMORY_BYTES].decode("utf-8", errors="ignore")
        out += "\n… (cắt bớt theo dung lượng)"
    return out


def load_project_context(root: str = ".", include_auto: bool = True) -> str:
    """
    Load toàn bộ context bộ nhớ của dự án để inject vào system prompt.

    Gồm: project memory (AILOCAL.md) + auto-memory (MEMORY.md).
    """
    parts = []

    pm_path = find_project_memory(root)
    if pm_path:
        try:
            with open(pm_path, "r", encoding="utf-8", errors="ignore") as f:
                content = _truncate(f.read())
            parts.append(f"# Project memory ({os.path.basename(pm_path)})\n{content}")
        except OSError:
            pass

    if include_auto:
        am_path = os.path.join(os.path.abspath(root), _MEMORY_DIR, _AUTO_MEMORY_NAME)
        if os.path.isfile(am_path):
            try:
                with open(am_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = _truncate(f.read())
                if content.strip():
                    parts.append(f"# Điều đã học (auto-memory)\n{content}")
            except OSError:
                pass

    return "\n\n".join(parts)


def get_memory(root: str = ".") -> str:
    """Đọc nội dung auto-memory hiện tại."""
    am_path = os.path.join(os.path.abspath(root), _MEMORY_DIR, _AUTO_MEMORY_NAME)
    if not os.path.isfile(am_path):
        return "(Chưa có ghi nhớ nào.)"
    with open(am_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def remember(fact: str, category: str = "general", root: str = ".") -> str:
    """
    Ghi một điều cần nhớ vào auto-memory (MEMORY.md).

    Args:
        fact: nội dung cần nhớ
        category: phân loại (general, preference, fact, command, ...)
        root: thư mục dự án
    """
    if not fact or not fact.strip():
        return "[ERROR] Nội dung ghi nhớ trống."

    path = _auto_memory_path(root)
    fact = fact.strip()

    # Khởi tạo file nếu chưa có
    if not os.path.isfile(path):
        header = (
            "# Auto-memory — AI-local agent\n\n"
            "_File này do agent tự ghi lại những điều học được qua các phiên làm việc._\n\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)

    # Đọc để tránh ghi trùng
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        existing = f.read()
    if fact in existing:
        return f"[OK] Đã có sẵn ghi nhớ này, bỏ qua."

    timestamp = datetime.now().strftime("%Y-%m-%d")
    entry = f"- **[{category}]** {fact} _(ghi {timestamp})_\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)

    return f"[OK] Đã ghi nhớ: {fact[:80]}"


def forget(keyword: str, root: str = ".") -> str:
    """Xóa các dòng ghi nhớ chứa keyword."""
    path = os.path.join(os.path.abspath(root), _MEMORY_DIR, _AUTO_MEMORY_NAME)
    if not os.path.isfile(path):
        return "(Chưa có ghi nhớ nào để xóa.)"

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    kept = [ln for ln in lines if not (ln.strip().startswith("-") and keyword.lower() in ln.lower())]
    removed = len(lines) - len(kept)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(kept)

    return f"[OK] Đã xóa {removed} ghi nhớ chứa '{keyword}'."


def init_project_memory(root: str = ".") -> str:
    """Tạo file AILOCAL.md mẫu cho dự án."""
    path = os.path.join(os.path.abspath(root), _PROJECT_MEMORY_NAMES[0])
    if os.path.isfile(path):
        return f"[OK] {_PROJECT_MEMORY_NAMES[0]} đã tồn tại."

    template = """# AILOCAL.md — Hướng dẫn dự án cho AI-local agent

> File này được agent đọc tự động ở đầu mỗi phiên. Viết vào đây những gì
> bạn không muốn phải giải thích lại mỗi lần.

## Dự án này là gì
(mô tả ngắn gọn)

## Lệnh thường dùng
- Build: `...`
- Test: `...`
- Chạy: `...`

## Quy ước code
- (ví dụ: dùng 4 space indent, comment tiếng Việt, ...)

## Lưu ý quan trọng
- (những điều agent cần biết trước khi làm việc)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)
    return f"[OK] Đã tạo {_PROJECT_MEMORY_NAMES[0]}. Hãy chỉnh sửa nội dung cho phù hợp."


# ─── Tool wrappers ────────────────────────────────────────────────────────────

def remember_tool(fact: str, category: str = "general") -> str:
    """Tool cho agent: ghi nhớ một điều."""
    return remember(fact, category)


def recall_tool() -> str:
    """Tool cho agent: đọc lại tất cả ghi nhớ."""
    return get_memory()
