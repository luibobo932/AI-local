"""
Code tools cho AI-local coding agent (làm cho agent "xịn" như Claude Code).

- glob_files: tìm file theo glob pattern (như Glob của Claude Code)
- multi_edit: nhiều find/replace trong cùng một file (như MultiEdit)
- apply_patch: áp một patch nhiều đoạn (search/replace blocks)
- todo_write / todo_read: agent tự lập & theo dõi checklist nhiều bước (như TodoWrite)
"""

import fnmatch
import glob as _glob
import json
import os

# Thư mục/file bỏ qua khi glob
_IGNORE = {".git", "__pycache__", "node_modules", ".venv", "venv",
           ".ai-local", "checkpoints", "models", "output"}


def glob_files(pattern: str, root: str = ".", max_results: int = 200) -> str:
    """
    Tìm file theo glob pattern (hỗ trợ **). Sắp xếp theo thời gian sửa đổi mới nhất.

    Ví dụ: glob_files("**/*.py"), glob_files("server.py"), glob_files("tools/*.py")
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        return f"[ERROR] Không phải thư mục: {root}"

    # Hỗ trợ ** recursive
    full_pattern = os.path.join(root, pattern)
    matches = _glob.glob(full_pattern, recursive=True)

    # Lọc bỏ thư mục ignore
    results = []
    for m in matches:
        parts = set(os.path.relpath(m, root).split(os.sep))
        if parts & _IGNORE:
            continue
        if os.path.isfile(m):
            results.append(m)

    if not results:
        return f"Không tìm thấy file khớp '{pattern}' trong {root}"

    # Sắp xếp theo mtime giảm dần
    results.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    results = results[:max_results]

    rels = [os.path.relpath(r, root) for r in results]
    header = f"# {len(rels)} file khớp '{pattern}':"
    return header + "\n" + "\n".join(rels)


def multi_edit(path: str, edits: list) -> str:
    """
    Áp nhiều find/replace tuần tự trong cùng một file (atomic — hoặc tất cả hoặc không).

    Args:
        path: đường dẫn file
        edits: list các dict {"old_text": "...", "new_text": "..."}
               (cũng chấp nhận JSON string)
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"[ERROR] File không tồn tại: {path}"

    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except json.JSONDecodeError:
            return "[ERROR] 'edits' phải là list dict {old_text, new_text}"
    if not isinstance(edits, list) or not edits:
        return "[ERROR] 'edits' phải là list không rỗng"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"[ERROR] Không đọc được file: {e}"

    # Validate & áp tuần tự lên bản nháp
    draft = content
    for i, e in enumerate(edits, 1):
        old = e.get("old_text", "")
        new = e.get("new_text", "")
        if not old:
            return f"[ERROR] Edit #{i}: thiếu old_text"
        count = draft.count(old)
        if count == 0:
            return f"[ERROR] Edit #{i}: không tìm thấy:\n{old[:150]}"
        if count > 1:
            return f"[ERROR] Edit #{i}: tìm thấy {count} lần, cần đoạn text dài hơn để duy nhất:\n{old[:150]}"
        draft = draft.replace(old, new, 1)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(draft)
    except Exception as e:
        return f"[ERROR] Không ghi được file: {e}"

    return f"[OK] Đã áp {len(edits)} chỉnh sửa vào {path}"


def apply_patch(path: str, patch: str) -> str:
    """
    Áp patch dạng search/replace blocks (đơn giản, không cần dòng số).

    Định dạng patch (mỗi block):
        <<<<<<< SEARCH
        đoạn code cũ
        =======
        đoạn code mới
        >>>>>>> REPLACE

    Có thể có nhiều block trong một patch.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"[ERROR] File không tồn tại: {path}"

    blocks = _parse_search_replace(patch)
    if not blocks:
        return ("[ERROR] Không parse được patch. Dùng định dạng:\n"
                "<<<<<<< SEARCH\n<cũ>\n=======\n<mới>\n>>>>>>> REPLACE")

    edits = [{"old_text": s, "new_text": r} for s, r in blocks]
    return multi_edit(path, edits)


def _parse_search_replace(patch: str) -> list:
    """Parse các block SEARCH/REPLACE."""
    blocks = []
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("<<<<<<<") and "SEARCH" in lines[i]:
            search_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("======="):
                search_lines.append(lines[i])
                i += 1
            i += 1  # bỏ qua =======
            replace_lines = []
            while i < len(lines) and not lines[i].strip().startswith(">>>>>>>"):
                replace_lines.append(lines[i])
                i += 1
            i += 1  # bỏ qua >>>>>>> REPLACE
            blocks.append(("\n".join(search_lines), "\n".join(replace_lines)))
        else:
            i += 1
    return blocks


# ─── TODO / plan tracking (như TodoWrite của Claude Code) ─────────────────────

_TODO_PATH = os.path.join(".ai-local", "todos.json")


def _todo_file() -> str:
    os.makedirs(os.path.dirname(_TODO_PATH), exist_ok=True)
    return _TODO_PATH


def todo_write(todos: list) -> str:
    """
    Ghi/cập nhật danh sách công việc cho nhiệm vụ nhiều bước.

    Args:
        todos: list dict {"content": "...", "status": "pending|in_progress|completed"}
               (cũng chấp nhận JSON string)
    """
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            return "[ERROR] 'todos' phải là list dict {content, status}"
    if not isinstance(todos, list):
        return "[ERROR] 'todos' phải là list"

    valid_status = {"pending", "in_progress", "completed"}
    cleaned = []
    for t in todos:
        if isinstance(t, str):
            t = {"content": t, "status": "pending"}
        content = t.get("content", "").strip()
        status = t.get("status", "pending")
        if not content:
            continue
        if status not in valid_status:
            status = "pending"
        cleaned.append({"content": content, "status": status})

    with open(_todo_file(), "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    return _render_todos(cleaned)


def todo_read() -> str:
    """Đọc danh sách công việc hiện tại."""
    path = os.path.join(".ai-local", "todos.json")
    if not os.path.isfile(path):
        return "(Chưa có công việc nào.)"
    try:
        with open(path, "r", encoding="utf-8") as f:
            todos = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "(Chưa có công việc nào.)"
    return _render_todos(todos)


def _render_todos(todos: list) -> str:
    if not todos:
        return "(Danh sách công việc rỗng.)"
    icon = {"pending": "☐", "in_progress": "▶", "completed": "☑"}
    lines = ["# Danh sách công việc:"]
    done = sum(1 for t in todos if t.get("status") == "completed")
    for t in todos:
        lines.append(f"  {icon.get(t.get('status'), '☐')} {t.get('content', '')}")
    lines.append(f"\nTiến độ: {done}/{len(todos)} hoàn thành")
    return "\n".join(lines)
