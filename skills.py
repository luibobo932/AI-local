"""
Skills system cho AI-local agent (lấy cảm hứng từ Claude Code skills).

Skill = một workflow đóng gói dạng markdown, load on-demand khi cần.
Mỗi skill là file .md trong skills/ với frontmatter mô tả:

    ---
    name: review-code
    description: Review code tìm bug và đề xuất cải thiện
    tools: read_file, search_files, git_diff
    ---

    # Hướng dẫn skill
    Bạn là reviewer. Hãy đọc diff, tìm bug...

Agent có thể: liệt kê skills, đọc skill, và áp dụng skill prompt vào task.

Sử dụng:
    from skills import list_skills, load_skill, get_skill_prompt
"""

import os
import re

_SKILLS_DIRS = ["skills", os.path.join(".ai-local", "skills")]


def _all_skill_dirs(root: str = ".") -> list[str]:
    root = os.path.abspath(os.path.expanduser(root))
    return [os.path.join(root, d) for d in _SKILLS_DIRS]


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Tách frontmatter YAML đơn giản và body."""
    meta = {}
    body = content
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, body.strip()


def list_skills(root: str = ".") -> list[dict]:
    """Liệt kê tất cả skills có sẵn."""
    skills = []
    seen = set()
    for d in _all_skill_dirs(root):
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".md"):
                continue
            name = os.path.splitext(fname)[0]
            if name in seen:
                continue
            seen.add(name)
            path = os.path.join(d, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    meta, _ = _parse_frontmatter(f.read())
                skills.append({
                    "name": meta.get("name", name),
                    "description": meta.get("description", ""),
                    "tools": meta.get("tools", ""),
                    "path": path,
                })
            except OSError:
                continue
    return skills


def load_skill(name: str, root: str = ".") -> dict | None:
    """Đọc một skill theo tên."""
    for d in _all_skill_dirs(root):
        path = os.path.join(d, f"{name}.md")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                meta, body = _parse_frontmatter(f.read())
            return {
                "name": meta.get("name", name),
                "description": meta.get("description", ""),
                "tools": [t.strip() for t in meta.get("tools", "").split(",") if t.strip()],
                "prompt": body,
                "path": path,
            }
    return None


def get_skill_prompt(name: str, root: str = ".") -> str:
    """Lấy prompt của skill để inject vào system prompt agent."""
    skill = load_skill(name, root)
    if skill is None:
        return ""
    return skill["prompt"]


def create_skill(name: str, description: str, prompt: str, tools: str = "", root: str = ".") -> str:
    """Tạo một skill mới."""
    root = os.path.abspath(os.path.expanduser(root))
    skills_dir = os.path.join(root, "skills")
    os.makedirs(skills_dir, exist_ok=True)
    path = os.path.join(skills_dir, f"{name}.md")

    content = f"""---
name: {name}
description: {description}
tools: {tools}
---

{prompt}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"[OK] Đã tạo skill '{name}' tại {path}"


def init_default_skills(root: str = ".") -> str:
    """Tạo một vài skill mẫu hữu ích."""
    created = []
    defaults = [
        {
            "name": "review-code",
            "description": "Review code tìm bug, lỗi tiềm ẩn và đề xuất cải thiện",
            "tools": "read_file, search_files, git_diff",
            "prompt": (
                "Bạn là một senior code reviewer. Hãy:\n"
                "1. Xem diff hoặc các file liên quan bằng git_diff / read_file\n"
                "2. Tìm bug, lỗi logic, vấn đề bảo mật, edge case chưa xử lý\n"
                "3. Đề xuất cải thiện ngắn gọn, ưu tiên vấn đề nghiêm trọng\n"
                "Trả lời bằng tiếng Việt, có dẫn chứng file:dòng cụ thể."
            ),
        },
        {
            "name": "explain-codebase",
            "description": "Giải thích cấu trúc và cách hoạt động của codebase",
            "tools": "list_dir, read_file, search_files",
            "prompt": (
                "Bạn là kỹ sư giúp người mới hiểu dự án. Hãy:\n"
                "1. Khám phá cấu trúc thư mục bằng list_dir\n"
                "2. Đọc các file entry point (main, app, server, cli)\n"
                "3. Giải thích: dự án làm gì, kiến trúc, luồng dữ liệu chính\n"
                "Trả lời tiếng Việt, dễ hiểu, có sơ đồ text nếu cần."
            ),
        },
        {
            "name": "fix-bug",
            "description": "Tìm và sửa bug dựa trên mô tả lỗi",
            "tools": "read_file, search_files, edit_file, run_command",
            "prompt": (
                "Bạn là kỹ sư debug. Quy trình:\n"
                "1. Tìm code liên quan bằng search_files\n"
                "2. Đọc kỹ để hiểu nguyên nhân gốc\n"
                "3. Sửa bằng edit_file (thay đổi tối thiểu, đúng trọng tâm)\n"
                "4. Kiểm tra lại bằng run_command nếu có test\n"
                "Giải thích nguyên nhân và cách sửa bằng tiếng Việt."
            ),
        },
        {
            "name": "research",
            "description": "Nghiên cứu một chủ đề bằng web search và tổng hợp",
            "tools": "web_search, web_fetch",
            "prompt": (
                "Bạn là trợ lý nghiên cứu. Hãy:\n"
                "1. Tìm kiếm web với web_search\n"
                "2. Đọc các nguồn quan trọng bằng web_fetch\n"
                "3. Tổng hợp thành câu trả lời có cấu trúc, kèm nguồn\n"
                "Trả lời tiếng Việt, khách quan, trích dẫn nguồn."
            ),
        },
    ]
    for sk in defaults:
        skill = load_skill(sk["name"], root)
        if skill is None:
            create_skill(sk["name"], sk["description"], sk["prompt"], sk["tools"], root)
            created.append(sk["name"])
    if created:
        return f"[OK] Đã tạo {len(created)} skill mẫu: {', '.join(created)}"
    return "[OK] Các skill mẫu đã tồn tại."


# ─── Tool wrappers ────────────────────────────────────────────────────────────

def list_skills_tool() -> str:
    """Tool cho agent: liệt kê skills."""
    skills = list_skills()
    if not skills:
        return "(Chưa có skill nào. Tạo bằng: python cli.py skills init)"
    lines = ["Skills có sẵn:"]
    for s in skills:
        lines.append(f"- {s['name']}: {s['description']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print(init_default_skills())
    else:
        for s in list_skills():
            print(f"{s['name']}: {s['description']}")
