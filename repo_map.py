"""
Repo Map — bản đồ codebase cho AI-local agent (lấy cảm hứng từ Aider).

Tạo một bản tóm tắt ngắn gọn của toàn bộ repository: danh sách file
kèm các symbol quan trọng (class, function, method) và signature.
Giúp agent "hiểu" cấu trúc codebase mà không cần đọc hết mọi file.

Xếp hạng file theo độ quan trọng (số symbol + số lần được tham chiếu)
để fit vào token budget giới hạn.

Sử dụng:
    from repo_map import build_repo_map
    print(build_repo_map(".", max_files=40))
"""

import ast
import fnmatch
import os
import re

# File/thư mục bỏ qua khi quét
_IGNORE_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".idea", ".vscode", "checkpoints", "models",
    ".pytest_cache", ".mypy_cache", "output", "data",
})
_IGNORE_FILES = frozenset({"*.pyc", "*.pyo", "*.so", "*.bin", "*.pt", "*.pkl"})

# Phần mở rộng được phân tích sâu (AST) — hiện hỗ trợ Python tốt nhất
_PY_EXT = {".py"}
# Các ngôn ngữ khác: trích xuất bằng regex
_REGEX_LANGS = {
    ".js": "javascript", ".ts": "typescript", ".jsx": "javascript",
    ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c",
}


def build_repo_map(
    root: str = ".",
    max_files: int = 50,
    max_symbols_per_file: int = 12,
    include_signatures: bool = True,
) -> str:
    """
    Tạo bản đồ codebase dạng text.

    Args:
        root: thư mục gốc repo
        max_files: số file tối đa hiển thị (xếp hạng theo độ quan trọng)
        max_symbols_per_file: số symbol tối đa mỗi file
        include_signatures: hiển thị signature function hay không

    Returns:
        Text bản đồ codebase
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        return f"[ERROR] Không phải thư mục: {root}"

    file_infos = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if any(fnmatch.fnmatch(fname, pat) for pat in _IGNORE_FILES):
                continue
            ext = os.path.splitext(fname)[1]
            if ext not in _PY_EXT and ext not in _REGEX_LANGS:
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root)
            try:
                if os.path.getsize(fpath) > 1024 * 1024:  # >1MB skip
                    continue
                symbols = _extract_symbols(fpath, ext)
                if symbols:
                    file_infos.append({
                        "path": rel,
                        "symbols": symbols,
                        "score": _score_file(rel, symbols),
                    })
            except Exception:
                continue

    if not file_infos:
        return f"# Repo map ({root})\n(Không tìm thấy file mã nguồn nào.)"

    # Xếp hạng và cắt
    file_infos.sort(key=lambda f: f["score"], reverse=True)
    shown = file_infos[:max_files]
    hidden = len(file_infos) - len(shown)

    lines = [f"# Repo map: {os.path.basename(root)} ({len(file_infos)} file mã nguồn)"]
    lines.append("# (file quan trọng nhất hiển thị trước; chỉ liệt kê symbol chính)\n")

    # Nhóm theo thư mục
    shown.sort(key=lambda f: f["path"])
    for info in shown:
        lines.append(f"📄 {info['path']}")
        for sym in info["symbols"][:max_symbols_per_file]:
            sig = sym["signature"] if include_signatures else sym["name"]
            indent = "    " if sym["kind"] == "method" else "  "
            kind_icon = {"class": "▸", "function": "•", "method": "·"}.get(sym["kind"], "•")
            lines.append(f"{indent}{kind_icon} {sig}")
        extra = len(info["symbols"]) - max_symbols_per_file
        if extra > 0:
            lines.append(f"    … +{extra} symbol nữa")
        lines.append("")

    if hidden > 0:
        lines.append(f"… và {hidden} file khác (ít symbol hơn).")

    return "\n".join(lines)


def _extract_symbols(fpath: str, ext: str) -> list[dict]:
    """Trích xuất symbols từ một file."""
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    if ext in _PY_EXT:
        return _extract_python(source)
    return _extract_regex(source, ext)


def _extract_python(source: str) -> list[dict]:
    """Dùng AST trích xuất class/function/method từ Python."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_regex(source, ".py")

    symbols = []

    def _signature(node) -> str:
        args = []
        a = node.args
        for arg in a.posonlyargs + a.args:
            args.append(arg.arg)
        if a.vararg:
            args.append("*" + a.vararg.arg)
        for arg in a.kwonlyargs:
            args.append(arg.arg)
        if a.kwarg:
            args.append("**" + a.kwarg.arg)
        return f"{node.name}({', '.join(args)})"

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [_name(b) for b in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            symbols.append({"kind": "class", "name": node.name, "signature": f"class {node.name}{base_str}"})
            # Methods
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if sub.name.startswith("_") and sub.name != "__init__":
                        continue
                    prefix = "async def " if isinstance(sub, ast.AsyncFunctionDef) else "def "
                    symbols.append({"kind": "method", "name": sub.name, "signature": prefix + _signature(sub)})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            symbols.append({"kind": "function", "name": node.name, "signature": prefix + _signature(node)})

    return symbols


def _name(node) -> str:
    """Lấy tên từ AST node (cho base class)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return "?"


def _extract_regex(source: str, ext: str) -> list[dict]:
    """Trích xuất symbols bằng regex cho ngôn ngữ ngoài Python."""
    symbols = []
    patterns = [
        (r"^\s*(?:export\s+)?(?:public\s+|private\s+)?class\s+(\w+)", "class"),
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", "function"),
        (r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>", "function"),
        (r"^\s*func\s+(\w+)\s*\(([^)]*)\)", "function"),  # Go
        (r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)", "function"),  # Rust
    ]
    for line in source.splitlines():
        for pat, kind in patterns:
            m = re.match(pat, line)
            if m:
                name = m.group(1)
                if len(m.groups()) > 1 and m.group(2) is not None:
                    sig = f"{name}({m.group(2).strip()})"
                else:
                    sig = name if kind != "class" else f"class {name}"
                symbols.append({"kind": kind, "name": name, "signature": sig})
                break
    return symbols


def _score_file(rel_path: str, symbols: list[dict]) -> float:
    """Xếp hạng độ quan trọng của file."""
    score = len(symbols)
    # File ở root quan trọng hơn
    depth = rel_path.count(os.sep)
    score -= depth * 0.5
    # File có class quan trọng hơn
    score += sum(2 for s in symbols if s["kind"] == "class")
    # Tên file gợi ý entry point
    base = os.path.basename(rel_path).lower()
    if base in ("main.py", "app.py", "server.py", "cli.py", "__init__.py", "agent.py", "index.js"):
        score += 5
    # Test file ít quan trọng hơn
    if "test" in base:
        score -= 3
    return score


# ─── Tool wrapper ─────────────────────────────────────────────────────────────

def repo_map_tool(root: str = ".", max_files: int = 40) -> str:
    """Tool: tạo repo map cho agent dùng."""
    return build_repo_map(root, max_files=max_files)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(build_repo_map(root))
