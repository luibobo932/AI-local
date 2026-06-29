"""File system tools cho AI-local agent."""

import fnmatch
import os
import re


def read_file(path: str, start_line: int = 1, end_line: int = None) -> str:
    """Đọc nội dung file, có thể chỉ định phạm vi dòng."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"[ERROR] File không tồn tại: {path}"
    if not os.path.isfile(path):
        return f"[ERROR] Đây là thư mục, không phải file: {path}"
    try:
        size = os.path.getsize(path)
        if size > 2 * 1024 * 1024:  # 2MB limit
            return f"[ERROR] File quá lớn ({size / 1024:.0f}KB). Dùng start_line/end_line để đọc từng phần."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(1, start_line) - 1
        end = end_line if end_line else len(lines)
        selected = lines[start:end]
        numbered = [f"{start + i + 1:4d} | {line}" for i, line in enumerate(selected)]
        header = f"# {path} (dòng {start+1}-{start+len(selected)}/{len(lines)})\n"
        return header + "".join(numbered)
    except Exception as e:
        return f"[ERROR] Không đọc được file: {e}"


def write_file(path: str, content: str) -> str:
    """Ghi nội dung vào file (tạo mới hoặc ghi đè)."""
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        lines = content.count("\n") + 1
        return f"[OK] Đã ghi {lines} dòng vào {path}"
    except Exception as e:
        return f"[ERROR] Không ghi được file: {e}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Thay thế chính xác old_text bằng new_text trong file."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"[ERROR] File không tồn tại: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"[ERROR] Không tìm thấy đoạn text này trong file:\n{old_text[:200]}"
        count = content.count(old_text)
        if count > 1:
            return f"[ERROR] Tìm thấy {count} lần xuất hiện. Cung cấp đoạn text dài hơn để phân biệt."
        new_content = content.replace(old_text, new_text, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"[OK] Đã thay thế trong {path}"
    except Exception as e:
        return f"[ERROR] Không sửa được file: {e}"


def list_dir(path: str = ".", recursive: bool = False, pattern: str = None) -> str:
    """Liệt kê nội dung thư mục."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"[ERROR] Đường dẫn không tồn tại: {path}"
    if not os.path.isdir(path):
        return f"[ERROR] Không phải thư mục: {path}"

    results = []
    try:
        if recursive:
            for root, dirs, files in os.walk(path):
                # Bỏ qua thư mục ẩn và __pycache__
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                rel_root = os.path.relpath(root, path)
                prefix = "" if rel_root == "." else rel_root + "/"
                for fname in sorted(files):
                    fpath = prefix + fname
                    if pattern and not fnmatch.fnmatch(fname, pattern):
                        continue
                    full = os.path.join(root, fname)
                    size = os.path.getsize(full)
                    results.append(f"  {fpath:60s} {_fmt_size(size):>8s}")
                if len(results) > 500:
                    results.append(f"  ... (cắt bớt, đã liệt kê {len(results)} mục)")
                    break
        else:
            entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name))
            for e in entries:
                if e.name.startswith("."):
                    continue
                if pattern and not fnmatch.fnmatch(e.name, pattern):
                    continue
                if e.is_dir():
                    results.append(f"  📁 {e.name}/")
                else:
                    size = e.stat().st_size
                    results.append(f"  📄 {e.name:55s} {_fmt_size(size):>8s}")

        if not results:
            return f"Thư mục rỗng: {path}"
        return f"# {path}\n" + "\n".join(results)
    except PermissionError:
        return f"[ERROR] Không có quyền đọc: {path}"


def search_files(pattern: str, path: str = ".", file_glob: str = None, max_results: int = 50) -> str:
    """Tìm kiếm text hoặc regex trong các file."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"[ERROR] Đường dẫn không tồn tại: {path}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        # Fallback sang tìm literal
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    results = []
    scanned = 0

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", ".git", "node_modules")]
        for fname in files:
            if file_glob and not fnmatch.fnmatch(fname, file_glob):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = os.path.relpath(fpath, path)
                            results.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(results) >= max_results:
                                results.append(f"... (dừng ở {max_results} kết quả)")
                                return "\n".join(results)
                scanned += 1
            except (PermissionError, IsADirectoryError):
                continue

    if not results:
        return f"Không tìm thấy '{pattern}' trong {scanned} file."
    return f"# Tìm thấy {len(results)} kết quả cho '{pattern}':\n" + "\n".join(results)


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
