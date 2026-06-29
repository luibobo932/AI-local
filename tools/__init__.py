"""
Tool registry cho AI-local agent framework.

Mỗi tool là một function với:
  - Tên (snake_case)
  - Schema OpenAI JSON (cho function calling)
  - Hàm thực thi đồng bộ

Sử dụng:
    from tools import TOOL_REGISTRY, call_tool, get_tool_schemas
"""

from tools.file_tools import (
    read_file, write_file, list_dir, search_files, edit_file
)
from tools.shell_tools import run_command
from tools.web_tools import web_fetch, web_search
from tools.git_tools import git_status, git_diff, git_log, git_commit


# ─── Lazy wrappers cho tools ở module gốc (tránh import vòng) ──────────────────

def _repo_map(root: str = ".", max_files: int = 40) -> str:
    from repo_map import repo_map_tool
    return repo_map_tool(root, max_files)


def _remember(fact: str, category: str = "general") -> str:
    from memory import remember_tool
    return remember_tool(fact, category)


def _recall() -> str:
    from memory import recall_tool
    return recall_tool()


def _list_skills() -> str:
    from skills import list_skills_tool
    return list_skills_tool()


# Registry: name → callable
TOOL_REGISTRY: dict[str, callable] = {
    "read_file":    read_file,
    "write_file":   write_file,
    "edit_file":    edit_file,
    "list_dir":     list_dir,
    "search_files": search_files,
    "run_command":  run_command,
    "web_fetch":    web_fetch,
    "web_search":   web_search,
    "git_status":   git_status,
    "git_diff":     git_diff,
    "git_log":      git_log,
    "git_commit":   git_commit,
    "repo_map":     _repo_map,
    "remember":     _remember,
    "recall":       _recall,
    "list_skills":  _list_skills,
}

# OpenAI function-calling schemas
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Đọc nội dung một file. Trả về text hoặc lỗi nếu file không tồn tại.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file cần đọc"},
                    "start_line": {"type": "integer", "description": "Dòng bắt đầu (1-indexed, mặc định 1)"},
                    "end_line": {"type": "integer", "description": "Dòng kết thúc (mặc định đọc hết)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Ghi nội dung vào file (tạo mới hoặc ghi đè). Tạo thư mục cha nếu chưa có.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file"},
                    "content": {"type": "string", "description": "Nội dung cần ghi"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Thay thế một đoạn text cụ thể trong file (tìm-và-thay-thế chính xác).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file"},
                    "old_text": {"type": "string", "description": "Đoạn text cần tìm (phải tồn tại chính xác trong file)"},
                    "new_text": {"type": "string", "description": "Đoạn text thay thế"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Liệt kê file và thư mục con trong một directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn thư mục (mặc định: thư mục hiện tại)"},
                    "recursive": {"type": "boolean", "description": "Liệt kê đệ quy hay không (mặc định false)"},
                    "pattern": {"type": "string", "description": "Glob pattern lọc file, ví dụ *.py"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Tìm kiếm text hoặc regex trong các file. Trả về danh sách kết quả (file:dòng:nội dung).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Chuỗi hoặc regex cần tìm"},
                    "path": {"type": "string", "description": "Thư mục tìm kiếm (mặc định: thư mục hiện tại)"},
                    "file_glob": {"type": "string", "description": "Lọc file theo glob, ví dụ *.py"},
                    "max_results": {"type": "integer", "description": "Số kết quả tối đa (mặc định 50)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Chạy lệnh shell. Trả về stdout, stderr và exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Lệnh shell cần chạy"},
                    "cwd": {"type": "string", "description": "Thư mục làm việc (mặc định: thư mục hiện tại)"},
                    "timeout": {"type": "integer", "description": "Timeout giây (mặc định 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Lấy nội dung một URL (HTML/JSON/text). Tự động trích xuất text thuần từ HTML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL cần lấy"},
                    "extract_text": {"type": "boolean", "description": "Trích xuất text thuần từ HTML (mặc định true)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Tìm kiếm web qua DuckDuckGo. Trả về danh sách kết quả (tiêu đề, URL, mô tả).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Từ khóa tìm kiếm"},
                    "max_results": {"type": "integer", "description": "Số kết quả tối đa (mặc định 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Xem trạng thái git của repository hiện tại.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cwd": {"type": "string", "description": "Thư mục git repo (mặc định: thư mục hiện tại)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Xem diff của các thay đổi chưa commit hoặc giữa 2 commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cwd": {"type": "string", "description": "Thư mục git repo"},
                    "ref1": {"type": "string", "description": "Commit/branch tham chiếu 1 (mặc định: staged/working tree)"},
                    "ref2": {"type": "string", "description": "Commit/branch tham chiếu 2"},
                    "path": {"type": "string", "description": "Giới hạn diff trong path này"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Xem lịch sử commit git.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cwd": {"type": "string", "description": "Thư mục git repo"},
                    "n": {"type": "integer", "description": "Số commit hiển thị (mặc định 10)"},
                    "branch": {"type": "string", "description": "Branch cần xem (mặc định: branch hiện tại)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage tất cả thay đổi và tạo commit mới.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                    "cwd": {"type": "string", "description": "Thư mục git repo"},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File cần stage (mặc định: tất cả thay đổi)",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_map",
            "description": "Tạo bản đồ codebase: danh sách file kèm class/function chính. Dùng để hiểu nhanh cấu trúc dự án trước khi đọc chi tiết.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Thư mục gốc dự án (mặc định: hiện tại)"},
                    "max_files": {"type": "integer", "description": "Số file tối đa (mặc định 40)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Ghi nhớ một điều quan trọng vào bộ nhớ dài hạn (MEMORY.md) để dùng cho các phiên sau. Dùng khi học được sở thích người dùng, quy ước dự án, hoặc thông tin cần nhớ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "Nội dung cần nhớ"},
                    "category": {"type": "string", "description": "Phân loại: general/preference/fact/command (mặc định general)"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Đọc lại tất cả ghi nhớ đã lưu trong bộ nhớ dài hạn.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "Liệt kê các skill (workflow đóng gói) có sẵn để áp dụng cho nhiệm vụ.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def get_tool_schemas(names: list[str] | None = None) -> list[dict]:
    """Lấy schemas của tools theo tên (mặc định: tất cả)."""
    if names is None:
        return TOOL_SCHEMAS
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in names]


def call_tool(name: str, arguments: dict) -> str:
    """Gọi một tool theo tên và trả về kết quả dạng string."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"[ERROR] Tool '{name}' không tồn tại. Có sẵn: {list(TOOL_REGISTRY)}"
    try:
        result = fn(**arguments)
        return str(result) if not isinstance(result, str) else result
    except TypeError as e:
        return f"[ERROR] Tham số sai cho tool '{name}': {e}"
    except Exception as e:
        return f"[ERROR] Tool '{name}' lỗi: {type(e).__name__}: {e}"
