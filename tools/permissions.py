"""
Permission system cho AI-local agent (kiểu Claude Code permission modes).

Phân loại tool theo độ rủi ro và kiểm soát quyền thực thi qua "mode":

- "auto"     : cho phép tất cả (mặc định, như bypassPermissions)
- "plan"     : chỉ tool đọc — agent lập kế hoạch, KHÔNG ghi/sửa/chạy lệnh
- "approve"  : tool đọc tự do; tool ghi/nguy hiểm cần nằm trong allowlist
- "readonly" : đồng nghĩa "plan"

Ngoài mode, có thể cấu hình allow/deny theo tên tool.

Sử dụng:
    from tools.permissions import PermissionPolicy
    policy = PermissionPolicy(mode="plan")
    ok, reason = policy.check("write_file", {"path": "x"})
"""

from dataclasses import dataclass, field

# Tool chỉ đọc — luôn an toàn
READ_ONLY = frozenset({
    "read_file", "list_dir", "search_files", "glob", "repo_map", "rag_search",
    "recall", "todo_read", "list_skills",
    "git_status", "git_diff", "git_log",
    "web_fetch", "web_search",
})

# Tool ghi/sửa — thay đổi trạng thái
WRITE = frozenset({
    "write_file", "edit_file", "multi_edit", "apply_patch",
    "remember", "todo_write", "git_commit",
})

# Tool nguy hiểm — chạy lệnh tùy ý
DANGEROUS = frozenset({
    "run_command",
})


def classify(tool_name: str) -> str:
    """Phân loại tool: 'read' | 'write' | 'dangerous' | 'unknown'."""
    if tool_name in READ_ONLY:
        return "read"
    if tool_name in WRITE:
        return "write"
    if tool_name in DANGEROUS:
        return "dangerous"
    return "unknown"


@dataclass
class PermissionPolicy:
    """Chính sách quyền cho một lần chạy agent."""
    mode: str = "auto"                       # auto | plan | approve | readonly
    allow: set = field(default_factory=set)  # tool luôn cho phép (kể cả mode hạn chế)
    deny: set = field(default_factory=set)   # tool luôn chặn

    def check(self, tool_name: str, arguments: dict | None = None) -> tuple[bool, str]:
        """
        Kiểm tra một lời gọi tool có được phép không.

        Returns:
            (allowed, reason). Nếu allowed=False, reason mô tả vì sao bị chặn.
        """
        # Deny luôn thắng
        if tool_name in self.deny:
            return False, f"Tool '{tool_name}' nằm trong danh sách chặn (deny)."

        # Allow ghi đè mode hạn chế
        if tool_name in self.allow:
            return True, ""

        kind = classify(tool_name)

        if self.mode in ("auto", "bypass"):
            return True, ""

        if self.mode in ("plan", "readonly"):
            if kind == "read":
                return True, ""
            return False, (
                f"[CHẾ ĐỘ PLAN] Tool '{tool_name}' ({kind}) bị chặn. "
                f"Chỉ được dùng tool đọc để lập kế hoạch. "
                f"Hãy mô tả thay đổi bạn ĐỊNH làm thay vì thực thi."
            )

        if self.mode == "approve":
            if kind == "read":
                return True, ""
            return False, (
                f"[CẦN PHÊ DUYỆT] Tool '{tool_name}' ({kind}) cần được phê duyệt. "
                f"Thêm '{tool_name}' vào allowlist để cho phép, "
                f"hoặc đề xuất thay đổi để người dùng duyệt."
            )

        # Mode lạ → mặc định an toàn: chỉ cho đọc
        if kind == "read":
            return True, ""
        return False, f"Mode '{self.mode}' không xác định — chặn tool ghi để an toàn."


def make_policy(mode: str = "auto", allow=None, deny=None) -> PermissionPolicy:
    """Tạo policy từ tham số đơn giản (allow/deny là list tên tool)."""
    return PermissionPolicy(
        mode=mode or "auto",
        allow=set(allow or []),
        deny=set(deny or []),
    )
