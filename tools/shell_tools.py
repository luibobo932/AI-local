"""Shell execution tool cho AI-local agent."""

import os
import shlex
import subprocess


# Lệnh nguy hiểm bị chặn
_BLOCKED = frozenset([
    "rm", "rmdir", "del", "format", "mkfs", "dd",
    "shutdown", "reboot", "halt", "poweroff",
    "sudo", "su", "chmod 777", "chmod -R 777",
    "curl | bash", "wget | bash",
    ":(){:|:&};:",  # fork bomb
])


def run_command(command: str, cwd: str = None, timeout: int = 30) -> str:
    """
    Chạy lệnh shell an toàn.

    - Chặn các lệnh phá hoại.
    - Timeout mặc định 30 giây.
    - Trả về stdout + stderr + exit code.
    """
    if not command or not command.strip():
        return "[ERROR] Lệnh trống."

    cmd_lower = command.lower().strip()
    for blocked in _BLOCKED:
        if cmd_lower.startswith(blocked + " ") or cmd_lower == blocked:
            return f"[BLOCKED] Lệnh '{blocked}' bị chặn vì lý do an toàn."

    cwd = cwd or os.getcwd()
    note = ""
    if not os.path.isdir(cwd):
        # Model local hay bịa placeholder (vd /path/to/project) → tự về thư mục
        # hiện tại kèm ghi chú, đỡ phí một bước agent chỉ để sửa cwd.
        note = f"[Lưu ý] cwd '{cwd}' không tồn tại — đã chạy trong '{os.getcwd()}' thay thế.\n"
        cwd = os.getcwd()

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        out_parts = []
        if note:
            out_parts.append(note.rstrip())
        if result.stdout:
            out_parts.append(result.stdout.rstrip())
        if result.stderr:
            out_parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        out_parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(out_parts) if out_parts else "[OK] (không có output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Lệnh quá thời gian {timeout}s"
    except Exception as e:
        return f"[ERROR] Không chạy được lệnh: {e}"
