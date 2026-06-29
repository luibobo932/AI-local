"""Git tools cho AI-local agent."""

import os
import subprocess


def _run_git(args: list[str], cwd: str = None, timeout: int = 30) -> str:
    """Chạy lệnh git và trả về output."""
    cwd = cwd or os.getcwd()
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout.rstrip()
        err = result.stderr.rstrip()
        if result.returncode != 0:
            return f"[ERROR git] {err or out}"
        return out or "[OK] (không có output)"
    except FileNotFoundError:
        return "[ERROR] git không được cài đặt."
    except subprocess.TimeoutExpired:
        return "[ERROR] git timeout."
    except Exception as e:
        return f"[ERROR] {e}"


def git_status(cwd: str = None) -> str:
    """Xem trạng thái git."""
    return _run_git(["status", "--short", "--branch"], cwd=cwd)


def git_diff(cwd: str = None, ref1: str = None, ref2: str = None, path: str = None) -> str:
    """Xem diff."""
    args = ["diff"]
    if ref1 and ref2:
        args += [ref1, ref2]
    elif ref1:
        args += [ref1]
    else:
        args += ["HEAD"]  # Staged + unstaged
    if path:
        args += ["--", path]
    result = _run_git(args, cwd=cwd)
    if len(result) > 10000:
        return result[:10000] + f"\n... (cắt bớt {len(result)-10000} ký tự)"
    return result


def git_log(cwd: str = None, n: int = 10, branch: str = None) -> str:
    """Xem lịch sử commit."""
    args = ["log", f"-{n}", "--oneline", "--decorate"]
    if branch:
        args.append(branch)
    return _run_git(args, cwd=cwd)


def git_commit(message: str, cwd: str = None, files: list[str] = None) -> str:
    """Stage files và tạo commit."""
    if not message or not message.strip():
        return "[ERROR] Commit message không được trống."

    cwd = cwd or os.getcwd()

    # Stage files
    if files:
        stage_result = _run_git(["add"] + files, cwd=cwd)
    else:
        stage_result = _run_git(["add", "-A"], cwd=cwd)

    if "[ERROR" in stage_result:
        return stage_result

    # Commit
    return _run_git(["commit", "-m", message], cwd=cwd)
