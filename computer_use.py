"""Computer-use cục bộ cho Minion.

Module này chỉ chạy thao tác trên máy khi UI gửi cờ bật rõ ràng.
Không gọi cloud, không tự quyết định thao tác ngoài lệnh người dùng.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import unicodedata
import urllib.parse
import difflib
import hashlib
import json
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ComputerUseResult:
    handled: bool
    ok: bool
    message: str
    action: str = ""
    data: dict = field(default_factory=dict)
    risk_level: str = "safe"
    needs_approval: bool = False
    approval_id: str = ""
    artifacts: list[dict] = field(default_factory=list)


_REPO_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _REPO_ROOT / "minion.config.json"
_DEFAULT_CONFIG = {
    "permission_mode": "ask_when_risky",
    "max_agent_steps": 8,
    "command_timeout_seconds": 20,
    "allowed_workspace_roots": ["."],
    "screenshot_dir": "output/computer-use",
}
_PENDING_APPROVALS: dict[str, dict] = {}
_APPROVAL_TTL_SECONDS = 600


def _load_minion_config() -> dict:
    config = dict(_DEFAULT_CONFIG)
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            config.update(raw)
    except Exception:
        pass
    return config


def _minion_config() -> dict:
    return _load_minion_config()


def _plain_text(text: str) -> str:
    text = text.lower().replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def _approval_key(command: str) -> str:
    return re.sub(r"\s+", " ", command or "").strip()


def _purge_expired_approvals() -> None:
    now = time.time()
    expired = [
        approval_id
        for approval_id, item in _PENDING_APPROVALS.items()
        if now - float(item.get("created_at", 0)) > _APPROVAL_TTL_SECONDS
    ]
    for approval_id in expired:
        _PENDING_APPROVALS.pop(approval_id, None)


def _consume_approval(command_key: str, approval_token: str | None) -> bool:
    if not approval_token:
        return False
    _purge_expired_approvals()
    item = _PENDING_APPROVALS.get(approval_token)
    if not item or item.get("command_key") != command_key:
        return False
    _PENDING_APPROVALS.pop(approval_token, None)
    return True


def _approval_result(command: str, action: str, risk_level: str, reason: str) -> ComputerUseResult:
    approval_id = uuid.uuid4().hex
    command_key = _approval_key(command)
    _PENDING_APPROVALS[approval_id] = {
        "command_key": command_key,
        "command": command,
        "action": action,
        "risk_level": risk_level,
        "reason": reason,
        "created_at": time.time(),
    }
    return ComputerUseResult(
        True,
        False,
        f"Minion cần anh xác nhận trước khi làm: {reason}",
        action or "approval_required",
        {
            "type": "approval_request",
            "approval_id": approval_id,
            "command": command,
            "action": action,
            "risk_level": risk_level,
            "reason": reason,
            "expires_in_seconds": _APPROVAL_TTL_SECONDS,
        },
        risk_level=risk_level,
        needs_approval=True,
        approval_id=approval_id,
    )


def _dangerous_shell_text(text: str) -> bool:
    plain = _plain_text(text)
    patterns = [
        r"\brm\s+-rf\b",
        r"\bremove-item\b.*\b-recurse\b",
        r"\brd\s+/s\b",
        r"\brmdir\s+/s\b",
        r"\bdel\s+/s\b",
        r"\bformat\b",
        r"\bdiskpart\b",
        r"\bshutdown\b",
        r"\brestart-computer\b",
        r"\bstop-computer\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-f",
        r"\breg\s+delete\b",
        r"\bbcdedit\b",
        r"\bcipher\s+/w\b",
    ]
    return any(re.search(pattern, plain) for pattern in patterns)


def _classify_command_risk(command: str, plain: str) -> tuple[str, str, str]:
    if _dangerous_shell_text(command):
        return "blocked", "blocked", "Lệnh thuộc nhóm nguy hiểm, Minion không tự chạy."
    if "chay lenh" in plain or "run command" in plain:
        return "shell", "needs_approval", "chạy lệnh shell trên máy"
    if "thay trong file" in plain or "replace in file" in plain:
        return "workspace_replace", "needs_approval", "sửa file trong workspace"
    if "dong cua so" in plain or "close window" in plain:
        return "close_window", "needs_approval", "đóng cửa sổ đang mở"
    if (
        "dat clipboard" in plain
        or "set clipboard" in plain
        or "ghi clipboard" in plain
        or "luu clipboard" in plain
        or "copy vao clipboard" in plain
    ):
        return "clipboard_write", "needs_approval", "ghi nội dung vào clipboard"
    return "", "safe", ""


def _looks_like_computer_command(plain: str) -> bool:
    if re.search(r"https?://", plain):
        return True
    if re.match(r"^(?:hay\s+)?(?:minion\s+)?(?:go|nhap|dan|click|double\s+click)\b", plain):
        return True
    if re.match(
        r"^(?:hay\s+)?(?:minion\s+)?mo\s+"
        r"(?:app|ung\s+dung|file|thu\s+muc|link|trang|"
        r"notepad|ghi\s+chu|chrome|edge|explorer|may\s+tinh|calculator|calc|"
        r"google|youtube|zalo|gmail|calendar|cmd|powershell)\b",
        plain,
    ):
        return True
    if re.match(r"^(?:hay\s+)?(?:minion\s+)?doi\s+[0-9]{1,2}\b", plain):
        return True

    keywords = [
        "tu lam",
        "auto lam",
        "agent",
        "cowork",
        "computer use",
        "dieu khien may",
        "dieu khien chuot",
        "con chuot",
        "chuot",
        "click",
        "double click",
        "bam phim",
        "nhan phim",
        "hotkey",
        "copy",
        "sao chep",
        "cut",
        "paste",
        "chon tat ca",
        "select all",
        "dong cua so",
        "chuyen cua so",
        "alt tab",
        "mo app",
        "mo ung dung",
        "mo file",
        "mo thu muc",
        "mo notepad",
        "mo chrome",
        "mo edge",
        "mo explorer",
        "mo may tinh",
        "mo link",
        "mo trang",
        "tim kiem",
        "tim google",
        "search",
        "youtube",
        "toa do chuot",
        "vi tri chuot",
        "chup man hinh",
        "xem man hinh",
        "xem ui",
        "doc ui",
        "doc man hinh",
        "click nut",
        "click text",
        "click ui",
        "click dong",
        "double click ui",
        "right click ui",
        "bam nut",
        "bam so",
        "nhap vao",
        "go vao",
        "dien vao",
        "set text",
        "clear text",
        "xoa text",
        "doc control",
        "doc text",
        "thong tin control",
        "focus",
        "danh sach cua so",
        "liet ke cua so",
        "xem cua so",
        "focus cua so",
        "chuyen den cua so",
        "kich hoat cua so",
        "thu nho cua so",
        "phong to cua so",
        "khoi phuc cua so",
        "dat cua so",
        "resize cua so",
        "doi kich thuoc cua so",
        "di chuyen cua so",
        "minimize window",
        "maximize window",
        "restore window",
        "resize window",
        "move window",
        "man hinh",
        "cua so dang mo",
        "cua so hien tai",
        "active window",
        "clipboard",
        "cuon",
        "scroll",
        "keo chuot",
        "drag",
        "chay lenh",
        "workspace",
        "trang thai workspace",
        "git status",
        "liet ke file",
        "list files",
        "tim file",
        "find file",
        "tim code",
        "search code",
        "doc file",
        "read file",
        "thay trong file",
        "replace in file",
    ]
    return any(k in plain for k in keywords)


def _screen_size() -> tuple[int, int]:
    if os.name != "nt":
        return 0, 0
    import ctypes

    return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)


def _mouse_position() -> tuple[int, int]:
    if os.name != "nt":
        return 0, 0
    import ctypes

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    point = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def _window_text(hwnd: int) -> str:
    if os.name != "nt" or not hwnd:
        return ""
    import ctypes

    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _window_class(hwnd: int) -> str:
    if os.name != "nt" or not hwnd:
        return ""
    import ctypes

    buffer = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value


def _window_rect(hwnd: int) -> tuple[int, int, int, int]:
    if os.name != "nt" or not hwnd:
        return 0, 0, 0, 0
    import ctypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return 0, 0, 0, 0
    return rect.left, rect.top, rect.right, rect.bottom


def _fallback_visible_window_hwnd() -> int:
    """Lay cua so that khi Windows bao nham foreground la lock-screen ao."""
    if os.name != "nt":
        return 0
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    candidates: list[int] = []
    skip_classes = {"Shell_TrayWnd", "WorkerW", "Progman"}
    skip_titles = {"Windows Default Lock Screen", "Program Manager"}

    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        title = _window_text(hwnd).strip()
        class_name = _window_class(hwnd)
        if not title or title in skip_titles or class_name in skip_classes:
            return True
        left, top, right, bottom = _window_rect(hwnd)
        if right - left < 80 or bottom - top < 80:
            return True
        candidates.append(int(hwnd))
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return candidates[0] if candidates else 0


def _active_hwnd() -> int:
    if os.name != "nt":
        return 0
    import ctypes

    hwnd = int(ctypes.windll.user32.GetForegroundWindow())
    title = _window_text(hwnd)
    class_name = _window_class(hwnd)
    if title == "Windows Default Lock Screen" and class_name == "Windows.UI.Core.CoreWindow":
        return _fallback_visible_window_hwnd() or hwnd
    return hwnd


def _visible_windows(limit: int = 80) -> list[dict]:
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    active = _active_hwnd()
    windows: list[dict] = []
    skip_classes = {"Shell_TrayWnd", "WorkerW", "Progman"}
    skip_titles = {"Windows Default Lock Screen", "Program Manager"}

    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if len(windows) >= limit:
            return False
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _window_text(hwnd).strip()
        class_name = _window_class(hwnd)
        if not title or title in skip_titles or class_name in skip_classes:
            return True
        left, top, right, bottom = _window_rect(hwnd)
        width = right - left
        height = bottom - top
        if width < 80 or height < 80:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        windows.append({
            "index": len(windows) + 1,
            "hwnd": int(hwnd),
            "title": title,
            "class_name": class_name,
            "pid": int(pid.value),
            "is_active": int(hwnd) == active,
            "rect": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": width,
                "height": height,
            },
        })
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return windows


def _find_window(query: str) -> dict | None:
    windows = _visible_windows(limit=120)
    query = query.strip()
    if not query:
        return None
    index_match = re.search(r"\b(?:so|#)?\s*([0-9]{1,2})\b", _plain_text(query))
    if index_match:
        index = int(index_match.group(1))
        for item in windows:
            if item["index"] == index:
                return item

    plain_query = _plain_text(query)
    exact = None
    partial = None
    for item in windows:
        haystack = _plain_text(" ".join([item.get("title", ""), item.get("class_name", "")]))
        if plain_query == _plain_text(item.get("title", "")):
            exact = item
            break
        if plain_query in haystack and partial is None:
            partial = item
    return exact or partial


def _window_list_result(limit: int = 40) -> ComputerUseResult:
    windows = _visible_windows(limit=limit)
    if not windows:
        return ComputerUseResult(True, False, "Chưa đọc được danh sách cửa sổ đang mở.", "window_list", {"type": "window_list", "windows": []})

    lines = ["Các cửa sổ đang mở:"]
    for item in windows[:25]:
        active = " *active*" if item.get("is_active") else ""
        lines.append(f"{item['index']}. {item['title']} [{item['class_name']}]{active}")
    if len(windows) > 25:
        lines.append(f"... còn {len(windows) - 25} cửa sổ khác.")
    return ComputerUseResult(True, True, "\n".join(lines), "window_list", {"type": "window_list", "windows": windows})


def _activate_window(hwnd: int) -> bool:
    if os.name != "nt" or not hwnd:
        return False
    import ctypes

    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    try:
        from pywinauto import Application
        Application(backend="uia").connect(handle=hwnd).window(handle=hwnd).set_focus()
    except Exception:
        pass
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.2)
    return _active_hwnd() == hwnd


def _extract_window_query(command: str, plain: str) -> str:
    quoted = _extract_quoted_text(command)
    if quoted:
        return quoted
    return re.sub(
        r"^.*?(?:focus cua so|chuyen den cua so|kich hoat cua so|mo cua so|cua so)\s+",
        "",
        plain,
        flags=re.IGNORECASE,
    ).strip()


def _focus_window_result(query: str) -> ComputerUseResult:
    item = _find_window(query)
    if not item:
        return ComputerUseResult(True, False, f"Không tìm thấy cửa sổ: {query}", "window_focus", {"type": "window_focus", "query": query})
    ok = _activate_window(int(item["hwnd"]))
    status = "Đã focus" if ok else "Đã gửi lệnh focus"
    return ComputerUseResult(
        True,
        True,
        f"{status} cửa sổ: {item['title']}",
        "window_focus",
        {"type": "window_focus", "window": item, "active_confirmed": ok},
    )


def _current_window_action(action: str) -> ComputerUseResult:
    if os.name != "nt":
        return ComputerUseResult(True, False, "Điều khiển cửa sổ hiện chỉ hỗ trợ Windows.", "window_action")
    import ctypes

    hwnd = _active_hwnd()
    if not hwnd:
        return ComputerUseResult(True, False, "Không đọc được cửa sổ đang active.", "window_action")
    codes = {
        "minimize": 6,  # SW_MINIMIZE
        "maximize": 3,  # SW_MAXIMIZE
        "restore": 9,  # SW_RESTORE
    }
    labels = {
        "minimize": "thu nhỏ",
        "maximize": "phóng to",
        "restore": "khôi phục",
    }
    ctypes.windll.user32.ShowWindow(hwnd, codes[action])
    return ComputerUseResult(
        True,
        True,
        f"Đã {labels[action]} cửa sổ: {_window_text(hwnd) or hwnd}",
        f"window_{action}",
        {"type": "window_action", "action": action, "hwnd": hwnd},
    )


def _set_window_bounds(command: str, plain: str) -> ComputerUseResult:
    if os.name != "nt":
        return ComputerUseResult(True, False, "Điều khiển cửa sổ hiện chỉ hỗ trợ Windows.", "window_bounds")
    import ctypes

    hwnd = _active_hwnd()
    if not hwnd:
        return ComputerUseResult(True, False, "Không đọc được cửa sổ đang active.", "window_bounds")

    numbers = [int(value) for value in re.findall(r"-?\d{1,5}", plain)]
    left, top, right, bottom = _window_rect(hwnd)
    width = max(100, right - left)
    height = max(100, bottom - top)

    if "resize" in plain or "doi kich thuoc" in plain:
        if len(numbers) < 2:
            return ComputerUseResult(True, False, "Anh cần ghi kích thước, ví dụ `resize cửa sổ 1200 800`.", "window_bounds")
        width, height = max(100, numbers[-2]), max(100, numbers[-1])
    elif "di chuyen" in plain or "move window" in plain:
        if len(numbers) < 2:
            return ComputerUseResult(True, False, "Anh cần ghi tọa độ, ví dụ `di chuyển cửa sổ 100 80`.", "window_bounds")
        left, top = numbers[-2], numbers[-1]
    else:
        if len(numbers) < 4:
            return ComputerUseResult(True, False, "Anh dùng `đặt cửa sổ x y rộng cao`, ví dụ `đặt cửa sổ 40 40 1200 800`.", "window_bounds")
        left, top, width, height = numbers[-4], numbers[-3], max(100, numbers[-2]), max(100, numbers[-1])

    SWP_NOZORDER = 0x0004
    ctypes.windll.user32.SetWindowPos(hwnd, 0, int(left), int(top), int(width), int(height), SWP_NOZORDER)
    return ComputerUseResult(
        True,
        True,
        f"Đã đặt cửa sổ `{_window_text(hwnd) or hwnd}` tại x={left}, y={top}, rộng={width}, cao={height}.",
        "window_bounds",
        {"type": "window_bounds", "hwnd": hwnd, "left": left, "top": top, "width": width, "height": height},
    )


def _set_mouse_position(x: int, y: int) -> None:
    if os.name != "nt":
        raise RuntimeError("Computer-use hiện chỉ hỗ trợ Windows.")
    import ctypes

    width, height = _screen_size()
    if width and height:
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


def _mouse_click(button: str = "left", double: bool = False) -> None:
    if os.name != "nt":
        raise RuntimeError("Computer-use hiện chỉ hỗ trợ Windows.")
    import ctypes

    user32 = ctypes.windll.user32
    events = {
        "left": (0x0002, 0x0004),
        "right": (0x0008, 0x0010),
        "middle": (0x0020, 0x0040),
    }
    down, up = events.get(button, events["left"])
    count = 2 if double else 1
    for _ in range(count):
        user32.mouse_event(down, 0, 0, 0, 0)
        time.sleep(0.04)
        user32.mouse_event(up, 0, 0, 0, 0)
        time.sleep(0.08)


def _mouse_scroll(amount: int) -> None:
    if os.name != "nt":
        raise RuntimeError("Computer-use hiện chỉ hỗ trợ Windows.")
    import ctypes

    ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(amount), 0)


def _mouse_drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.35) -> None:
    if os.name != "nt":
        raise RuntimeError("Computer-use hiện chỉ hỗ trợ Windows.")
    import ctypes

    user32 = ctypes.windll.user32
    _set_mouse_position(start_x, start_y)
    time.sleep(0.08)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    steps = max(6, int(duration / 0.03))
    for index in range(1, steps + 1):
        x = start_x + (end_x - start_x) * index / steps
        y = start_y + (end_y - start_y) * index / steps
        _set_mouse_position(int(x), int(y))
        time.sleep(duration / steps)
    user32.mouse_event(0x0004, 0, 0, 0, 0)


def _active_window_title() -> str:
    if os.name != "nt":
        return ""
    return _window_text(_active_hwnd())


def _foreground_hwnd() -> int:
    return _active_hwnd()


def _ui_control_items(limit: int = 80, hwnd: int | None = None) -> list[tuple[dict, object]]:
    """Đọc control UI Automation của cửa sổ active, giữ cả metadata và object."""
    try:
        from pywinauto import Desktop
    except Exception as exc:
        raise RuntimeError(f"Thiếu pywinauto để đọc UI Automation: {exc}") from exc

    hwnd = hwnd or _foreground_hwnd()
    if not hwnd:
        return []

    window = Desktop(backend="uia").window(handle=hwnd)
    window_title = _window_text(hwnd)
    items = []
    for index, control in enumerate(window.descendants()[:limit], start=1):
        info = control.element_info
        rect = info.rectangle
        name = (info.name or "").strip()
        control_type = (info.control_type or "").strip()
        if not name and control_type in {"Pane", "Custom", "Group"}:
            continue
        item = {
            "index": len(items) + 1,
            "name": name,
            "control_type": control_type,
            "class_name": info.class_name or "",
            "hwnd": int(hwnd),
            "window_title": window_title,
            "rect": {
                "left": rect.left,
                "top": rect.top,
                "right": rect.right,
                "bottom": rect.bottom,
                "width": rect.width(),
                "height": rect.height(),
            },
        }
        items.append((item, control))
    return items


def _ui_elements(limit: int = 80) -> list[dict]:
    """Đọc cây UI Automation của cửa sổ active bằng pywinauto."""
    return [item for item, _control in _ui_control_items(limit=limit)]


def _ui_tree_result(limit: int = 80) -> ComputerUseResult:
    title = _active_window_title() or "không đọc được tên cửa sổ"
    elements = _ui_elements(limit=limit)
    lines = [f"UI của cửa sổ active: {title}", f"Đọc được {len(elements)} control:"]
    for item in elements[:25]:
        name = item["name"] or "(không tên)"
        rect = item["rect"]
        lines.append(
            f"{item['index']}. {item['control_type']} - {name} "
            f"({rect['left']},{rect['top']} - {rect['right']},{rect['bottom']})"
        )
    if len(elements) > 25:
        lines.append(f"... còn {len(elements) - 25} control khác.")
    return ComputerUseResult(
        True,
        True,
        "\n".join(lines),
        "ui_tree",
        {
            "type": "ui_tree",
            "active_window": title,
            "elements": elements,
        },
    )


def _find_ui_element(query: str, limit: int = 160, preferred_types: set[str] | None = None) -> tuple[dict, object] | None:
    plain_query = _plain_text(query)
    if not plain_query:
        return None

    def match_in_items(items: list[tuple[dict, object]]) -> tuple[dict, object] | None:
        local_fallback = None
        preferred_fallback = None
        for item, control in items:
            haystack = " ".join([
                item.get("name", ""),
                item.get("control_type", ""),
                item.get("class_name", ""),
            ])
            plain = _plain_text(haystack)
            if not plain:
                continue
            if plain_query == plain or plain_query == _plain_text(item.get("name", "")):
                return item, control
            if plain_query in plain and local_fallback is None:
                local_fallback = (item, control)
            if preferred_types and plain_query in plain and item.get("control_type") in preferred_types and preferred_fallback is None:
                preferred_fallback = (item, control)
        return preferred_fallback or local_fallback

    active_hwnd = _foreground_hwnd()
    active_match = match_in_items(_ui_control_items(limit=limit, hwnd=active_hwnd))
    if active_match:
        return active_match

    seen = {active_hwnd}
    for window in _visible_windows(limit=40):
        hwnd = int(window.get("hwnd") or 0)
        if not hwnd or hwnd in seen:
            continue
        seen.add(hwnd)
        try:
            match = match_in_items(_ui_control_items(limit=limit, hwnd=hwnd))
        except Exception:
            continue
        if match:
            return match
    return None


def _extract_ui_query(command: str, plain: str) -> str:
    quoted = _extract_quoted_text(command)
    if quoted:
        return quoted
    return re.sub(
        r"^.*?(?:click nut|click text|bam nut|focus|clear text|xoa text|doc control|doc text|thong tin control|chọn|chon)\s+",
        "",
        plain,
        flags=re.IGNORECASE,
    ).strip()


def _click_ui_element(query: str) -> ComputerUseResult:
    found = _find_ui_element(query)
    if not found:
        return ComputerUseResult(True, False, f"Không tìm thấy control/text: {query}", "ui_click")
    item, control = found
    control.click_input()
    return ComputerUseResult(
        True,
        True,
        f"Đã click UI: {item['control_type']} - {item['name'] or query}",
        "ui_click",
        {"type": "ui_element", "element": item},
    )


def _ui_element_details(item: dict, control) -> dict:
    details = dict(item)
    values = {}
    for attr in ("window_text", "texts", "legacy_properties", "get_value"):
        try:
            value = getattr(control, attr)()
            values[attr] = value
        except Exception:
            pass
    if values:
        details["values"] = values
    return details


def _click_ui_index(index: int, button: str = "left", double: bool = False) -> ComputerUseResult:
    items = _ui_control_items(limit=160)
    if index < 1 or index > len(items):
        return ComputerUseResult(True, False, f"Không có control UI số {index}. Anh gửi `xem UI` để lấy danh sách mới.", "ui_click_index")
    item, control = items[index - 1]
    control.click_input(button=button, double=double)
    click_label = f"{'double ' if double else ''}{button}"
    return ComputerUseResult(
        True,
        True,
        f"Đã {click_label} click UI số {index}: {item['control_type']} - {item['name'] or '(không tên)'}",
        "ui_click_index",
        {"type": "ui_element", "element": item},
    )


def _read_ui_element(query: str) -> ComputerUseResult:
    found = _find_ui_element(query)
    if not found:
        return ComputerUseResult(True, False, f"Không tìm thấy control/text: {query}", "ui_read")
    item, control = found
    details = _ui_element_details(item, control)
    lines = [
        f"Control: {item.get('control_type') or ''} - {item.get('name') or '(không tên)'}",
        f"Class: {item.get('class_name') or '(trống)'}",
    ]
    rect = item.get("rect") or {}
    if rect:
        lines.append(f"Vị trí: {rect.get('left')},{rect.get('top')} - {rect.get('right')},{rect.get('bottom')}")
    values = details.get("values") or {}
    for key, value in values.items():
        text = str(value)
        if len(text) > 400:
            text = text[:400] + "... (đã cắt bớt)"
        lines.append(f"{key}: {text}")
    return ComputerUseResult(True, True, "\n".join(lines), "ui_read", {"type": "ui_element", "element": details})


def _extract_text_target(command: str, plain: str) -> tuple[str, str]:
    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', command)
    quoted_values = [next(group for group in groups if group) for groups in quoted]
    if len(quoted_values) >= 2:
        return quoted_values[0], quoted_values[1]
    if len(quoted_values) == 1:
        text = quoted_values[0]
        target = re.sub(r"^.*?(?:vao|vào|to|ô|o)\s+", "", plain, flags=re.IGNORECASE).strip()
        return text, target

    match = re.search(r"(?:go|nhap|dien|set text)\s+(.+?)\s+(?:vao|vào|to)\s+(.+)$", command, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def _set_ui_text(command: str, plain: str) -> ComputerUseResult:
    text, target = _extract_text_target(command, plain)
    if not text or not target:
        return ComputerUseResult(True, False, 'Anh dùng dạng `nhập "nội dung" vào "tên ô"`.', "ui_set_text")
    found = _find_ui_element(target, preferred_types={"Edit", "ComboBox", "Document"})
    if not found:
        return ComputerUseResult(True, False, f"Không tìm thấy ô/control: {target}", "ui_set_text")
    item, control = found
    try:
        control.set_focus()
    except Exception:
        control.click_input()
    try:
        control.set_edit_text(text)
        method = "set_edit_text"
    except Exception:
        _press_hotkey(["ctrl", "a"])
        _send_unicode_text(text)
        method = "keyboard"
    return ComputerUseResult(
        True,
        True,
        f"Đã nhập vào UI `{item['name'] or target}` bằng {method}: {text}",
        "ui_set_text",
        {"type": "ui_element", "element": item, "length": len(text), "method": method},
    )


def _clear_ui_text(query: str) -> ComputerUseResult:
    found = _find_ui_element(query, preferred_types={"Edit", "ComboBox", "Document"})
    if not found:
        return ComputerUseResult(True, False, f"Không tìm thấy ô/control để xóa text: {query}", "ui_clear_text")
    item, control = found
    try:
        control.set_edit_text("")
        method = "set_edit_text"
    except Exception:
        try:
            control.set_focus()
        except Exception:
            control.click_input()
        _press_hotkey(["ctrl", "a"])
        _press_hotkey(["backspace"])
        method = "keyboard"
    return ComputerUseResult(True, True, f"Đã xóa text trong UI: {item['name'] or query}", "ui_clear_text", {"type": "ui_element", "element": item, "method": method})


def _focus_ui_element(query: str) -> ComputerUseResult:
    found = _find_ui_element(query)
    if not found:
        return ComputerUseResult(True, False, f"Không tìm thấy control/text để focus: {query}", "ui_focus")
    item, control = found
    try:
        control.set_focus()
    except Exception:
        control.click_input()
    return ComputerUseResult(
        True,
        True,
        f"Đã focus UI: {item['control_type']} - {item['name'] or query}",
        "ui_focus",
        {"type": "ui_element", "element": item},
    )


def _send_unicode_text(text: str) -> None:
    if os.name != "nt":
        raise RuntimeError("Computer-use hiện chỉ hỗ trợ Windows.")
    import ctypes

    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", ctypes.c_ulong),
            ("wParamL", ctypes.c_ushort),
            ("wParamH", ctypes.c_ushort),
        ]

    class INPUTUNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("union", INPUTUNION)]

    ctypes.windll.user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
    ctypes.windll.user32.SendInput.restype = ctypes.c_uint

    def send_unit(unit: int, flags: int) -> None:
        item = INPUT(type=1, union=INPUTUNION(ki=KEYBDINPUT(0, unit, flags, 0, 0)))
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(item))
        if sent != 1:
            raise RuntimeError("Windows không nhận phím Unicode.")

    units = text.encode("utf-16-le")
    for i in range(0, len(units), 2):
        unit = int.from_bytes(units[i:i + 2], "little")
        send_unit(unit, 0x0004)
        send_unit(unit, 0x0004 | 0x0002)
        time.sleep(0.004)


_VK = {
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B,
    "windows": 0x5B,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "len": 0x26,
    "xuong": 0x28,
    "trai": 0x25,
    "phai": 0x27,
}
for i in range(1, 13):
    _VK[f"f{i}"] = 0x6F + i
for c in "abcdefghijklmnopqrstuvwxyz":
    _VK[c] = ord(c.upper())
for n in "0123456789":
    _VK[n] = ord(n)


def _press_hotkey(keys: list[str]) -> None:
    if os.name != "nt":
        raise RuntimeError("Computer-use hiện chỉ hỗ trợ Windows.")
    import ctypes

    user32 = ctypes.windll.user32
    parsed = []
    for key in keys:
        clean = key.strip().lower()
        if not clean:
            continue
        vk = _VK.get(clean)
        if vk is None:
            raise ValueError(f"Chưa hỗ trợ phím: {key}")
        parsed.append(vk)

    if not parsed:
        raise ValueError("Chưa thấy phím cần bấm.")

    for vk in parsed:
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.03)
    for vk in reversed(parsed):
        user32.keybd_event(vk, 0, 0x0002, 0)
        time.sleep(0.03)


def _open_clipboard_with_retry(user32, attempts: int = 12) -> None:
    for _ in range(attempts):
        if user32.OpenClipboard(None):
            return
        time.sleep(0.05)
    raise RuntimeError("Không mở được clipboard.")


def _read_clipboard_text() -> str:
    if os.name != "nt":
        raise RuntimeError("Clipboard hiện chỉ hỗ trợ Windows.")
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    CF_UNICODETEXT = 13
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    _open_clipboard_with_retry(user32)
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return ""
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _write_clipboard_text(text: str) -> None:
    if os.name != "nt":
        raise RuntimeError("Clipboard hiện chỉ hỗ trợ Windows.")
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    data = (text + "\0").encode("utf-16-le")
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    if not handle:
        raise RuntimeError("Không cấp phát được bộ nhớ clipboard.")
    ptr = kernel32.GlobalLock(handle)
    if not ptr:
        kernel32.GlobalFree(handle)
        raise RuntimeError("Không khóa được bộ nhớ clipboard.")
    try:
        ctypes.memmove(ptr, data, len(data))
    finally:
        kernel32.GlobalUnlock(handle)

    try:
        _open_clipboard_with_retry(user32)
    except Exception:
        kernel32.GlobalFree(handle)
        raise
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            raise RuntimeError("Không ghi được clipboard.")
        handle = None
    finally:
        user32.CloseClipboard()


def _clipboard_result(command: str, plain: str) -> ComputerUseResult | None:
    if "clipboard" not in plain and "bo nho tam" not in plain:
        return None

    wants_read = (
        "xem clipboard" in plain
        or "doc clipboard" in plain
        or "noi dung clipboard" in plain
        or plain in {"clipboard", "bo nho tam"}
    )
    wants_write = (
        "dat clipboard" in plain
        or "set clipboard" in plain
        or "ghi clipboard" in plain
        or "luu clipboard" in plain
        or "copy vao clipboard" in plain
    )

    if wants_read and not wants_write:
        text = _read_clipboard_text()
        preview = text if len(text) <= 1200 else text[:1200] + "\n... (đã cắt bớt)"
        return ComputerUseResult(
            True,
            True,
            f"Clipboard hiện có:\n{preview or '(trống)'}",
            "clipboard_read",
            {"type": "clipboard", "text": text, "length": len(text)},
        )

    if wants_write:
        text = _extract_quoted_text(command)
        if text is None:
            match = re.search(r"clipboard\s*[:,-]?\s*(.+)$", command, flags=re.IGNORECASE | re.DOTALL)
            text = match.group(1).strip() if match else ""
        if not text:
            return ComputerUseResult(True, False, "Anh cần ghi nội dung muốn đặt vào clipboard.", "clipboard_write")
        _write_clipboard_text(text)
        return ComputerUseResult(
            True,
            True,
            f"Đã đặt clipboard ({len(text)} ký tự).",
            "clipboard_write",
            {"type": "clipboard", "length": len(text)},
        )

    return ComputerUseResult(True, False, "Anh dùng `xem clipboard` hoặc `đặt clipboard \"nội dung\"`.", "clipboard")


def _extract_quoted_text(text: str) -> str | None:
    match = re.search(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', text)
    if not match:
        return None
    return next(group for group in match.groups() if group)


def _extract_all_quoted_text(text: str) -> list[str]:
    matches = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', text)
    return [next(group for group in groups if group) for groups in matches]


def _workspace_root() -> Path:
    return _REPO_ROOT


def _workspace_roots() -> list[Path]:
    config = _minion_config()
    roots = config.get("allowed_workspace_roots") or ["."]
    resolved: list[Path] = []
    for item in roots:
        try:
            path = Path(str(item)).expanduser()
            if not path.is_absolute():
                path = _REPO_ROOT / path
            path = path.resolve()
        except Exception:
            continue
        if path not in resolved:
            resolved.append(path)
    if _REPO_ROOT not in resolved:
        resolved.insert(0, _REPO_ROOT)
    return resolved


def _path_is_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve_workspace_path(target: str) -> Path:
    if not target:
        raise ValueError("Thiếu đường dẫn file.")
    root = _workspace_root()
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    allowed_roots = _workspace_roots()
    if not any(_path_is_inside(path, allowed_root) for allowed_root in allowed_roots):
        raise ValueError("Đường dẫn nằm ngoài workspace Minion, không xử lý.")
    return path


def _trim_output(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (đã cắt bớt output)"


def _command_timeout(default: int = 20) -> int:
    try:
        return max(3, min(int(_minion_config().get("command_timeout_seconds", default)), 120))
    except Exception:
        return default


def _backup_file(path: Path, original_text: str) -> Path:
    backup_dir = _REPO_ROOT / "output" / "minion-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = backup_dir / f"{path.name}.{digest}.{stamp}.bak"
    backup.write_text(original_text, encoding="utf-8")
    return backup


def _workspace_replace_preview(path: Path, old: str, new: str) -> tuple[str, int, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        return text, 0, ""
    updated = text.replace(old, new)
    diff = "\n".join(difflib.unified_diff(
        text.splitlines(),
        updated.splitlines(),
        fromfile=str(path),
        tofile=str(path) + " (minion)",
        lineterm="",
    ))
    return updated, count, _trim_output(diff, 12000)


def _workspace_patch_key(path: Path, old: str, new: str) -> str:
    digest = hashlib.sha256((str(path) + "\0" + old + "\0" + new).encode("utf-8")).hexdigest()
    return f"workspace_patch:{digest}"


def _workspace_status() -> ComputerUseResult:
    root = _workspace_root()
    lines = [f"Workspace: {root}"]
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        status = completed.stdout.strip() or "(git clean)"
        lines.append("Git status:")
        lines.append(_trim_output(status, 2000))
    except Exception as exc:
        lines.append(f"Không đọc được git status: {exc}")
    return ComputerUseResult(True, True, "\n".join(lines), "workspace_status", {"type": "workspace", "root": str(root)})


def _workspace_list_files(command: str, plain: str) -> ComputerUseResult:
    root = _workspace_root()
    quoted = _extract_quoted_text(command)
    filter_text = quoted or re.sub(r"^.*?(?:liet ke file|list files|tim file|find file)\s*", "", plain, flags=re.IGNORECASE).strip()
    try:
        completed = subprocess.run(
            ["rg", "--files"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        files = completed.stdout.splitlines()
    except Exception:
        files = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]
    if filter_text:
        plain_filter = _plain_text(filter_text)
        files = [name for name in files if plain_filter in _plain_text(name)]
    files = sorted(files)[:200]
    body = "\n".join(files) if files else "(không có file khớp)"
    return ComputerUseResult(True, True, f"File trong workspace:\n{body}", "workspace_files", {"type": "workspace", "files": files})


def _workspace_search_code(command: str, plain: str) -> ComputerUseResult:
    pattern = _extract_quoted_text(command)
    if not pattern:
        pattern = re.sub(r"^.*?(?:tim code|search code)\s+", "", command, flags=re.IGNORECASE).strip()
    if not pattern:
        return ComputerUseResult(True, False, 'Anh dùng `tìm code "nội dung cần tìm"`.', "workspace_search")
    root = _workspace_root()
    try:
        completed = subprocess.run(
            ["rg", "-n", "--hidden", "-S", pattern],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        output = completed.stdout.strip()
        if completed.returncode == 1 and not output:
            output = "(không có kết quả)"
        elif completed.returncode not in (0, 1):
            output = (completed.stderr or "").strip() or f"rg exit {completed.returncode}"
    except Exception as exc:
        output = f"Lỗi tìm code: {exc}"
    return ComputerUseResult(True, True, f"Kết quả tìm code `{pattern}`:\n{_trim_output(output, 6000)}", "workspace_search", {"type": "workspace", "pattern": pattern})


def _workspace_read_file(command: str, plain: str) -> ComputerUseResult:
    target = _extract_quoted_text(command)
    if not target:
        target = re.sub(r"^.*?(?:doc file|read file)\s+", "", command, flags=re.IGNORECASE).strip()
    try:
        path = _resolve_workspace_path(target)
        if not path.exists() or not path.is_file():
            return ComputerUseResult(True, False, f"Không tìm thấy file: {path}", "workspace_read")
        if path.stat().st_size > 2_000_000:
            return ComputerUseResult(True, False, f"File quá lớn để đọc trực tiếp: {path}", "workspace_read")
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return ComputerUseResult(True, False, f"Không đọc được file: {exc}", "workspace_read")
    return ComputerUseResult(True, True, f"{path}:\n{_trim_output(text, 8000)}", "workspace_read", {"type": "workspace", "path": str(path), "length": len(text)})


def _workspace_replace_in_file(command: str, plain: str) -> ComputerUseResult:
    values = _extract_all_quoted_text(command)
    if len(values) < 3:
        return ComputerUseResult(True, False, 'Anh dùng `thay trong file "path" "text cũ" "text mới"`.', "workspace_replace")
    target, old, new = values[0], values[1], values[2]
    try:
        path = _resolve_workspace_path(target)
        if not path.exists() or not path.is_file():
            return ComputerUseResult(True, False, f"Không tìm thấy file: {path}", "workspace_replace")
        if path.stat().st_size > 2_000_000:
            return ComputerUseResult(True, False, f"File quá lớn để sửa bằng exact replace: {path}", "workspace_replace")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ComputerUseResult(True, False, "Không tìm thấy text cũ trong file, chưa sửa gì.", "workspace_replace")
        backup = _backup_file(path, text)
        path.write_text(text.replace(old, new), encoding="utf-8")
    except Exception as exc:
        return ComputerUseResult(True, False, f"Không sửa được file: {exc}", "workspace_replace")
    return ComputerUseResult(
        True,
        True,
        f"Đã thay {count} chỗ trong file: {path}\nBackup: {backup}",
        "workspace_replace",
        {"type": "workspace", "path": str(path), "count": count, "backup": str(backup)},
        risk_level="needs_approval",
    )


def _workspace_result(command: str, plain: str) -> ComputerUseResult | None:
    if "liet ke file" in plain or "list files" in plain or "tim file" in plain or "find file" in plain:
        return _workspace_list_files(command, plain)
    if "tim code" in plain or "search code" in plain:
        return _workspace_search_code(command, plain)
    if "doc file" in plain or "read file" in plain:
        return _workspace_read_file(command, plain)
    if "thay trong file" in plain or "replace in file" in plain:
        return _workspace_replace_in_file(command, plain)
    if "workspace" in plain or "git status" in plain or "trang thai workspace" in plain:
        return _workspace_status()
    return None


def workspace_status_result() -> ComputerUseResult:
    return _workspace_status()


def workspace_list_files_result(query: str = "") -> ComputerUseResult:
    command = f'liet ke file "{query}"' if query else "liet ke file"
    return _workspace_list_files(command, _plain_text(command))


def workspace_search_result(pattern: str) -> ComputerUseResult:
    command = f'tim code "{pattern}"'
    return _workspace_search_code(command, _plain_text(command))


def workspace_read_result(path: str) -> ComputerUseResult:
    command = f'doc file "{path}"'
    return _workspace_read_file(command, _plain_text(command))


def workspace_patch_result(
    path_text: str,
    old: str,
    new: str,
    apply: bool = False,
    approval_token: str | None = None,
) -> ComputerUseResult:
    try:
        path = _resolve_workspace_path(path_text)
        if not path.exists() or not path.is_file():
            return ComputerUseResult(True, False, f"Không tìm thấy file: {path}", "workspace_patch")
        if path.stat().st_size > 2_000_000:
            return ComputerUseResult(True, False, f"File quá lớn để patch trực tiếp: {path}", "workspace_patch")
        updated, count, diff = _workspace_replace_preview(path, old, new)
        if count == 0:
            return ComputerUseResult(True, False, "Không tìm thấy text cũ trong file, chưa sửa gì.", "workspace_patch")
    except Exception as exc:
        return ComputerUseResult(True, False, f"Không chuẩn bị patch được: {exc}", "workspace_patch")

    data = {
        "type": "workspace_diff",
        "path": str(path),
        "count": count,
        "diff": diff,
        "apply": apply,
    }
    if not apply:
        return ComputerUseResult(
            True,
            True,
            f"Diff preview cho {path} ({count} chỗ):\n{diff}",
            "workspace_diff",
            data,
        )

    command_key = _workspace_patch_key(path, old, new)
    if not _consume_approval(command_key, approval_token):
        result = _approval_result(command_key, "workspace_patch", "needs_approval", "áp dụng patch sửa file trong workspace")
        result.data.update({"path": str(path), "diff": diff, "count": count})
        return result

    try:
        original = path.read_text(encoding="utf-8", errors="replace")
        backup = _backup_file(path, original)
        path.write_text(updated, encoding="utf-8")
    except Exception as exc:
        return ComputerUseResult(True, False, f"Không áp dụng patch được: {exc}", "workspace_patch", data)

    data["backup"] = str(backup)
    return ComputerUseResult(
        True,
        True,
        f"Đã áp dụng patch {count} chỗ trong file: {path}\nBackup: {backup}",
        "workspace_patch",
        data,
        risk_level="needs_approval",
    )


def workspace_run_command_result(command: str, approval_token: str | None = None) -> ComputerUseResult:
    command = (command or "").strip()
    if not command:
        return ComputerUseResult(True, False, "Thiếu lệnh cần chạy.", "workspace_run")
    if _dangerous_shell_text(command):
        return ComputerUseResult(
            True,
            False,
            "Lệnh thuộc nhóm nguy hiểm, Minion không tự chạy.",
            "workspace_run",
            {"type": "workspace_run", "command": command},
            risk_level="blocked",
        )

    command_key = "workspace_run:" + hashlib.sha256(command.encode("utf-8")).hexdigest()
    if not _consume_approval(command_key, approval_token):
        result = _approval_result(command_key, "workspace_run", "needs_approval", "chạy lệnh trong workspace")
        result.data.update({"command": command})
        return result

    root = _workspace_root()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_command_timeout(),
        )
        output = (completed.stdout or "").strip()
        error = (completed.stderr or "").strip()
        combined = "\n".join(part for part in [output, error] if part).strip() or "(không có output)"
        combined = _trim_output(combined, 12000)
        ok = completed.returncode == 0
        return ComputerUseResult(
            True,
            ok,
            f"Đã chạy lệnh trong workspace, exit code {completed.returncode}:\n{combined}",
            "workspace_run",
            {
                "type": "workspace_run",
                "command": command,
                "cwd": str(root),
                "returncode": completed.returncode,
                "output": combined,
            },
            risk_level="needs_approval",
        )
    except subprocess.TimeoutExpired:
        return ComputerUseResult(
            True,
            False,
            f"Lệnh quá thời gian {_command_timeout()} giây, Minion đã dừng chờ.",
            "workspace_run",
            {"type": "workspace_run", "command": command, "cwd": str(root), "timeout": _command_timeout()},
            risk_level="needs_approval",
        )
    except Exception as exc:
        return ComputerUseResult(True, False, f"Không chạy được lệnh: {exc}", "workspace_run", {"command": command})


def _open_file_or_folder(command: str) -> ComputerUseResult | None:
    target = _extract_quoted_text(command)
    if not target:
        match = re.search(r"([A-Za-z]:\\[^\"']+)$", command.strip())
        target = match.group(1).strip() if match else ""
    if not target:
        return None
    if re.match(r"https?://", target, flags=re.IGNORECASE):
        return None
    if not (re.match(r"^[A-Za-z]:\\", target) or "\\" in target or "/" in target or target.startswith(("~", "."))):
        return None

    path = Path(target).expanduser()
    if not path.exists():
        return ComputerUseResult(True, False, f"Không tìm thấy file/thư mục: {path}", "open_path")
    os.startfile(str(path))
    return ComputerUseResult(True, True, f"Đã mở: {path}", "open_path", {"path": str(path)})


def _run_shell_command(command: str, plain: str) -> ComputerUseResult | None:
    if "chay lenh" not in plain and "run command" not in plain:
        return None

    shell_text = _extract_quoted_text(command)
    if not shell_text:
        shell_text = re.sub(r"^.*?(chạy lệnh|chay lenh|run command)\s+", "", command, flags=re.IGNORECASE).strip()
    if not shell_text:
        return ComputerUseResult(True, False, "Anh cần ghi rõ lệnh muốn chạy, ví dụ: `chạy lệnh \"dir\"`.", "shell")

    completed = subprocess.run(
        shell_text,
        shell=True,
        capture_output=True,
        text=True,
        timeout=_command_timeout(),
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    combined = "\n".join(part for part in [output, error] if part).strip()
    if len(combined) > 1400:
        combined = combined[:1400] + "\n... (đã cắt bớt output)"
    if not combined:
        combined = "(không có output)"
    ok = completed.returncode == 0
    return ComputerUseResult(
        True,
        ok,
        f"Đã chạy lệnh, exit code {completed.returncode}:\n{combined}",
        "shell",
        {"returncode": completed.returncode},
        risk_level="needs_approval",
    )


def _open_target(command: str, plain: str) -> ComputerUseResult:
    path_result = _open_file_or_folder(command)
    if path_result is not None:
        return path_result

    url_match = re.search(r"https?://\S+", command)
    if url_match:
        url = url_match.group(0).rstrip(".,)")
        webbrowser.open(url)
        return ComputerUseResult(True, True, f"Đã mở link: {url}", "open_url")

    search_match = re.search(r"(?:tim kiem|tim google|search|google)\s+(.+)$", plain)
    if search_match and search_match.group(1).strip() not in {"google", "youtube"}:
        query = search_match.group(1).strip()
        url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
        webbrowser.open(url)
        return ComputerUseResult(True, True, f"Đã tìm Google: {query}", "search", {"query": query, "url": url})

    youtube_match = re.search(r"(?:tim youtube|youtube)\s+(.+)$", plain)
    if youtube_match and youtube_match.group(1).strip() != "youtube":
        query = youtube_match.group(1).strip()
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
        webbrowser.open(url)
        return ComputerUseResult(True, True, f"Đã tìm YouTube: {query}", "search", {"query": query, "url": url})

    app_map = {
        "notepad": "notepad.exe",
        "ghi chu": "notepad.exe",
        "may tinh": "calc.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "chrome": "chrome.exe",
        "edge": "msedge.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
    }
    for key, exe in app_map.items():
        if key in plain:
            subprocess.Popen([exe], shell=False)
            return ComputerUseResult(True, True, f"Đã mở {key}.", "open_app")

    site_map = {
        "google": "https://www.google.com",
        "youtube": "https://www.youtube.com",
        "zalo": "https://chat.zalo.me",
        "gmail": "https://mail.google.com",
        "calendar": "https://calendar.google.com",
    }
    for key, url in site_map.items():
        if key in plain:
            webbrowser.open(url)
            return ComputerUseResult(True, True, f"Đã mở {key}: {url}", "open_url")

    return ComputerUseResult(True, False, "Anh muốn mở app/link nào? Ví dụ: `mở notepad`, `mở https://google.com`.", "open")


def _screenshot() -> ComputerUseResult:
    try:
        from PIL import ImageGrab
    except Exception:
        return ComputerUseResult(
            True,
            False,
            "Chưa chụp màn hình được vì thiếu Pillow. Chạy `pip install pillow` rồi thử lại.",
            "screenshot",
        )

    screenshot_dir = str(_minion_config().get("screenshot_dir") or "output/computer-use")
    out_dir = Path(screenshot_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"screenshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    image = ImageGrab.grab()
    image.save(path)
    width, height = image.size
    filename = path.name
    artifact = {
        "type": "screenshot",
        "url": f"/api/computer-use/files/{filename}",
        "path": str(path.resolve()),
        "width": width,
        "height": height,
    }
    return ComputerUseResult(
        True,
        True,
        f"Đã chụp màn hình {width}x{height}. Anh có thể dùng tọa độ trên ảnh để bảo Minion click hoặc kéo chuột.",
        "screenshot",
        artifact,
        artifacts=[artifact],
    )


def get_computer_state() -> dict:
    """Trạng thái máy hiện tại phục vụ UI/debug computer-use."""
    x, y = _mouse_position()
    width, height = _screen_size()
    return {
        "cursor": {"x": x, "y": y},
        "screen": {"width": width, "height": height},
        "active_window": _active_window_title(),
    }


def _split_sequence(command: str) -> list[str]:
    """Tách chuỗi thao tác đơn giản, tránh tách khi không có dấu hiệu nhiều bước."""
    if not re.search(r"[;\n]|(\s+(rồi|roi|sau đó|sau do)\s+)", command, flags=re.IGNORECASE):
        return [command]
    parts = re.split(r"[;\n]+|\s+(?:rồi|roi|sau đó|sau do)\s+", command, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _strip_agent_prefix(command: str) -> str:
    return re.sub(
        r"^\s*(?:minion\s+)?(?:tự\s+làm|tu\s+lam|auto\s+làm|auto\s+lam|agent|cowork)\s*[:,-]?\s*",
        "",
        command,
        flags=re.IGNORECASE,
    ).strip()


def _task_steps(task: str) -> list[str]:
    """Lập các bước rõ ràng từ task người dùng nhập."""
    cleaned = _strip_agent_prefix(task)
    parts = _split_sequence(cleaned)
    if len(parts) > 1:
        return parts[:8]

    plain = _plain_text(cleaned)
    # Pattern phổ biến: mở app/link rồi gõ nội dung vào đó.
    match = re.search(r"^(mo\s+.+?)\s+(?:va|roi|sau do)\s+(go|nhap)\s+(.+)$", plain)
    if match:
        original_match = re.search(
            r"^(mở\s+.+?)\s+(?:và|rồi|sau đó)\s+(gõ|nhập)\s+(.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if original_match:
            return [original_match.group(1), f"{original_match.group(2)} {original_match.group(3)}"]
    return [cleaned]


def plan_agent_steps(task: str, max_steps: int | None = None) -> list[str]:
    configured_max = int(_minion_config().get("max_agent_steps", 8) or 8)
    max_steps = max(1, min(int(max_steps or configured_max), 20))
    return [step for step in _task_steps(task) if step][:max_steps]


def screenshot_result() -> ComputerUseResult:
    return _screenshot()


def run_agent_task(
    task: str,
    enabled: bool,
    max_steps: int | None = None,
    approval_token: str | None = None,
    _approved: bool = False,
) -> ComputerUseResult | None:
    """Chạy task nhiều bước có observe -> act -> verify ở mức local."""
    plain = _plain_text(task)
    if not re.match(r"^(?:minion\s+)?(?:tu lam|auto lam|agent|cowork)\b", plain):
        return None

    if not enabled:
        return ComputerUseResult(
            True,
            False,
            "Computer Use đang tắt. Anh bật nút `Điều khiển máy` rồi gửi lại task.",
            "agent_disabled",
        )

    steps = plan_agent_steps(task, max_steps)
    if not steps:
        return ComputerUseResult(True, False, "Anh cần ghi rõ task muốn Minion tự làm.", "agent_run")

    logs = []
    ok = True
    observation = get_computer_state()
    for index, step in enumerate(steps, start=1):
        result = execute_computer_command(step, True, _depth=1, approval_token=approval_token, _approved=_approved)
        if result is None:
            result = ComputerUseResult(True, False, f"Chưa hiểu bước: {step}", "unknown")
        step_shot = _screenshot()
        logs.append({
            "index": index,
            "instruction": step,
            "ok": result.ok,
            "action": result.action,
            "message": result.message,
            "risk_level": result.risk_level,
            "needs_approval": result.needs_approval,
            "approval_id": result.approval_id,
            "data": result.data,
            "screenshot": step_shot.data if step_shot.ok else None,
        })
        ok = ok and result.ok
        if result.needs_approval or result.risk_level == "blocked":
            break
        time.sleep(0.4)

    shot = _screenshot()
    status = "completed" if ok else "needs_attention"
    summary_lines = [
        "Minion đã chạy task nhiều bước:",
        *[f"{item['index']}. {item['instruction']} -> {item['message']}" for item in logs],
        "Đã chụp lại màn hình cuối để anh kiểm tra.",
    ]
    return ComputerUseResult(
        True,
        ok,
        "\n".join(summary_lines),
        "agent_run",
        {
            "type": "agent_run",
            "status": status,
            "ok": ok,
            "observation": observation,
            "plan": steps,
            "steps": logs,
            "screenshot": shot.data,
        },
    )


def execute_computer_command(
    command: str,
    enabled: bool,
    _depth: int = 0,
    approval_token: str | None = None,
    _approved: bool = False,
) -> ComputerUseResult | None:
    """Parse và chạy lệnh computer-use đơn giản từ câu tiếng Việt."""
    plain = _plain_text(command)
    if not _looks_like_computer_command(plain):
        return None

    if not enabled:
        return ComputerUseResult(
            True,
            False,
            "Computer Use đang tắt. Anh bật nút `Điều khiển máy` rồi gửi lại lệnh.",
            "disabled",
        )

    try:
        action_hint, risk_level, reason = _classify_command_risk(command, plain)
        command_key = _approval_key(command)
        approved = _approved or _consume_approval(command_key, approval_token)
        if risk_level == "blocked":
            return ComputerUseResult(
                True,
                False,
                reason,
                action_hint or "blocked",
                {"type": "risk_block", "command": command, "reason": reason},
                risk_level="blocked",
            )
        if risk_level == "needs_approval" and not approved:
            return _approval_result(command, action_hint, risk_level, reason)

        if _depth == 0:
            agent_result = run_agent_task(command, enabled, approval_token=approval_token, _approved=approved)
            if agent_result is not None:
                return agent_result

        parts = _split_sequence(command)
        if _depth == 0 and len(parts) > 1:
            messages = []
            overall_ok = True
            actions = []
            for index, part in enumerate(parts, start=1):
                result = execute_computer_command(part, True, _depth=1, _approved=approved)
                if result is None:
                    result = ComputerUseResult(True, False, f"Chưa hiểu bước {index}: {part}", "unknown")
                messages.append(f"{index}. {result.message}")
                actions.append(result.action)
                overall_ok = overall_ok and result.ok
                if result.needs_approval or result.risk_level == "blocked":
                    break
                time.sleep(0.35)
            return ComputerUseResult(True, overall_ok, "\n".join(messages), "sequence", {"actions": actions})

        wait_match = re.search(r"doi\s+([0-9]{1,2})(?:\s*(?:giay|s|second))?", plain)
        if wait_match:
            seconds = max(0, min(int(wait_match.group(1)), 30))
            time.sleep(seconds)
            return ComputerUseResult(True, True, f"Đã đợi {seconds} giây.", "wait", {"seconds": seconds})

        workspace_result = _workspace_result(command, plain)
        if workspace_result is not None:
            return workspace_result

        shell_result = _run_shell_command(command, plain)
        if shell_result is not None:
            return shell_result

        clipboard_result = _clipboard_result(command, plain)
        if clipboard_result is not None:
            return clipboard_result

        if (
            "danh sach cua so" in plain
            or "liet ke cua so" in plain
            or "xem cac cua so" in plain
            or plain == "xem cua so"
        ):
            return _window_list_result()

        if (
            "focus cua so" in plain
            or "chuyen den cua so" in plain
            or "kich hoat cua so" in plain
            or re.match(r"^(?:hay\s+)?(?:minion\s+)?mo cua so\b", plain)
        ):
            query = _extract_window_query(command, plain)
            return _focus_window_result(query)

        if "thu nho cua so" in plain or "minimize window" in plain:
            return _current_window_action("minimize")

        if "phong to cua so" in plain or "maximize window" in plain:
            return _current_window_action("maximize")

        if "khoi phuc cua so" in plain or "restore window" in plain:
            return _current_window_action("restore")

        if (
            "dat cua so" in plain
            or "resize cua so" in plain
            or "doi kich thuoc cua so" in plain
            or "di chuyen cua so" in plain
            or "resize window" in plain
            or "move window" in plain
        ):
            return _set_window_bounds(command, plain)

        if "dong cua so" in plain or "close window" in plain:
            _press_hotkey(["alt", "f4"])
            return ComputerUseResult(True, True, "Đã gửi phím đóng cửa sổ: Alt+F4.", "close_window")

        if "chuyen cua so" in plain or "alt tab" in plain:
            _press_hotkey(["alt", "tab"])
            return ComputerUseResult(True, True, "Đã chuyển cửa sổ: Alt+Tab.", "switch_window")

        if "chon tat ca" in plain or "select all" in plain:
            _press_hotkey(["ctrl", "a"])
            return ComputerUseResult(True, True, "Đã chọn tất cả: Ctrl+A.", "select_all")

        if "sao chep" in plain or plain == "copy" or " copy " in f" {plain} ":
            _press_hotkey(["ctrl", "c"])
            return ComputerUseResult(True, True, "Đã copy: Ctrl+C.", "copy")

        if re.match(r"^(?:hay\s+)?(?:minion\s+)?(?:cat|cut)\b", plain):
            _press_hotkey(["ctrl", "x"])
            return ComputerUseResult(True, True, "Đã cắt: Ctrl+X.", "cut")

        if re.match(r"^(?:hay\s+)?(?:minion\s+)?(?:paste|dan)\b", plain):
            _press_hotkey(["ctrl", "v"])
            return ComputerUseResult(True, True, "Đã dán: Ctrl+V.", "paste")

        if "cua so hien tai" in plain or "cua so dang mo" in plain or "active window" in plain:
            title = _active_window_title() or "không đọc được tên cửa sổ"
            return ComputerUseResult(True, True, f"Cửa sổ đang active: {title}", "active_window", {"title": title})

        if "xem ui" in plain or "doc ui" in plain or "doc man hinh" in plain:
            return _ui_tree_result()

        if (
            "set text" in plain
            or (
                re.match(r"^(?:hay\s+)?(?:minion\s+)?(?:go|nhap|dien)\b", plain)
                and (" vao " in f" {plain} " or " to " in f" {plain} ")
            )
        ):
            return _set_ui_text(command, plain)

        if "clear text" in plain or "xoa text" in plain:
            query = _extract_ui_query(command, plain)
            return _clear_ui_text(query)

        if "doc control" in plain or "doc text" in plain or "thong tin control" in plain:
            query = _extract_ui_query(command, plain)
            return _read_ui_element(query)

        ui_index_match = re.search(r"(?:double click ui|right click ui|click ui|click dong|bam so)\s*(?:so)?\s*([0-9]{1,3})", plain)
        if ui_index_match:
            button = "right" if "right click" in plain or "click phai" in plain else "left"
            double = "double click" in plain
            return _click_ui_index(int(ui_index_match.group(1)), button=button, double=double)

        if "click nut" in plain or "click text" in plain or "bam nut" in plain:
            query = _extract_ui_query(command, plain)
            return _click_ui_element(query)

        if plain.startswith("focus ") or " focus " in f" {plain} ":
            query = _extract_ui_query(command, plain)
            return _focus_ui_element(query)

        if "vi tri chuot" in plain or "toa do chuot" in plain:
            x, y = _mouse_position()
            w, h = _screen_size()
            return ComputerUseResult(
                True,
                True,
                f"Vị trí chuột hiện tại: x={x}, y={y}. Màn hình: {w}x{h}.",
                "mouse_position",
                {"x": x, "y": y, "screen_width": w, "screen_height": h},
            )

        if "chup man hinh" in plain or "xem man hinh" in plain:
            return _screenshot()

        if re.match(r"^(?:hay\s+)?(?:minion\s+)?mo\b", plain) or "mo link" in plain or "mo trang" in plain:
            return _open_target(command, plain)

        coords = re.findall(r"(-?\d{1,5})\D+(-?\d{1,5})", plain)
        if ("keo chuot" in plain or "drag" in plain) and len(coords) >= 2:
            start_x, start_y = map(int, coords[-2])
            end_x, end_y = map(int, coords[-1])
            _mouse_drag(start_x, start_y, end_x, end_y)
            return ComputerUseResult(
                True,
                True,
                f"Đã kéo chuột từ x={start_x}, y={start_y} tới x={end_x}, y={end_y}.",
                "mouse_drag",
                {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y},
            )

        if ("di chuyen" in plain or "dua chuot" in plain) and coords:
            x, y = map(int, coords[-1])
            _set_mouse_position(x, y)
            return ComputerUseResult(True, True, f"Đã di chuyển chuột tới x={x}, y={y}.", "mouse_move", {"x": x, "y": y})

        if "click" in plain or "bam chuot" in plain:
            if coords:
                x, y = map(int, coords[-1])
                _set_mouse_position(x, y)
                time.sleep(0.08)
            button = "right" if "phai" in plain or "right" in plain else "middle" if "giua" in plain else "left"
            double = "double" in plain or "nhap dup" in plain or "2 lan" in plain
            _mouse_click(button=button, double=double)
            x, y = _mouse_position()
            return ComputerUseResult(
                True,
                True,
                f"Đã {('double ' if double else '')}click {button} tại x={x}, y={y}.",
                "mouse_click",
                {"x": x, "y": y, "button": button, "double": double},
            )

        if "cuon" in plain or "scroll" in plain:
            match = re.search(r"(-?\d{1,5})", plain)
            units = abs(int(match.group(1))) if match else 5
            amount = units * 120
            if "xuong" in plain or "down" in plain:
                amount = -amount
            _mouse_scroll(amount)
            direction = "xuống" if amount < 0 else "lên"
            return ComputerUseResult(True, True, f"Đã cuộn {direction} {units} nấc.", "mouse_scroll", {"amount": amount})

        if re.match(r"^(?:hay\s+)?(?:minion\s+)?(?:go|nhap)\b", plain):
            text = _extract_quoted_text(command)
            if text is None:
                text = re.sub(r"^(hãy\s+)?(gõ|go|nhập|nhap|dán|dan)\s+", "", command, flags=re.IGNORECASE).strip()
            if not text:
                return ComputerUseResult(True, False, "Anh cần ghi rõ nội dung muốn gõ, ví dụ: `gõ \"xin chào\"`.", "type")
            _send_unicode_text(text)
            return ComputerUseResult(True, True, f"Đã gõ: {text}", "type")

        if "bam phim" in plain or "nhan phim" in plain or "hotkey" in plain:
            quoted = _extract_quoted_text(command)
            key_text = quoted or re.sub(r".*(bam phim|nhan phim|hotkey)\s+", "", plain).strip()
            keys = re.split(r"\s*\+\s*|\s*,\s*|\s+", key_text)
            _press_hotkey([k for k in keys if k])
            return ComputerUseResult(True, True, f"Đã bấm phím: {'+'.join(keys)}", "hotkey")

        return ComputerUseResult(
            True,
            False,
            "Em chưa hiểu thao tác máy này. Anh có thể dùng: `mở notepad`, `vị trí chuột`, `di chuyển chuột tới 500 300`, `click`, `gõ \"nội dung\"`, `bấm phím ctrl+l`.",
            "unknown",
        )
    except Exception as exc:
        return ComputerUseResult(True, False, f"Không thực hiện được thao tác máy: {exc}", "error")
