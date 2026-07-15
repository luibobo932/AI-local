"""Định nghĩa tool và vòng lặp Ollama tool-calling cho Minion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
StepCallback = Callable[[dict[str, Any]], None]
StopCallback = Callable[[], bool]


AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "workspace_status",
            "description": "Xem trạng thái Git và workspace hiện tại.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_diagnostics",
            "description": "Chạy chẩn đoán dự án gồm compile, test và health check.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_review",
            "description": "Review thay đổi hiện tại và trả finding có cấu trúc.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_diff",
            "description": "Xem diff hiện tại của workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_list_files",
            "description": "Liệt kê file trong workspace, có thể lọc theo tên.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_search",
            "description": "Tìm chuỗi hoặc biểu thức trong mã nguồn workspace.",
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "properties": {"pattern": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_read",
            "description": "Đọc một file văn bản nằm trong workspace.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_patch",
            "description": "Thay chính xác đoạn văn bản trong file. apply=false để xem trước; apply=true sẽ qua lớp xin phép.",
            "parameters": {
                "type": "object",
                "required": ["path", "old", "new"],
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "apply": {"type": "boolean", "default": False},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_run",
            "description": "Chạy lệnh trong workspace. Luôn đi qua phân loại rủi ro và xin phép.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_command",
            "description": "Thực hiện lệnh desktop bằng tiếng Việt. Các thao tác rủi ro vẫn cần xác nhận.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Tìm trí nhớ local liên quan đến câu hỏi hoặc nhiệm vụ.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_remember",
            "description": "Chỉ lưu trí nhớ khi người dùng nói rõ muốn Minion ghi nhớ điều đó.",
            "parameters": {
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {"type": "string"},
                    "category": {"type": "string", "default": "general"},
                    "source": {"type": "string", "default": "user"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_ingest",
            "description": "Chỉ khi người dùng yêu cầu rõ: nhập một file văn bản trong workspace vào kho kiến thức local.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "category": {"type": "string", "default": "knowledge"},
                },
            },
        },
    },
]


@dataclass(slots=True)
class AgentOutcome:
    ok: bool
    status: str
    message: str
    steps: int
    model: str


def _normalize_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Chấp nhận model local trả tool call dạng JSON text thay vì field tool_calls."""
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    values = value if isinstance(value, list) else [value]
    calls = []
    for item in values:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = item.get("name") or function.get("name")
        arguments = item.get("arguments") or function.get("arguments") or {}
        if name:
            calls.append({"function": {"name": str(name), "arguments": arguments}})
    return calls


async def run_tool_agent(
    *,
    base_url: str,
    model: str,
    task: str,
    system_prompt: str,
    max_steps: int,
    execute_tool: ToolExecutor,
    on_step: StepCallback | None = None,
    should_stop: StopCallback | None = None,
    temperature: float = 0.1,
) -> AgentOutcome:
    """Chạy vòng lặp model -> tool -> kết quả cho tới khi model kết luận."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    max_steps = max(1, min(int(max_steps), 20))

    async with httpx.AsyncClient(timeout=None) as client:
        for index in range(1, max_steps + 1):
            if should_stop and should_stop():
                return AgentOutcome(False, "stopped", "Đã dừng theo yêu cầu.", index - 1, model)

            response = await client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": AGENT_TOOLS,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            payload = response.json()
            assistant = payload.get("message") or {}
            tool_calls = assistant.get("tool_calls") or []
            content = str(assistant.get("content") or "").strip()
            if not tool_calls and content:
                tool_calls = _tool_calls_from_content(content)
            messages.append(assistant)

            if not tool_calls:
                final = content or "Minion đã xử lý xong nhưng model không trả nội dung kết luận."
                return AgentOutcome(True, "completed", final, index, model)

            for call in tool_calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = _normalize_arguments(function.get("arguments"))
                result = await execute_tool(name, arguments)
                event = {
                    "index": index,
                    "tool": name,
                    "arguments": arguments,
                    "result": result,
                }
                if on_step:
                    on_step(event)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
                if result.get("needs_approval"):
                    return AgentOutcome(False, "needs_approval", str(result.get("message") or "Cần xác nhận."), index, model)
                if result.get("risk_level") == "blocked":
                    return AgentOutcome(False, "blocked", str(result.get("message") or "Thao tác bị chặn."), index, model)

    return AgentOutcome(
        False,
        "max_steps",
        f"Minion đã dùng hết {max_steps} bước nhưng chưa có kết luận cuối.",
        max_steps,
        model,
    )
