"""Test bộ não agent: auto-verify, verify gate, nén ngữ cảnh, gợi ý kế hoạch."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent as agent_mod
from agent import (
    AgentConfig,
    _auto_verify_note,
    _compact_messages,
    _planning_hint,
    _run_function_calling,
    _total_size,
)


# ─── _auto_verify_note ────────────────────────────────────────────────────────

def test_auto_verify_ok():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("x = 1\n")
        path = f.name
    try:
        note = _auto_verify_note("edit_file", {"path": path})
        assert "✅" in note and "OK" in note
    finally:
        os.unlink(path)


def test_auto_verify_syntax_error():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("def broken(:\n")
        path = f.name
    try:
        note = _auto_verify_note("write_file", {"path": path})
        assert "❌" in note and "CÚ PHÁP" in note.upper()
    finally:
        os.unlink(path)


def test_auto_verify_skips_non_edit_and_non_py():
    assert _auto_verify_note("read_file", {"path": "a.py"}) == ""
    assert _auto_verify_note("edit_file", {"path": "a.txt"}) == ""
    assert _auto_verify_note("edit_file", {"path": "khong_ton_tai_xyz.py"}) == ""


# ─── _compact_messages ────────────────────────────────────────────────────────

def _fake_convo(n_steps: int, tool_output_len: int = 3000):
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task gốc"},
    ]
    for i in range(n_steps):
        msgs.append({
            "role": "assistant", "content": f"nghĩ bước {i}",
            "tool_calls": [{"id": f"c{i}", "function": {"name": "read_file", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * tool_output_len})
    return msgs


def test_compact_noop_under_budget():
    msgs = _fake_convo(2, tool_output_len=100)
    assert _compact_messages(msgs, 100_000) is msgs


def test_compact_truncates_old_tool_output():
    msgs = _fake_convo(6, tool_output_len=3000)
    out = _compact_messages(msgs, 12_000)
    assert _total_size(out) <= 12_000
    # system + task gốc còn nguyên
    assert out[0]["content"] == "system prompt"
    assert out[1]["content"] == "task gốc"
    # nhóm cuối cùng không bị đụng tới
    assert out[-1]["content"] == "X" * 3000


def test_compact_drops_groups_without_orphan_tools():
    msgs = _fake_convo(10, tool_output_len=5000)
    out = _compact_messages(msgs, 8_000)
    # Không có tool message mồ côi: tool phải đứng ngay sau assistant có tool_calls hoặc tool khác
    for i, m in enumerate(out):
        if m.get("role") == "tool":
            prev = out[i - 1]
            assert prev.get("role") == "tool" or prev.get("tool_calls"), f"tool mồ côi tại {i}"
    # Có ghi chú nén
    assert any("[Ngữ cảnh đã nén]" in str(m.get("content", "")) for m in out)


# ─── _planning_hint ───────────────────────────────────────────────────────────

def test_planning_hint_simple_task():
    assert _planning_hint("đọc file README", 10) == ""


def test_planning_hint_long_task():
    task = "refactor toàn bộ module server và viết test cho từng endpoint rồi cập nhật tài liệu API sau đó chạy toàn bộ test suite để xác nhận"
    assert "todo_write" in _planning_hint(task, 10)


def test_planning_hint_disabled_for_short_budget():
    assert _planning_hint("x" * 200, 2) == ""


# ─── Verify gate (mock LLM) ───────────────────────────────────────────────────

def test_verify_gate_forces_run_before_finish(monkeypatch, tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")

    edit_call = {
        "id": "c1",
        "function": {
            "name": "edit_file",
            "arguments": (
                '{"path": "%s", "old_text": "x = 1", "new_text": "x = 2"}'
                % str(target).replace("\\", "\\\\")
            ),
        },
    }
    run_call = {
        "id": "c2",
        "function": {"name": "run_command", "arguments": '{"command": "python -c \\"print(11)\\""}'},
    }

    script = [
        # Bước 1: model sửa file
        {"choices": [{"message": {"content": "sửa file", "tool_calls": [edit_call]}}]},
        # Bước 2: model đòi kết thúc luôn (chưa kiểm chứng) → gate phải chặn
        {"choices": [{"finish_reason": "stop", "message": {"content": "xong rồi (chưa test)"}}]},
        # Bước 3: sau khi bị nhắc, model chạy kiểm chứng
        {"choices": [{"message": {"content": "chạy thử", "tool_calls": [run_call]}}]},
        # Bước 4: kết thúc thật
        {"choices": [{"finish_reason": "stop", "message": {"content": "đã kiểm chứng, hoàn thành"}}]},
    ]
    seen_messages = []

    def fake_chat(base_url, model, messages, tools, temperature, max_tokens, timeout, api_key=""):
        seen_messages.append([dict(m) for m in messages])
        return script.pop(0)

    monkeypatch.setattr(agent_mod, "_chat_request", fake_chat)

    cfg = AgentConfig(model="fake", max_steps=6, auto_verify=True)
    result = _run_function_calling("sửa mod.py đổi x thành 2", cfg)

    assert result.success
    assert result.answer == "đã kiểm chứng, hoàn thành"
    # Lời nhắc kiểm chứng đã được chèn vào hội thoại
    flat = [m.get("content", "") for msgs in seen_messages for m in msgs]
    assert any("[Kiểm chứng bắt buộc]" in c for c in flat)
    # File thực sự được sửa và run_command đã chạy
    assert target.read_text(encoding="utf-8") == "x = 2\n"
    tool_names = [tc.name for s in result.steps for tc in s.tool_calls]
    assert "edit_file" in tool_names and "run_command" in tool_names


def test_no_gate_when_no_edits(monkeypatch):
    script = [
        {"choices": [{"finish_reason": "stop", "message": {"content": "trả lời thẳng"}}]},
    ]

    def fake_chat(*a, **k):
        return script.pop(0)

    monkeypatch.setattr(agent_mod, "_chat_request", fake_chat)
    cfg = AgentConfig(model="fake", max_steps=4, auto_verify=True)
    result = _run_function_calling("1+1 bằng mấy", cfg)
    assert result.answer == "trả lời thẳng"
    assert len(result.steps) == 1
