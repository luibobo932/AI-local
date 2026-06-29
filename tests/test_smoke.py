"""
Smoke test cho Minion MVP — chỉ dùng stdlib (unittest).

Dựng một LLM server giả (OpenAI-compatible) để kiểm tra end-to-end:
- llm.LLMClient gọi được endpoint
- agent.run_agent chạy vòng lặp, gọi tool, dừng đúng
- Permission policy chặn tool ghi ở chế độ plan
- Các tool cơ bản hoạt động

Chạy:
    python -m unittest tests.test_smoke
    hoặc: python tests/test_smoke.py
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Cho phép import module ở thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LLMClient, load_env  # noqa: E402
from tools import call_tool, get_tool_schemas, TOOL_REGISTRY  # noqa: E402
from tools.permissions import make_policy, classify  # noqa: E402


# ─── Fake LLM server (OpenAI-compatible) ──────────────────────────────────────

class _FakeLLMHandler(BaseHTTPRequestHandler):
    """Trả lời theo kịch bản: bước 1 gọi tool, bước 2 trả lời cuối."""
    call_count = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json({"data": [{"id": "fake-model"}]})
        else:
            self._json({"error": "nf"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode()) if length else {}
        type(self).call_count += 1

        has_tools = bool(body.get("tools"))
        # Lượt đầu + có tools → yêu cầu gọi tool list_dir
        if type(self).call_count == 1 and has_tools:
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})},
                }],
            }
            self._json({"choices": [{"message": msg, "finish_reason": "tool_calls"}]})
        else:
            self._json({"choices": [{
                "message": {"role": "assistant", "content": "Đã xong nhiệm vụ."},
                "finish_reason": "stop",
            }]})

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _FakeLLMHandler.call_count = 0
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLMHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    # ── LLM client ──
    def test_llm_client_chat(self):
        client = LLMClient(base_url=self.base_url, model="fake-model")
        text = client.chat_text([{"role": "user", "content": "hi"}])
        self.assertIn("xong", text.lower())

    def test_llm_list_models(self):
        client = LLMClient(base_url=self.base_url, model="fake-model")
        self.assertIn("fake-model", client.list_models())

    # ── Tools ──
    def test_tool_registry_matches_schemas(self):
        reg = set(TOOL_REGISTRY)
        sch = {s["function"]["name"] for s in get_tool_schemas()}
        self.assertEqual(reg, sch)

    def test_read_only_tools_work(self):
        out = call_tool("list_dir", {"path": "."})
        self.assertNotIn("[ERROR]", out)
        out2 = call_tool("glob", {"pattern": "*.py"})
        self.assertIn("minion.py", out2)

    def test_unknown_tool(self):
        self.assertIn("không tồn tại", call_tool("khong_co", {}))

    # ── Permission ──
    def test_classify(self):
        self.assertEqual(classify("read_file"), "read")
        self.assertEqual(classify("write_file"), "write")
        self.assertEqual(classify("run_command"), "dangerous")

    def test_plan_blocks_write(self):
        policy = make_policy("plan")
        out = call_tool("write_file", {"path": "x.txt", "content": "hi"}, policy=policy)
        self.assertTrue(out.startswith("[BLOCKED]"))
        # đọc vẫn chạy
        out2 = call_tool("list_dir", {"path": "."}, policy=policy)
        self.assertFalse(out2.startswith("[BLOCKED]"))

    def test_dangerous_command_blocked(self):
        out = call_tool("run_command", {"command": "rm -rf /"})
        self.assertIn("[BLOCKED]", out)

    # ── Agent end-to-end ──
    def test_agent_runs_and_uses_tool(self):
        from agent import run_agent
        _FakeLLMHandler.call_count = 0
        result = run_agent(
            task="liệt kê thư mục",
            base_url=self.base_url,
            model="fake-model",
            mode="function_calling",
            max_steps=5,
            use_memory=False,
        )
        self.assertTrue(result.success)
        self.assertIn("xong", result.answer.lower())
        # Có ít nhất 1 tool call (list_dir)
        tool_names = [tc.name for s in result.steps for tc in s.tool_calls]
        self.assertIn("list_dir", tool_names)

    def test_agent_max_steps_limit(self):
        # max_steps=1: không được lặp vô hạn
        from agent import run_agent
        _FakeLLMHandler.call_count = 0
        result = run_agent(
            task="x", base_url=self.base_url, model="fake-model",
            mode="function_calling", max_steps=1, use_memory=False,
        )
        self.assertLessEqual(len(result.steps), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
