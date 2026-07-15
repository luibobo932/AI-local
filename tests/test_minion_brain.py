import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from minion_agent import AGENT_TOOLS, run_tool_agent
from minion_core import load_minion_config, route_model
from minion_memory import MemoryStore
from train_minion_qlora import validate_dataset


class MinionCoreTests(unittest.TestCase):
    def test_config_keeps_defaults_when_old_file_has_few_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "minion.config.json"
            path.write_text(json.dumps({"max_agent_steps": 3}), encoding="utf-8")
            config = load_minion_config(path)

        self.assertEqual(config["max_agent_steps"], 3)
        self.assertEqual(config["model_router"]["general_model"], "qwen3:8b")
        self.assertTrue(config["memory"]["enabled"])

    def test_router_selects_code_model_and_general_model(self):
        config = load_minion_config("missing-config-for-test.json")
        self.assertEqual(route_model("Hãy sửa code Python trong workspace", config), "qwen2.5-coder:latest")
        self.assertEqual(route_model("Lập kế hoạch gọi khách hôm nay", config), "qwen3:8b")
        self.assertEqual(route_model("Sửa code", config, requested_model="phogpt-local"), "phogpt-local")

    def test_agent_tool_names_are_unique_and_include_guarded_writes(self):
        names = [item["function"]["name"] for item in AGENT_TOOLS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("workspace_patch", names)
        self.assertIn("workspace_run", names)
        self.assertIn("memory_search", names)
        self.assertIn("knowledge_ingest", names)

    def test_sft_sample_uses_conversational_schema(self):
        count, errors = validate_dataset(Path("data/minion_sft_sample.jsonl"))
        self.assertEqual(count, 3)
        self.assertEqual(errors, [])


class MinionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp_dir.name) / "memory.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_search_list_and_delete(self):
        item = self.store.add(
            "Anh Duy ưu tiên báo cáo nhà phố ngắn gọn và có số liệu kiểm chứng.",
            category="preference",
            source="user",
            importance=0.9,
        )
        self.store.add("Robot thử nghiệm phải có nút dừng khẩn cấp.", category="robotics")

        results = self.store.search("báo cáo nhà phố", limit=3, min_score=0.1)
        self.assertEqual(results[0].id, item.id)
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(len(self.store.list(category="preference")), 1)
        self.assertTrue(self.store.delete(item.id))
        self.assertFalse(self.store.delete(item.id))

    def test_semantic_embedding_can_rank_related_memory(self):
        first = self.store.add("Quy trình gọi chủ nhà", embedding=[1.0, 0.0])
        self.store.add("Hướng dẫn chế tạo robot", embedding=[0.0, 1.0])
        results = self.store.search("nội dung không trùng từ", query_embedding=[0.9, 0.1], min_score=0.0)
        self.assertEqual(results[0].id, first.id)

    def test_reingest_can_replace_every_chunk_from_same_source(self):
        self.store.add("Phần một", source="file:manual.md")
        self.store.add("Phần hai", source="file:manual.md")
        self.store.add("Ghi nhớ khác", source="user")
        self.assertEqual(self.store.delete_by_source("file:manual.md"), 2)
        self.assertEqual(self.store.count(), 1)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeAsyncClient:
    responses = []

    def __init__(self, *args, **kwargs):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        self.calls.append((url, json))
        return _FakeResponse(self.responses.pop(0))


class MinionAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_executes_tool_then_returns_final_answer(self):
        _FakeAsyncClient.responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "workspace_status", "arguments": {}}}
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Workspace đang sạch."}},
        ]
        executor = AsyncMock(return_value={"ok": True, "message": "clean", "risk_level": "safe"})
        events = []

        with patch("minion_agent.httpx.AsyncClient", _FakeAsyncClient):
            outcome = await run_tool_agent(
                base_url="http://127.0.0.1:11434",
                model="qwen3:8b",
                task="Kiểm tra workspace",
                system_prompt="Bạn là Minion",
                max_steps=3,
                execute_tool=executor,
                on_step=events.append,
            )

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.message, "Workspace đang sạch.")
        executor.assert_awaited_once_with("workspace_status", {})
        self.assertEqual(events[0]["tool"], "workspace_status")

    async def test_tool_loop_stops_when_approval_is_required(self):
        _FakeAsyncClient.responses = [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "workspace_run", "arguments": {"command": "dir"}}}
                    ],
                }
            }
        ]
        executor = AsyncMock(return_value={
            "ok": False,
            "message": "Cần anh xác nhận.",
            "risk_level": "needs_approval",
            "needs_approval": True,
        })

        with patch("minion_agent.httpx.AsyncClient", _FakeAsyncClient):
            outcome = await run_tool_agent(
                base_url="http://127.0.0.1:11434",
                model="qwen3:8b",
                task="Chạy lệnh",
                system_prompt="Bạn là Minion",
                max_steps=3,
                execute_tool=executor,
            )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.status, "needs_approval")


if __name__ == "__main__":
    unittest.main()
