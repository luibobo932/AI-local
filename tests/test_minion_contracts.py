import unittest
from unittest.mock import patch

from computer_use import (
    ComputerUseResult,
    execute_computer_command,
    workspace_current_diff_result,
    workspace_diagnostics_result,
    workspace_patch_result,
    workspace_review_result,
    workspace_status_result,
)


class MinionComputerUseContracts(unittest.TestCase):
    def test_real_estate_queries_are_not_computer_commands(self):
        samples = [
            "Tìm nhà Gò Vấp diện tích trên 60m2",
            "mô tả căn nhà Quận 10",
            "Có căn nào ngang từ 4m không?",
            "căn này có thay đổi giá không?",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertIsNone(execute_computer_command(text, True))

    def test_safe_computer_command_is_handled(self):
        result = execute_computer_command("workspace status", True)
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "workspace_status")

    def test_shell_command_requires_approval(self):
        result = execute_computer_command('chay lenh "dir"', True)
        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertTrue(result.needs_approval)
        self.assertEqual(result.risk_level, "needs_approval")

    def test_dangerous_command_is_blocked(self):
        result = execute_computer_command('chay lenh "git reset --hard"', True)
        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertFalse(result.needs_approval)
        self.assertEqual(result.risk_level, "blocked")

    def test_workspace_patch_preview_does_not_apply(self):
        status = workspace_status_result()
        self.assertTrue(status.ok)
        preview = workspace_patch_result(
            "minion.config.json",
            '"default_model": "minion"',
            '"default_model": "minion"',
            apply=False,
        )
        self.assertTrue(preview.ok)
        self.assertEqual(preview.action, "workspace_diff")
        self.assertEqual(preview.data.get("count"), 1)

    def test_workspace_diagnostics_is_safe_and_structured(self):
        fake_checks = [
            {"name": "git status", "ok": True, "command": "git status --short", "returncode": 0, "output": "clean"},
            {"name": "python compile", "ok": True, "command": "python -m py_compile computer_use.py server.py", "returncode": 0, "output": "(không có output)"},
            {"name": "unit tests", "ok": True, "command": "python -m unittest tests.test_minion_contracts", "returncode": 0, "output": "OK"},
        ]
        with (
            patch("computer_use._run_diagnostic_command", side_effect=fake_checks),
            patch("computer_use._server_port_check_result", return_value={"name": "server health", "ok": True, "command": "connect 127.0.0.1:11435", "returncode": 0, "output": "OK"}),
        ):
            result = execute_computer_command("kiểm tra dự án", True)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "workspace_diagnostics")
        self.assertEqual(result.risk_level, "safe")
        self.assertEqual(result.data.get("type"), "workspace_diagnostics")
        self.assertEqual(result.data.get("summary", {}).get("passed"), 4)

    def test_workspace_diagnostics_public_function(self):
        with (
            patch("computer_use._run_diagnostic_command", return_value={"name": "check", "ok": True, "command": "cmd", "returncode": 0, "output": "OK"}),
            patch("computer_use._server_port_check_result", return_value={"name": "server health", "ok": True, "command": "connect", "returncode": 0, "output": "OK"}),
        ):
            result = workspace_diagnostics_result()

        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("summary", {}).get("total"), 4)

    def test_workspace_review_reports_findings(self):
        diagnostics = ComputerUseResult(
            True,
            True,
            "diag",
            "workspace_diagnostics",
            {
                "type": "workspace_diagnostics",
                "checks": [
                    {"name": "git status", "ok": True, "command": "git status --short --branch --untracked-files=all", "returncode": 0, "output": "## branch\n M computer_use.py"},
                    {"name": "python compile", "ok": True, "command": "python -m py_compile computer_use.py server.py", "returncode": 0, "output": "(không có output)"},
                    {"name": "unit tests", "ok": True, "command": "python -m unittest tests.test_minion_contracts", "returncode": 0, "output": "OK"},
                    {"name": "server health", "ok": True, "command": "connect 127.0.0.1:11435", "returncode": 0, "output": "OK"},
                ],
            },
        )
        review_checks = [
            {"name": "diff whitespace", "ok": True, "command": "git diff --check", "returncode": 0, "output": "(không có output)"},
            {"name": "todo scan", "ok": True, "command": "rg TODO", "returncode": 0, "output": "computer_use.py:1:# TODO: demo"},
        ]
        with (
            patch("computer_use._workspace_diagnostics", return_value=diagnostics),
            patch("computer_use._run_diagnostic_command", side_effect=review_checks),
        ):
            result = execute_computer_command("review dự án", True)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "workspace_review")
        self.assertEqual(result.data.get("type"), "workspace_review")
        titles = [item.get("title") for item in result.data.get("findings", [])]
        self.assertIn("Có thay đổi chưa commit", titles)
        self.assertTrue(any("TODO" in title for title in titles))

    def test_workspace_review_public_function(self):
        diagnostics = ComputerUseResult(True, True, "diag", "workspace_diagnostics", {"checks": []})
        with (
            patch("computer_use._workspace_diagnostics", return_value=diagnostics),
            patch("computer_use._run_diagnostic_command", return_value={"name": "check", "ok": True, "command": "cmd", "returncode": 0, "output": "(không có output)"}),
        ):
            result = workspace_review_result()

        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("type"), "workspace_review")

    def test_workspace_current_diff_is_safe_and_structured(self):
        fake_checks = [
            {"name": "git status", "ok": True, "command": "git status --short --branch --untracked-files=all", "returncode": 0, "output": "## branch\n M computer_use.py\n?? notes.txt"},
            {"name": "git diff stat", "ok": True, "command": "git diff --stat HEAD --", "returncode": 0, "output": " computer_use.py | 2 ++"},
            {"name": "git diff files", "ok": True, "command": "git diff --name-only HEAD --", "returncode": 0, "output": "computer_use.py"},
            {"name": "git diff", "ok": True, "command": "git diff --color=never HEAD --", "returncode": 0, "output": "diff --git a/computer_use.py b/computer_use.py"},
        ]
        with patch("computer_use._run_diagnostic_command", side_effect=fake_checks):
            result = execute_computer_command("xem diff", True)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "workspace_current_diff")
        self.assertEqual(result.risk_level, "safe")
        self.assertEqual(result.data.get("type"), "workspace_current_diff")
        self.assertEqual(result.data.get("summary", {}).get("changed"), 2)
        self.assertIn("notes.txt", result.data.get("files", []))

    def test_workspace_current_diff_public_function(self):
        with patch("computer_use._run_diagnostic_command", return_value={"name": "check", "ok": True, "command": "cmd", "returncode": 0, "output": "(không có output)"}):
            result = workspace_current_diff_result()

        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("type"), "workspace_current_diff")

    def test_youtube_query_keeps_vietnamese_accents(self):
        with (
            patch("computer_use.webbrowser.open") as open_mock,
            patch("computer_use._resolve_youtube_first_video_url", return_value="https://www.youtube.com/watch?v=abc12345678"),
        ):
            result = execute_computer_command("mở youtube bài come my way của Sơn Tùng và bật lên luôn", True)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "youtube_play")
        self.assertIn("Sơn Tùng", result.message)
        self.assertEqual(result.data.get("query"), "come my way của Sơn Tùng")
        open_mock.assert_called_once_with("https://www.youtube.com/watch?v=abc12345678")

    def test_youtube_then_volume_sequence(self):
        command = "mở youtube bài come my way của Sơn Tùng và bật lên luôn. mở âm lượng tối đa máy tính cho tôi"
        with (
            patch("computer_use.webbrowser.open"),
            patch("computer_use._resolve_youtube_first_video_url", return_value="https://www.youtube.com/watch?v=abc12345678"),
            patch("computer_use._tap_virtual_key") as volume_mock,
        ):
            result = execute_computer_command(command, True)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "sequence")
        self.assertEqual(result.data.get("actions"), ["youtube_play", "volume_max"])
        volume_mock.assert_called_once()

    def test_volume_max_is_handled(self):
        with patch("computer_use._tap_virtual_key") as volume_mock:
            result = execute_computer_command("mở âm lượng tối đa máy tính cho tôi", True)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "volume_max")
        volume_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
