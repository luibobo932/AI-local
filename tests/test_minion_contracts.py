import unittest

from computer_use import execute_computer_command, workspace_patch_result, workspace_status_result


class MinionComputerUseContracts(unittest.TestCase):
    def test_real_estate_queries_are_not_computer_commands(self):
        samples = [
            "Tìm nhà Gò Vấp diện tích trên 60m2",
            "mô tả căn nhà Quận 10",
            "Có căn nào ngang từ 4m không?",
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


if __name__ == "__main__":
    unittest.main()
