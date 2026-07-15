import json
import tempfile
import unittest
from pathlib import Path

from data.build_minion_sft import build_dataset
from data.validate_minion_sft import validate


class MinionTrainingDataTests(unittest.TestCase):
    def test_seed_dataset_is_large_and_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "seed.jsonl"
            records = build_dataset(output)
            report = validate(output)

        self.assertGreaterEqual(len(records), 100)
        self.assertTrue(report["ok"], report["errors"])
        self.assertGreaterEqual(report["tool_examples"], 5)

    def test_seed_dataset_has_no_duplicate_user_prompt(self):
        records = [json.loads(line) for line in Path("data/minion_sft_seed.jsonl").read_text(encoding="utf-8").splitlines()]
        prompts = []
        for record in records:
            user = next(message["content"] for message in record["messages"] if message["role"] == "user")
            prompts.append(" ".join(user.lower().split()))
        self.assertEqual(len(prompts), len(set(prompts)))


if __name__ == "__main__":
    unittest.main()
