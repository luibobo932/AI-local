import tempfile
import unittest
from pathlib import Path

from data.build_minion_v4 import build
from data.build_minion_dpo import build as build_dpo
from data.validate_minion_v4 import validate
from evals.build_minion_v4_eval import build_cases


class MinionV4DataTests(unittest.TestCase):
    def test_frozen_eval_has_100_cases(self):
        cases = build_cases()
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({case["id"] for case in cases}), 100)
        self.assertEqual(sum(bool(case["critical"]) for case in cases), 30)

    def test_dataset_counts_and_family_isolation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            report = build(output)
            validation = validate(output)
        self.assertEqual(report["splits"], {"train": 700, "validation": 51, "test": 51})
        self.assertTrue(validation["ok"], validation["errors"])

    def test_dpo_pairs_have_isolated_families(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_dpo(Path(temp_dir))
        self.assertEqual(report["splits"], {"train": 319, "validation": 41})
        self.assertFalse(report["family_overlap"])


if __name__ == "__main__":
    unittest.main()
