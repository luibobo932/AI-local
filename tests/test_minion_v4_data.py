import tempfile
import unittest
from pathlib import Path

from data.build_minion_v4 import build
from data.build_minion_dpo import build as build_dpo
from data.build_minion_v4_hardening import build as build_hardening
from data.validate_minion_v4 import validate
from evals.build_minion_v4_eval import build_cases
from evals.check_minion_v4_gate import check
from evals.evaluate_minion_v4 import load_cases
from align_minion_dpo import assess_training_gate


class MinionV4DataTests(unittest.TestCase):
    def test_frozen_eval_has_100_cases(self):
        cases = build_cases()
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({case["id"] for case in cases}), 100)
        self.assertEqual(sum(bool(case["critical"]) for case in cases), 30)

    def test_frozen_eval_can_load_critical_subset(self):
        cases = load_cases(Path("evals/minion_v4_eval.jsonl"), critical_only=True)
        self.assertEqual(len(cases), 30)
        self.assertTrue(all(case["critical"] for case in cases))

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

    def test_hardening_dataset_has_targeted_examples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_hardening(Path(temp_dir) / "train.jsonl")
        self.assertEqual(report["count"], 430)
        self.assertGreaterEqual(report["categories"]["safety"], 120)
        self.assertGreaterEqual(report["categories"]["robotics"], 75)

    def test_gate_fails_closed_on_critical_failure(self):
        report = {
            "adapter_result": {
                "critical_failures": ["safety_01"],
                "categories": {name: {"rate": 1.0} for name in ("identity", "honesty", "safety", "tool_calling", "real_estate", "coding_git", "robotics")},
            }
        }
        decision = check(report)
        self.assertFalse(decision["passed"])
        self.assertIn("Còn 1 critical failure", decision["reasons"])

    def test_dpo_gate_only_allows_failures_in_explicit_repair_mode(self):
        report = {"adapter_result": {"critical_failures": ["safety_01"]}}
        normal = assess_training_gate(report, repair_mode=False)
        repair = assess_training_gate(report, repair_mode=True)
        self.assertFalse(normal["allowed"])
        self.assertEqual(normal["exit_code"], 2)
        self.assertTrue(repair["allowed"])
        self.assertTrue(repair["deployment_locked"])

    def test_dpo_gate_fails_closed_on_missing_result(self):
        decision = assess_training_gate({}, repair_mode=True)
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
