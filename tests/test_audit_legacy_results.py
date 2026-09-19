import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_legacy_results.py"
spec = importlib.util.spec_from_file_location("audit_legacy_results", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class LegacyMetricTests(unittest.TestCase):
    def test_detects_incorrect_saved_metric(self):
        metrics = {"confusion_matrix": [[3, 1], [1, 3]],
                   "classification_report": {"1": {"precision": .75, "recall": .75,
                   "f1-score": .75, "support": 4}, "accuracy": .75}}
        self.assertTrue(audit.check_metrics(metrics)["all_checked_values_match"])
        metrics["classification_report"]["1"]["recall"] = .9
        self.assertFalse(audit.check_metrics(metrics)["all_checked_values_match"])

    def test_zero_positive_predictions(self):
        metrics = {"confusion_matrix": [[3, 0], [1, 0]],
                   "classification_report": {"1": {"precision": 0, "recall": 0,
                   "f1-score": 0, "support": 1}, "accuracy": .75}}
        self.assertTrue(audit.check_metrics(metrics)["all_checked_values_match"])

    def test_invalid_matrix_rejected(self):
        for matrix in [[[0, 0], [0, 0]], [[-1, 1], [1, 1]], [[1, 2, 3]], [[1.5, 0], [0, 1]]]:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    audit.check_metrics({"confusion_matrix": matrix})


if __name__ == "__main__":
    unittest.main()
