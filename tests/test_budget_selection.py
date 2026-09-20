from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from select_training_budget import choose_budget


class BudgetTests(unittest.TestCase):
    def records(self):
        return [{'source': source, 'model': model, 'updates': updates, 'validation_ap': score}
                for source in ['iscx', 'mendeley'] for model in ['mlp', 'dqn', 'ddqn']
                for updates, score in [(1000, .80), (2000, .849), (4000, .85)]]

    def test_chooses_smallest_near_best_without_claiming_convergence(self):
        result = choose_budget(self.records(), .005)
        self.assertEqual(result['selected_updates'], 2000)
        self.assertTrue(result['best_at_largest_candidate'])
        self.assertFalse(result['convergence_established'])

    def test_rejects_incomplete_comparison(self):
        with self.assertRaises(ValueError):
            choose_budget(self.records()[:-1], .005)


if __name__ == '__main__':
    unittest.main()
