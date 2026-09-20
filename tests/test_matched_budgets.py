import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_research_benchmark import apply_matched_budgets


class MatchedBudgetTests(unittest.TestCase):
    def test_changes_only_neural_updates_without_mutating_selection(self):
        original={'selected':{s:{**{m:{'updates':8000,'learning_rate':.001} for m in ['mlp','dqn','ddqn']},
            'logistic':{'C':100},'random_forest':{'max_depth':8,'min_samples_leaf':1}} for s in ['iscx','mendeley']}}
        before=copy.deepcopy(original)
        result=apply_matched_budgets(original,{'source_updates':{'iscx':16000,'mendeley':28000}})
        self.assertEqual(original,before)
        for source,budget in [('iscx',16000),('mendeley',28000)]:
            for name in ['mlp','dqn','ddqn']:
                self.assertEqual(result['selected'][source][name],{'updates':budget,'learning_rate':.001})
            for name in ['logistic','random_forest']:
                self.assertEqual(result['selected'][source][name],before['selected'][source][name])

    def test_invalid_budget_or_missing_source_rejected(self):
        for budgets in [{'iscx':16000},{'iscx':16000,'mendeley':0},{'iscx':16000,'mendeley':1.5}]:
            with self.assertRaises(ValueError):
                apply_matched_budgets({'selected':{'iscx':{m:{} for m in ['mlp','dqn','ddqn']},'mendeley':{}}},{'source_updates':budgets})
