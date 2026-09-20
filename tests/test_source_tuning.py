from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tune_source_models import choose
from run_research_benchmark import train_neural, score_model


class SourceTuningTests(unittest.TestCase):
    def test_objective_and_tie_rule(self):
        rows = [{'candidate': 2, 'validation_ap': .9}, {'candidate': 1, 'validation_ap': .9}, {'candidate': 0, 'validation_ap': .8}]
        self.assertEqual(choose(rows)['candidate'], 1)

    def test_validation_observer_preserves_training_and_prefix_checkpoint(self):
        torch.set_num_threads(1)
        x = np.array([[-1.], [1.]] * 16, dtype=np.float32)
        y = np.array([0,1] * 16)
        scores = {}
        def observe(update, model):
            if update == 10:
                scores[update] = score_model(model, 'ddqn', x).copy()
        full, history, info = train_neural(x,y,'ddqn',101,20,batch_size=8,observer=observe)
        reference, ref_history, ref_info = train_neural(x,y,'ddqn',101,20,batch_size=8)
        prefix, _, _ = train_neural(x,y,'ddqn',101,10,batch_size=8)
        np.testing.assert_array_equal(scores[10], score_model(prefix,'ddqn',x))
        np.testing.assert_array_equal(score_model(full,'ddqn',x), score_model(reference,'ddqn',x))
        self.assertEqual(history, ref_history)
        self.assertEqual(info, ref_info)


if __name__ == '__main__':
    unittest.main()
