import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_research_benchmark as bench


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_q_targets_and_terminal_mask(self):
        online = torch.tensor([[3., 1.], [0., 4.]])
        target = torch.tensor([[2., 10.], [5., 3.]])
        r = torch.tensor([1., -1.]); done = torch.tensor([False, True])
        dqn = bench.q_targets(r, done, online, target, .5, "dqn")
        ddqn = bench.q_targets(r, done, online, target, .5, "ddqn")
        torch.testing.assert_close(dqn, torch.tensor([6., -1.]))
        torch.testing.assert_close(ddqn, torch.tensor([2., -1.]))
        self.assertFalse(ddqn.requires_grad)

    def test_reward_follows_label_not_position(self):
        labels = np.array([0, 1, 0, 1]); actions = np.array([0, 1, 1, 0])
        expected = np.array([2., 3., -2., -3.])
        np.testing.assert_array_equal(bench.rewards_for(actions, labels, np.array([2., 3.])), expected)
        order = np.array([3, 0, 2, 1])
        np.testing.assert_array_equal(bench.rewards_for(actions[order], labels[order], np.array([2., 3.])), expected[order])

    def test_replay_wrap_and_terminal_storage(self):
        replay = bench.Replay(3, 1)
        x = np.arange(5, dtype=np.float32)[:, None]
        replay.add(x, np.zeros(5, dtype=int), np.arange(5), x + 1, np.array([False]*4+[True]))
        s, a, r, ns, done = replay.sample(3, np.random.RandomState(0))
        self.assertEqual(set(s[:, 0].tolist()), {2., 3., 4.})
        self.assertEqual(int(done.sum()), 1)
        torch.testing.assert_close(ns, s + 1)

    def test_constant_score_fpr_constraint_can_predict_none(self):
        y = np.array([0, 0, 1, 1]); score = np.ones(4)
        thresholds = bench.select_thresholds(y, score, .5, .01)
        result = bench.evaluate(y, score, thresholds['validation_fpr_limit'])
        self.assertEqual(result['fpr'], 0)
        self.assertEqual(result['recall'], 0)
        self.assertEqual(bench.evaluate(y, score, .5)['f1'], 2/3)

    def test_gamma_zero_control_and_seed_reproducibility(self):
        x = np.array([[-1.], [1.]] * 32, dtype=np.float32)
        y = np.array([0, 1] * 32)
        a, _, _ = bench.train_neural(x, y, 'dqn', 11, 50, batch_size=16, gamma=0)
        b, _, _ = bench.train_neural(x, y, 'ddqn', 11, 50, batch_size=16, gamma=0)
        np.testing.assert_array_equal(bench.score_model(a,'dqn',x), bench.score_model(b,'ddqn',x))

    def test_supervised_network_learns_separable_examples(self):
        x = np.array([[-1.], [1.]] * 32, dtype=np.float32)
        y = np.array([0, 1] * 32)
        net, history, info = bench.train_neural(x, y, 'mlp', 11, 50, batch_size=16)
        pred = bench.score_model(net, 'mlp', x) > 0
        self.assertGreater((pred == y).mean(), .95)
        self.assertEqual(info['optimizer_updates'], 50)
        self.assertEqual(len(history), 50)

    def test_export_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); name='iscx_to_iscx_train.csv'; path=root/name
            frame=pd.DataFrame({c: [0.,1.] for c in bench.INPUTS})
            frame.insert(0,'sample_id',['a','b']); frame['label']=[0,1]
            frame.to_csv(path,index=False)
            manifest={'exports':{name:{'sha256':bench.file_hash(path),'rows':2,'transformer_source':'iscx'}}}
            bench.load_export(root,manifest,'iscx','iscx','train')
            path.write_text(path.read_text()+'\n')
            with self.assertRaises(ValueError):
                bench.load_export(root,manifest,'iscx','iscx','train')

    def test_ablation_removes_excluded_signals_and_indicators_without_mutation(self):
        x = np.arange(24, dtype=np.float32).reshape(3, 8)
        before = x.copy()
        changed = x.copy()
        removed = [bench.INPUTS.index(n) for n in ['url_length', 'domain_url_ratio', 'url_length_missing', 'domain_url_ratio_missing']]
        changed[:, removed] = 999
        a = bench.apply_feature_condition(x, 'without_url_length_ratio')
        b = bench.apply_feature_condition(changed, 'without_url_length_ratio')
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(x, before)
        retained = [i for i in range(8) if i not in removed]
        np.testing.assert_array_equal(a[:, retained], x[:, retained])
        np.testing.assert_array_equal(a[:, removed], 0)


if __name__ == '__main__':
    unittest.main()
