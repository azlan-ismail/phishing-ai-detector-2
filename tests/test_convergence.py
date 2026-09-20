import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_source_convergence import Plateau
from run_research_benchmark import train_neural, score_model


class ConvergenceTests(unittest.TestCase):
    def test_cumulative_small_gains_reset_against_anchor(self):
        rule=Plateau(.001,2)
        self.assertFalse(rule.observe(.8))
        self.assertFalse(rule.observe(.8006))
        self.assertFalse(rule.observe(.8012))
        self.assertEqual(rule.stale,0)
        self.assertFalse(rule.observe(.8011))
        self.assertTrue(rule.observe(.8013))

    def test_plateau_and_regression(self):
        rule=Plateau(.001,2)
        self.assertFalse(rule.observe(.9))
        self.assertFalse(rule.observe(.89))
        self.assertTrue(rule.observe(.9))

    def test_early_stop_matches_full_prefix_and_actual_budget(self):
        torch.set_num_threads(1)
        x=np.array([[-1.],[1.]]*16,dtype=np.float32)
        y=np.array([0,1]*16)
        early,h,info=train_neural(x,y,'ddqn',101,20,batch_size=8,observer=lambda u,m:u==10)
        prefix,ph,pi=train_neural(x,y,'ddqn',101,10,batch_size=8)
        self.assertEqual(info,pi)
        self.assertEqual(len(h),10)
        self.assertEqual(h,ph)
        np.testing.assert_array_equal(score_model(early,'ddqn',x),score_model(prefix,'ddqn',x))


if __name__=='__main__':
    unittest.main()
