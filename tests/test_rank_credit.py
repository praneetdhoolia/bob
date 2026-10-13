"""Exact sign-averaged sensitivity and actual-core temporal-credit execution."""
import itertools
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from liquid_core import LiquidCore
from online_context import rank_one_credit,RankOneContextLearner


class RankCreditTests(unittest.TestCase):
    def test_all_signs_recover_expected_linear_sensitivity(self):
        torch.manual_seed(91)
        jac_state=torch.randn(3,3,dtype=torch.float64)
        jac_parameters=torch.randn(3,4,dtype=torch.float64)
        u=torch.randn(3,dtype=torch.float64);v=torch.randn(4,dtype=torch.float64)
        tangent=jac_state@u;expected=torch.outer(tangent,v)+jac_parameters
        estimates=[]
        for signs in itertools.product([-1.,1.],repeat=3):
            noise=torch.tensor(signs,dtype=torch.float64)
            next_u,next_v=rank_one_credit(tangent,[v],noise,[noise@jac_parameters])
            estimates.append(torch.outer(next_u,next_v[0]))
        self.assertTrue(torch.allclose(torch.stack(estimates).mean(0),expected,atol=1e-10,rtol=1e-10))

    def test_real_core_step_has_bounded_sensitivity_storage(self):
        torch.set_num_threads(2);torch.manual_seed(102)
        core=LiquidCore().eval();learner=RankOneContextLearner(core,train_writes=True)
        state=core.start().detach().clone();original=state.clone()
        for _ in range(3):
            state,code,metrics=learner.step(state,torch.randn(1,3,4096),.1)
            self.assertTrue(torch.isfinite(state).all())
            self.assertLess(metrics['sensitivityBytes'],3_000_000)
            self.assertLessEqual(metrics['parameterRmsDrift'],.010001)
            self.assertFalse(state.requires_grad)
        self.assertFalse(torch.equal(state,original))
        self.assertGreater(float(learner.state_factor.norm()),0)
        learner.reset_credit();self.assertEqual(float(learner.state_factor.norm()),0)

    def test_short_credit_control_discards_earlier_state_factor(self):
        torch.set_num_threads(2);torch.manual_seed(113)
        core=LiquidCore().eval()
        learner=RankOneContextLearner(core,train_writes=True,reset_each_packet=True)
        state=core.start().detach().clone()
        for _ in range(3):
            state,_,_=learner.step(state,torch.randn(1,3,4096),.1)
            # Only the current sign vector remains after reset/reduction. Its
            # constant magnitude proves no earlier tangent leaks into the trace.
            self.assertEqual(torch.unique(learner.state_factor.abs()).numel(),1)
            self.assertTrue(torch.isfinite(state).all())


if __name__=='__main__':unittest.main()
