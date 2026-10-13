"""Causal ordering, shared weights and retention bounds for experimental learning."""
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'neural'))
from liquid_core import LiquidCore
from online_context import ContextLearner


class ContextTests(unittest.TestCase):
    def test_learning_changes_shared_weights_without_mutating_prior(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        core=LiquidCore().eval();prior=core.start().detach().clone()
        tokens=torch.randn(1,3,4096);original=prior.clone()
        frozen={name:p.detach().clone() for name,p in core.named_parameters() if not name.startswith('input_context') and name!='context_gain'}
        with torch.no_grad():before=core.predict_sensory(prior).clone()
        learner=ContextLearner(core,learning_rate=.001,max_rms_drift=.0001)
        for _ in range(4):result=learner.update(prior,tokens)
        self.assertTrue(torch.equal(prior,original))
        self.assertFalse(torch.equal(before,core.predict_sensory(prior)))
        self.assertLessEqual(result['parameterRmsDrift'],.00010001)
        self.assertTrue(all(torch.equal(frozen[name],p) for name,p in core.named_parameters() if name in frozen))
        with torch.no_grad():
            changed=core.feedback(prior,65)
            core.input_context[1].weight.zero_();core.input_context[1].bias.zero_()
            disabled=core.feedback(prior,65)
        self.assertFalse(torch.equal(changed,disabled))

    def test_inference_tensors_can_train_and_zero_context_activation_preserves_behavior(self):
        core=LiquidCore().eval();torch.set_num_threads(2)
        with torch.inference_mode():
            prior=core.start().clone();tokens=torch.randn(1,2,4096)
            expected=core.advance(core.feedback(prior,65),.01).clone()
        learner=ContextLearner(core)
        with torch.no_grad():actual=core.advance(core.feedback(prior,65),.01)
        self.assertTrue(torch.equal(actual,expected))
        learner.update(prior,tokens)

    def test_next_packet_loss_reaches_previous_shared_write(self):
        torch.manual_seed(59);torch.set_num_threads(2)
        core=LiquidCore().eval();learner=ContextLearner(core,train_writes=True)
        old=core.write_strength.weight.detach().clone()
        packet=torch.randn(1,3,4096)
        state=core.observe(core.start().detach().clone(),packet,.1)
        state=core.advance(state,.1)
        # Context's zero initialization blocks state credit on the first update.
        learner.update(state,torch.randn(1,3,4096))
        state=core.observe(state.detach(),packet,.1)
        state=core.advance(state,.1)
        learner.update(state,torch.randn(1,3,4096))
        self.assertFalse(torch.equal(old,core.write_strength.weight))
        self.assertTrue(torch.isfinite(core.write_strength.weight).all())

    def test_temporal_context_control_keeps_write_weights_frozen(self):
        core=LiquidCore().eval();learner=ContextLearner(core,temporal_credit=True)
        old=core.write_strength.weight.detach().clone()
        packet=torch.randn(1,2,4096);state=core.start().detach().clone()
        for _ in range(3):
            state=core.observe(state.detach(),packet,.1)
            learner.update(core.advance(state,.1),torch.randn(1,2,4096))
        self.assertTrue(learner.temporal_credit);self.assertFalse(learner.train_writes)
        self.assertTrue(torch.equal(old,core.write_strength.weight))
        self.assertIsNone(core.write_strength.weight.grad)


if __name__=='__main__':unittest.main()
