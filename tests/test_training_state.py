"""Optimizer plus random state continuation matches an uninterrupted trajectory."""
import copy
import random
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from training_state import capture_training,restore_training


class TrainingStateTests(unittest.TestCase):
    def test_exact_resumption_and_changed_config_fallback(self):
        torch.manual_seed(14);random.seed(15)
        model=torch.nn.Linear(3,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        names=[name for name,_ in model.named_parameters()];configuration={'task':'fixture'}
        def step(current,opt):
            opt.zero_grad();loss=(current(torch.randn(4,3))-random.random()).square().mean()
            loss.backward();opt.step()
        step(model,optimizer)
        weights=copy.deepcopy(model.state_dict())
        state=copy.deepcopy(capture_training(optimizer,names,configuration,'cpu'))
        step(model,optimizer);expected=copy.deepcopy(model.state_dict())
        resumed=torch.nn.Linear(3,1);resumed.load_state_dict(weights)
        resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=.01)
        self.assertFalse(restore_training(state,resumed_optimizer,names,{'task':'changed'},'cpu'))
        self.assertTrue(restore_training(state,resumed_optimizer,names,configuration,'cpu'))
        step(resumed,resumed_optimizer)
        for name,value in expected.items():self.assertTrue(torch.equal(value,resumed.state_dict()[name]))


if __name__=='__main__':unittest.main()
