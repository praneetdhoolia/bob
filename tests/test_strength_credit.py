"""Independent autodiff oracles for coupled shared-state derivatives."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from liquid_core import LiquidCore
from strength_credit import StrengthCredit


class StrengthCreditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(204)
        self.core=LiquidCore().double().eval().requires_grad_(False)
        with torch.no_grad():
            self.core.input_context[1].weight.normal_(0,.003)
            self.core.context_gain.fill_(.2);self.core.flow_rate.fill_(.5)
        self.state=torch.randn(1,4,64,64,dtype=torch.float64)*.05
        self.directions=torch.randn(4,64,64,2,dtype=torch.float64)
        self.credit=StrengthCredit(self.core,chunk_size=32)

    def oracle(self,function):
        return torch.stack([torch.autograd.functional.jvp(function,self.state,self.directions[...,index][None])[1][0] for index in range(2)],-1)

    def test_flow_and_feedback_match_autodiff_with_dense_context_coupling(self):
        for operation,reference in [
            (lambda:self.credit.advance(self.state,self.directions,.03),lambda state:self.core.advance(state,.03)),
            (lambda:self.credit.feedback(self.state,self.directions,65),lambda state:self.core.feedback(state,65))]:
            state,derivative=operation()
            self.assertTrue(torch.allclose(state,reference(self.state),atol=1e-12,rtol=1e-12))
            self.assertTrue(torch.allclose(derivative,self.oracle(reference),atol=1e-9,rtol=1e-9))

    def test_observation_state_and_parameter_terms_match_autodiff(self):
        tokens=torch.randn(1,3,4096,dtype=torch.float64)
        old=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        from torch.nn.attention import sdpa_kernel,SDPBackend
        try:
            with sdpa_kernel(SDPBackend.MATH):
                _,derivative=self.credit.observe(self.state,self.directions,tokens,.1,False)
                expected=self.oracle(lambda state:self.core.observe(state,tokens,.1))
                self.assertTrue(torch.allclose(derivative,expected,atol=1e-8,rtol=1e-8))
                zero=torch.zeros(4,64,64,1028,dtype=torch.float64)
                _,local=self.credit.observe(self.state,zero,tokens,.1)
                self.core.write_strength.requires_grad_(True)
                noise=torch.randn_like(self.state)
                result=self.core.observe(self.state,tokens,.1)
                actual=torch.autograd.grad((result*noise).sum(),list(self.core.write_strength.parameters()))
                expected=torch.cat([value.flatten() for value in actual])
                contracted=(local*noise[0,...,None]).sum((0,1,2))
                self.assertTrue(torch.allclose(contracted,expected,atol=1e-8,rtol=1e-8))
        finally:torch.backends.mha.set_fastpath_enabled(old)

    def test_multiple_packets_and_own_codes_match_unrolled_gradient(self):
        core=self.core;core.write_strength.requires_grad_(True)
        tokens=[torch.randn(1,2,4096,dtype=torch.float64) for _ in range(2)]
        state=self.state.clone();eligibility=torch.zeros(4,64,64,1028,dtype=torch.float64)
        reference=self.state.clone()
        old=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        from torch.nn.attention import sdpa_kernel,SDPBackend
        try:
            with sdpa_kernel(SDPBackend.MATH):
                for packet,code in zip(tokens,[65,66]):
                    state,eligibility=self.credit.advance(state,eligibility,.02)
                    state,eligibility=self.credit.observe(state,eligibility,packet,.1)
                    state,eligibility=self.credit.feedback(state,eligibility,code)
                    reference=core.feedback(core.observe(core.advance(reference,.02),packet,.1),code)
                noise=torch.randn_like(reference)
                actual=torch.autograd.grad((reference*noise).sum(),list(core.write_strength.parameters()))
                expected=torch.cat([value.flatten() for value in actual])
                measured=(eligibility*noise[0,...,None]).sum((0,1,2))
                self.assertTrue(torch.allclose(state,reference,atol=1e-11,rtol=1e-11))
                self.assertTrue(torch.allclose(measured,expected,atol=1e-8,rtol=1e-8))
        finally:torch.backends.mha.set_fastpath_enabled(old)


if __name__=='__main__':unittest.main()
