"""The urge accumulator: exact physical-time integration and onset gating."""
from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from liquid_core import BLANK,BOUNDARY
from urge import UrgeGate


def readout(blank):
    probabilities=torch.full((258,),(1-blank)/257);probabilities[BLANK]=blank
    return probabilities


class UrgeTests(unittest.TestCase):
    def test_integration_is_exact_across_subdivided_intervals(self):
        whole,split=UrgeGate(),UrgeGate()
        whole.advance(.06,readout(.2))
        split.advance(.03,readout(.2));split.advance(.03,readout(.2))
        self.assertAlmostEqual(whole.urge,split.urge,places=12)
        self.assertAlmostEqual(whole.urge,.8*.1*(1-math.exp(-.6)),places=7)  # float32 readout
        unchanged=whole.urge;whole.advance(0.,readout(.0))
        self.assertEqual(whole.urge,unchanged)

    def test_onset_needs_both_a_code_and_sufficient_urge(self):
        gate=UrgeGate()
        gate.advance(.03,readout(0.))
        self.assertLess(gate.urge,gate.threshold);self.assertFalse(gate.permits(ord('A')))
        gate.advance(.06,readout(0.))
        self.assertGreaterEqual(gate.urge,gate.threshold)
        self.assertFalse(gate.permits(BLANK));self.assertFalse(gate.permits(BOUNDARY))
        self.assertTrue(gate.permits(ord('A')));self.assertEqual(gate.releases,1)
        # Once speaking, the utterance continues without re-crossing the threshold.
        gate.urge=0.;self.assertTrue(gate.permits(ord('b')));self.assertTrue(gate.permits(BOUNDARY))
        gate.finished();self.assertFalse(gate.speaking);self.assertEqual(gate.urge,0.)

    def test_surprise_raises_urge_but_cannot_speak_alone(self):
        calm,startled=UrgeGate(),UrgeGate()
        for gate in [calm,startled]:
            for index in range(20):gate.observe_packet(torch.full((256,),.01*(index%2)))
        startled.observe_packet(torch.full((256,),3.))
        self.assertGreater(startled.surprise,calm.surprise)
        for gate in [calm,startled]:gate.advance(.05,readout(.9))
        self.assertGreater(startled.urge,calm.urge)
        self.assertFalse(startled.permits(BLANK))

    def test_disabled_gate_reproduces_ungated_emission(self):
        gate=UrgeGate(enabled=False)
        self.assertTrue(gate.permits(BOUNDARY));self.assertFalse(gate.speaking)
        self.assertTrue(gate.permits(ord('A')));self.assertTrue(gate.permits(BOUNDARY))
        self.assertFalse(gate.permits(BLANK));self.assertEqual(gate.releases,0)


if __name__=='__main__':unittest.main()
