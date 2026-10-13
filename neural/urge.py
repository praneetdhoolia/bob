"""A measurable urge to output, shared by the live runtime and offline evaluation.

The urge U is a leaky accumulator integrated in physical time:

    dU/dt = -U / tau + drive,   drive = readiness + surprise_gain * surprise

* readiness = 1 - P(BLANK) from the translator reading the common state;
* surprise  = how unusual the latest sensory change is, as a nonnegative
  running z-score of ||x_t - x_{t-1}|| for the frozen native packet target.

An utterance may begin only when U >= threshold and the translator proposes a
non-blank code. At that onset the agent releases its *stimulus*: the numerical
readout read(S) that the translator then interprets as bytes. Surprise changes
the hazard of speaking; it cannot trigger speech without a proposed code.

Motivation, not equivalence: leaky stochastic accumulators model the timing of
self-initiated movement (Schurger et al. 2012), neuromodulatory ramps shift the
hazard of acting without triggering it (Hamilos et al. 2021), and streaming
video systems accumulate an informativeness score before responding (MMDuet).
This variable is a functional mechanism; it is not evidence of felt desire.
"""
import math

import torch

from liquid_core import BLANK, BOUNDARY

TAU_SECONDS = .1
THRESHOLD = .05
SURPRISE_GAIN = .1


class UrgeGate:
    def __init__(self, tau=TAU_SECONDS, threshold=THRESHOLD, surprise_gain=SURPRISE_GAIN, enabled=True):
        if tau <= 0 or threshold < 0 or surprise_gain < 0:
            raise ValueError('Urge time constant must be positive; threshold and gain nonnegative.')
        self.tau = tau; self.threshold = threshold; self.surprise_gain = surprise_gain; self.enabled = enabled
        self.urge = 0.; self.readiness = 0.; self.surprise = 0.
        self.speaking = False; self.releases = 0
        self.previous_target = None; self.change_mean = 0.; self.change_square = 0.; self.changes = 0

    def observe_packet(self, target):
        """Update surprise from a frozen native packet target, before assimilation."""
        target = target.detach().float().reshape(-1)
        if self.previous_target is not None:
            change = float((target-self.previous_target).norm())
            if self.changes >= 5:
                variance = max(self.change_square-self.change_mean**2, 1e-12)
                self.surprise = max(0., (change-self.change_mean)/math.sqrt(variance))
            # Slow running statistics: an unusual change stays unusual for a while.
            rate = 1/(self.changes+1) if self.changes < 100 else .01
            self.change_mean += rate*(change-self.change_mean)
            self.change_square += rate*(change*change-self.change_square)
            self.changes += 1
        self.previous_target = target.clone()

    def advance(self, dt, probabilities):
        """Integrate U exactly over dt with the drive held constant."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('Physical time must be finite and nonnegative.')
        self.readiness = float(1-probabilities[BLANK])
        drive = self.readiness + self.surprise_gain*self.surprise
        decay = math.exp(-dt/self.tau)
        self.urge = self.urge*decay + drive*self.tau*(1-decay)
        return self.urge

    def permits(self, code):
        """Whether a proposed code may be emitted now; records onset releases."""
        if code == BLANK or (code == BOUNDARY and not self.speaking):
            return False  # A boundary with nothing said is not an utterance onset.
        if self.speaking or not self.enabled:
            self.speaking = True
            return True
        if self.urge >= self.threshold:
            self.speaking = True; self.releases += 1
            return True
        return False

    def finished(self):
        """An utterance boundary ends speaking; the urge is spent."""
        self.speaking = False; self.urge = 0.

    def measurement(self):
        return dict(urge=self.urge, urgeThreshold=self.threshold, readiness=self.readiness,
                    surprise=self.surprise, speaking=self.speaking, urgeReleases=self.releases)


def stimulus_summary(core, state, probabilities):
    """Compact description of the released stimulus: the readout vector itself
    plus the interpreter's first reading of it. No caption or label is involved."""
    vector = core.read(state)[0].detach().float()
    top = probabilities.topk(5)
    return dict(dimensions=int(vector.numel()), rms=float(vector.square().mean().sqrt()),
                maximum=float(vector.abs().max()),
                projection=vector.reshape(32, -1).mean(-1).tolist(),
                firstReading=[dict(code=int(code), probability=float(p)) for p, code in zip(top.values, top.indices)])
