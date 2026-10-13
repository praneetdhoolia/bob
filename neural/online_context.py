"""Experimental local learning in the core's existing shared context pathway.

There is no extra neural module or recurrent world state. Optimizer moments and
the trust-region anchor are learning machinery, not an input-history cache.
Deployment is deliberately disabled until predictive and retention controls pass.
"""
import torch
import torch.nn.functional as F
from liquid_core import BLANK


class ContextLearner:
    def __init__(self, core, learning_rate=1e-4, max_rms_drift=.01, coupling=.02, train_writes=False, temporal_credit=False):
        if learning_rate <= 0 or max_rms_drift <= 0 or not 0<=coupling<=1:
            raise ValueError('Learning rate and drift bound must be positive.')
        self.core = core
        self.max_rms_drift = max_rms_drift
        self.train_writes = train_writes
        self.temporal_credit = temporal_credit or train_writes
        core.requires_grad_(False)
        core.input_context.requires_grad_(True)
        core.write_strength.requires_grad_(train_writes)
        self.parameters = [parameter for parameter in core.parameters() if parameter.requires_grad]
        self.anchor = [p.detach().clone() for p in self.parameters]
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        # Zero legacy context makes this activation initially behavior preserving.
        # Preserve existing gains on already-trained contextual checkpoints.
        with torch.no_grad():
            if not torch.count_nonzero(core.input_context[1].weight) and not torch.count_nonzero(core.input_context[1].bias):
                # Coupling scales all three context paths, including perception
                # (context_gain[0]); recorded receipts used this convention.
                core.context_gain.fill_(coupling)

    def update(self, prior_state, incoming_tokens, base=None):
        # A caller must capture state BEFORE assimilating this packet. Clones
        # permit use after inference-mode encoding without autograd alias errors.
        with torch.inference_mode(False), torch.enable_grad():
            state = prior_state.clone() if self.temporal_credit else prior_state.detach().clone()
            tokens = incoming_tokens.detach().clone()
            with torch.no_grad():
                target = self.core.sensory_target(tokens)
            prediction = self.core.predict_sensory(state)
            if base is not None:
                # Residual forecast: the state predicts the change from the last packet.
                prediction = prediction + base.detach()
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite predictive learning loss.')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm,rms=self.apply_step()
            return dict(priorPredictionMse=float(loss.detach()),gradientNorm=float(norm),
                        parameterRmsDrift=float(rms))

    def apply_step(self):
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            # A small gate cannot borrow a large context matrix's drift allowance.
            for parameter,anchor in zip(self.parameters,self.anchor):
                delta=parameter-anchor
                factor=(self.max_rms_drift/delta.square().mean().sqrt().clamp_min(1e-30)).clamp(max=1.)
                parameter.copy_(anchor+delta*factor)
            squared=sum((p-a).square().sum() for p,a in zip(self.parameters,self.anchor))
            rms=torch.sqrt(squared/sum(p.numel() for p in self.parameters))
        return norm,rms


def rank_one_credit(state_tangent, parameter_factor, noise, local_parameter_gradient):
    """UORO's rank-two to rank-one reduction; no full Jacobian allocation.

    The sign vector is independently sampled by the caller. Scaling influences
    variance, not the expectation for fixed transition parameters.
    """
    epsilon=1e-7
    vnorm=torch.sqrt(sum(value.square().sum() for value in parameter_factor))
    bnorm=torch.sqrt(sum(value.square().sum() for value in local_parameter_gradient))
    rho0=torch.sqrt(vnorm/(state_tangent.norm()+epsilon))+epsilon
    rho1=torch.sqrt(bnorm/(noise.norm()+epsilon))+epsilon
    next_state=rho0*state_tangent+rho1*noise
    next_parameters=[value/rho0+gradient/rho1 for value,gradient in zip(parameter_factor,local_parameter_gradient)]
    return next_state,next_parameters


class RankOneContextLearner(ContextLearner):
    """Experimental long-range credit, within the same core and sensory loss.

    Implements the rank-one update in Tallec/Ollivier, arXiv:1702.05043v3.
    Random gradient noise, changing weights, clipping and trust bounds mean this
    optimizer cannot claim a convergence theorem for the complete learning rule.
    Discrete emitted codes are held fixed for each transition derivative.
    """
    def __init__(self,*args,reset_each_packet=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset_each_packet=reset_each_packet
        self.reset_credit()

    def reset_credit(self):
        self.state_factor=torch.zeros_like(self.core.start())
        self.parameter_factor=[torch.zeros_like(p) for p in self.parameters]

    def transition(self,state,tokens,interval,code):
        for _ in range(10):state=self.core.advance(state,interval/10)
        state=self.core.observe(state,tokens,interval)
        if code!=BLANK:state=self.core.feedback(state,code)
        return state

    def step(self,state,tokens,interval):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        # Math attention supports the double differentiation used by jvp.
        with sdpa_kernel(SDPBackend.MATH),torch.enable_grad():
            variable=state.detach().requires_grad_(True)
            prior=variable
            for _ in range(10):prior=self.core.advance(prior,interval/10)
            with torch.no_grad():target=self.core.sensory_target(tokens)
            loss=F.mse_loss(self.core.predict_sensory(prior),target)
            gradients=torch.autograd.grad(loss,[variable]+self.parameters,retain_graph=True,allow_unused=True)
            observed=self.core.observe(prior,tokens,interval)
            with torch.no_grad():code=int(self.core.logits(observed).argmax(-1))
            changed=self.core.feedback(observed,code) if code!=BLANK else observed
            noise=torch.empty_like(changed).bernoulli_(.5).mul_(2).sub_(1)
            local=torch.autograd.grad((changed*noise).sum(),self.parameters,allow_unused=True)
            local=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(self.parameters,local)]
            _,tangent=torch.autograd.functional.jvp(
                lambda value:self.transition(value,tokens,interval,code),
                state.detach(),self.state_factor,create_graph=False)
            state_gradient=gradients[0]
            multiplier=(state_gradient*self.state_factor).sum() if state_gradient is not None else loss.new_zeros(())
            self.optimizer.zero_grad(set_to_none=True)
            for p,direct,factor in zip(self.parameters,gradients[1:],self.parameter_factor):
                p.grad=(torch.zeros_like(p) if direct is None else direct.detach())+multiplier.detach()*factor
            if self.reset_each_packet:
                self.reset_credit();tangent.zero_()
            next_u,next_v=rank_one_credit(tangent.detach(),self.parameter_factor,noise,local)
            if not torch.isfinite(next_u).all() or any(not torch.isfinite(value).all() for value in next_v):
                raise FloatingPointError('Non-finite temporal sensitivity.')
            self.state_factor=next_u.detach();self.parameter_factor=[value.detach() for value in next_v]
            norm,rms=self.apply_step()
            return changed.detach(),code,dict(priorPredictionMse=float(loss.detach()),
                gradientNorm=float(norm),parameterRmsDrift=float(rms),
                sensitivityBytes=(self.state_factor.numel()+sum(p.numel() for p in self.parameter_factor))*4)
