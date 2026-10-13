"""Structured temporal derivatives for the existing shared numerical state.

Only the 1,028 sensory write-strength parameters participate. Other weights
must stay frozen. No input history, second world state or semantic slots are
retained. Dense cross-coordinate effects through shared context are preserved.
Supports one state, ordered writes and the accepted disabled processing scheme.
"""
import torch
import torch.nn.functional as F
from liquid_core import delta_chunk_write,delta_write
from liquid_core import BLANK
from online_context import ContextLearner


def normalized_derivative(raw, derivative):
    norm=raw.norm(dim=1,keepdim=True)
    key=F.normalize(raw,dim=1)
    projected=derivative-key[:,:,None]*(key[:,:,None]*derivative).sum(1,keepdim=True)
    return torch.where((norm>=1e-12)[:,:,None],projected/norm.clamp_min(1e-12)[:,:,None],derivative/1e-12)


class StrengthCredit:
    parameters=1028

    def __init__(self,core,chunk_size=16):
        if core._active_internal or int(core.observation_scheme)!=0:
            raise ValueError('Strength credit supports ordered writes and accepted inactive processing only.')
        self.core=core;self.chunk_size=chunk_size

    def context(self,state,eligibility):
        core=self.core;read=core.read(state)[0];context=core.input_context(read[None])[0]
        layer=core.input_context[0];linear=core.input_context[1]
        normalized=(read-read.mean())/torch.sqrt(read.var(unbiased=False)+layer.eps)
        weighted=linear.weight*layer.weight[None]/torch.sqrt(read.var(unbiased=False)+layer.eps)
        jacobian=weighted-weighted.mean(1,keepdim=True)-(weighted@normalized)[:,None]*normalized[None]/read.numel()
        jacobian=jacobian*(1-context.square())[:,None]
        queries=F.normalize(core.read_queries,dim=-1)
        projected=torch.einsum('hvkp,hqk->hqvp',eligibility,queries).reshape(2048,-1)
        return context,jacobian@projected

    @torch.no_grad()
    def advance(self,state,eligibility,interval):
        core=self.core;context,context_credit=self.context(state,eligibility)
        mean=state[0].mean(-1).flatten()
        source=mean+core.context_gain[2]*context
        hidden=torch.tanh(core.flow[0](source));output=core.flow[2](hidden)
        derivative=core.flow[2].weight@((1-hidden.square())[:,None]*core.flow[0].weight)
        values=torch.tanh(output[:256]).reshape(4,64);raw_keys=output[256:].reshape(4,64)
        keys=F.normalize(raw_keys,dim=-1)
        dv=(1-values.square())[:,:,None]*derivative[:256].reshape(4,64,256)
        dk=normalized_derivative(raw_keys,derivative[256:].reshape(4,64,256))
        target_derivative=(dv[:,:,None,:]*keys[:,None,:,None]+values[:,:,None,None]*dk[:,None,:,:]).reshape(16384,256)
        source_credit=eligibility.mean(2).reshape(256,-1)+core.context_gain[2]*context_credit
        amount=-torch.expm1(-F.softplus(core.flow_rate)*interval)
        target_credit=(target_derivative@source_credit).reshape_as(eligibility)
        changed=(1-amount)[:,None,None,None]*eligibility+amount[:,None,None,None]*target_credit
        return core.advance(state,interval),changed

    def observation_context_jacobian(self,state,tokens,context,interval):
        core=self.core
        def operation(value):
            keys,values,strength,_=core.input_writes(tokens,value,interval,'ordered')
            return delta_chunk_write(state,keys,values,strength)
        # Disable inference-only transformer kernels: forward AD needs math SDPA.
        fastpath=torch.backends.mha.get_fastpath_enabled()
        from torch.nn.attention import sdpa_kernel,SDPBackend
        parts=[]
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            with torch.inference_mode(False),torch.enable_grad(),sdpa_kernel(SDPBackend.MATH):
                context=context.detach().clone()[None];tokens=tokens.detach().clone();state=state.detach().clone()
                identity=torch.eye(256,device=context.device,dtype=context.dtype)
                for start in range(0,256,self.chunk_size):
                    basis=identity[start:start+self.chunk_size,None]
                    partial=torch.func.vmap(lambda direction:torch.func.jvp(operation,(context,),(direction,))[1])(basis)
                    parts.append(partial.detach().reshape(len(basis),16384))
        finally:torch.backends.mha.set_fastpath_enabled(fastpath)
        return torch.cat(parts).T.contiguous()

    @torch.no_grad()
    def observe(self,state,eligibility,tokens,interval,include_parameter_term=True):
        core=self.core
        if state.shape!=(1,4,64,64):raise ValueError('Strength credit requires one shared state.')
        if tokens.ndim==2:tokens=tokens[None]
        context,context_credit=self.context(state,eligibility)
        keys,values,strength,features=core.input_writes(tokens,context[None],interval,'ordered')
        retention=torch.eye(64,device=state.device,dtype=state.dtype).repeat(4,1,1)
        local=torch.zeros(4,64,64,257,device=state.device,dtype=state.dtype) if include_parameter_term else None
        intermediate=state[0]
        for index in range(tokens.shape[1]):
            key=keys[0,:,index];value=values[0,:,index];rate=strength[0,:,index]
            residual=value-torch.einsum('hvk,hk->hv',intermediate,key)
            product=torch.einsum('hvk,hk->hv',retention,key)
            retention=retention-rate[:,None,None]*product[:,:,None]*key[:,None,:]
            if local is not None:
                prediction=torch.einsum('hvkp,hk->hvp',local,key)
                local=local-rate[:,None,None,None]*prediction[:,:,None,:]*key[:,None,:,None]
                basis=torch.cat((features[0,index],features.new_ones(1)))
                dr=rate[:,None]*(1-rate[:,None])*basis[None]
                local+=residual[:,:,None,None]*key[:,None,:,None]*dr[:,None,None,:]
            intermediate=delta_write(intermediate[None],key[None],value[None],rate[None])[0]
        changed=torch.einsum('hvkp,hkj->hvjp',eligibility,retention)
        if torch.count_nonzero(context_credit):
            jacobian=self.observation_context_jacobian(state,tokens,context,interval)
            changed+=(jacobian@context_credit).reshape_as(changed)
        if local is not None:
            if eligibility.shape[-1]!=self.parameters:raise ValueError('Unexpected write-strength parameter count.')
            for head in range(4):
                changed[head,...,head*256:(head+1)*256]+=local[head,...,:256]
                changed[head,...,1024+head]+=local[head,...,256]
        return core.observe(state,tokens,interval),changed

    @torch.no_grad()
    def feedback(self,state,eligibility,code):
        core=self.core;context,context_credit=self.context(state,eligibility)
        source=state[0].mean(-1).flatten()+core.context_gain[1]*context
        code=torch.as_tensor([code],device=state.device,dtype=torch.long)
        joined=torch.cat((source,core.emission_embedding(code)[0]))
        hidden=torch.tanh(core.emission_write[0](joined));output=core.emission_write[2](hidden)
        derivative=core.emission_write[2].weight@((1-hidden.square())[:,None]*core.emission_write[0].weight[:,:256])
        raw_keys=output[:256].reshape(4,64);keys=F.normalize(raw_keys,dim=1)
        values=torch.tanh(output[256:512]).reshape(4,64);rates=torch.sigmoid(output[512:])
        dk=normalized_derivative(raw_keys,derivative[:256].reshape(4,64,256))
        dv=(1-values.square())[:,:,None]*derivative[256:512].reshape(4,64,256)
        dr=rates[:,None]*(1-rates[:,None])*derivative[512:]
        residual=values-torch.einsum('hvk,hk->hv',state[0],keys)
        dres=dv-torch.einsum('hvk,hkc->hvc',state[0],dk)
        jacobian=(dr[:,None,None,:]*residual[:,:,None,None]*keys[:,None,:,None]
            +rates[:,None,None,None]*(dres[:,:,None,:]*keys[:,None,:,None]+residual[:,:,None,None]*dk[:,None,:,:]))
        source_credit=eligibility.mean(2).reshape(256,-1)+core.context_gain[1]*context_credit
        direct=eligibility-rates[:,None,None,None]*torch.einsum('hvkp,hk->hvp',eligibility,keys)[:,:,None,:]*keys[:,None,:,None]
        changed=direct+(jacobian.reshape(16384,256)@source_credit).reshape_as(eligibility)
        return core.feedback(state,code),changed


class ExactStrengthLearner(ContextLearner):
    """Full state sensitivity for strength weights; every other weight freezes.

    Derivatives match fixed-weight autodiff conditional on chosen discrete codes.
    Online parameter changes and trust projection still limit convergence claims.
    The table is learning machinery, never a second cognitive state or replay.
    """
    def __init__(self,core,learning_rate=1e-4,max_rms_drift=.01,chunk_size=64,reset_each_packet=False):
        if not torch.count_nonzero(core.input_context[1].weight):
            raise ValueError('Fit a state-dependent sensory predictor before strength-only learning.')
        super().__init__(core,learning_rate,max_rms_drift,train_writes=True)
        core.input_context.requires_grad_(False)
        self.parameters=list(core.write_strength.parameters())
        self.anchor=[parameter.detach().clone() for parameter in self.parameters]
        self.optimizer=torch.optim.Adam(self.parameters,lr=learning_rate)
        self.derivative=StrengthCredit(core,chunk_size)
        self.reset_each_packet=reset_each_packet
        self.reset_credit()

    def reset_credit(self):
        self.eligibility=self.core.initial.new_zeros(4,64,64,1028)

    @torch.no_grad()
    def step(self,state,tokens,interval):
        eligibility=self.eligibility;prior=state.detach()
        for _ in range(10):prior,eligibility=self.derivative.advance(prior,eligibility,interval/10)
        prediction,sensitivity=self.derivative.context(prior,eligibility)
        target=self.core.sensory_target(tokens)[0]
        loss=(prediction-target).square().mean()
        gradient=2*(prediction-target)@sensitivity/prediction.numel()
        if self.reset_each_packet:eligibility=torch.zeros_like(eligibility)
        changed,eligibility=self.derivative.observe(prior,eligibility,tokens,interval)
        code=int(self.core.logits(changed).argmax(-1))
        if code!=BLANK:changed,eligibility=self.derivative.feedback(changed,eligibility,code)
        if not torch.isfinite(eligibility).all():raise FloatingPointError('Non-finite strength sensitivity.')
        self.eligibility=eligibility.detach()
        self.optimizer.zero_grad(set_to_none=True)
        self.parameters[0].grad=gradient[:1024].reshape_as(self.parameters[0]).clone()
        self.parameters[1].grad=gradient[1024:].clone()
        norm,rms=self.apply_step()
        return changed.detach(),code,dict(priorPredictionMse=float(loss),gradientNorm=float(norm),
            parameterRmsDrift=float(rms),sensitivityBytes=eligibility.numel()*eligibility.element_size())
