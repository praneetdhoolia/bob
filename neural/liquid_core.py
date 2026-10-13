"""One numerical state, physical-time recurrence and a learned byte readout.

No pretrained language decoder, prompt or decoder KV cache is used here.
Sensory encoders provide vectors; this module owns all persistent cognition.
"""
import math
from numbers import Real
import torch
from torch import nn
import torch.nn.functional as F

BLANK = 256
BOUNDARY = 257


def delta_write(memory, key, value, strength):
    prediction = torch.einsum('bhvk,bhk->bhv', memory, key)
    return memory + strength[..., None, None] * (value - prediction)[..., None] * key[..., None, :]


def delta_chunk_write(memory, keys, values, strength):
    products = keys @ keys.transpose(-1, -2)
    lower = torch.tril(strength[..., None] * products, diagonal=-1)
    lower = lower + torch.eye(keys.shape[-2], device=keys.device, dtype=keys.dtype)
    residual = strength[..., None] * (values - keys @ memory.transpose(-1, -2))
    corrections = torch.linalg.solve_triangular(lower, residual, upper=False, unitriangular=True)
    return memory + corrections.transpose(-1, -2) @ keys


def delta_flow_write(memory, keys, values, strength, interval):
    """Integrate simultaneous associative writes at fixed learned coefficients.

    dS/dt = B - S A; A is positive semidefinite. An augmented exponential
    integrates even rank-deficient A without inversion or an extra state.
    """
    dt = torch.as_tensor(interval, device=memory.device, dtype=memory.dtype).reshape(-1)
    if dt.numel() == 1:
        dt = dt.expand(memory.shape[0])
    if not torch.isfinite(dt).all() or (dt < 0).any():
        raise ValueError('Physical write interval must be finite and nonnegative.')
    rate = -torch.log1p(-strength.clamp(max=1.-torch.finfo(strength.dtype).eps))/.1
    weighted_keys = keys * rate[..., None]
    a = keys.transpose(-1, -2) @ weighted_keys
    b = values.transpose(-1, -2) @ weighted_keys
    size = memory.shape[-1]
    identity = torch.eye(size, device=memory.device, dtype=memory.dtype).expand_as(a)
    upper = torch.cat((-a, identity), -1)
    generator = torch.cat((upper, torch.zeros_like(upper)), -2)
    exponential = torch.matrix_exp(generator * dt[:, None, None, None])
    retention = exponential[..., :size, :size]
    integral = exponential[..., :size, size:]
    return memory @ retention + b @ integral


class LiquidCore(nn.Module):
    architecture = 'single-liquid-associative-byte-core-v7'
    dynamic_architecture = 'single-liquid-associative-byte-core-v6'
    selective_architecture = 'single-liquid-associative-byte-core-v5'
    simultaneous_architecture = 'single-liquid-associative-byte-core-v4'
    shared_architecture = 'single-liquid-associative-byte-core-v3'
    contextual_architecture = 'single-liquid-associative-byte-core-v2'
    legacy_architecture = 'single-liquid-associative-byte-core-v1'

    def __init__(self):
        super().__init__()
        # Checkpointed architecture configuration, not an extra cognitive state.
        self.register_buffer('observation_scheme', torch.tensor(0, dtype=torch.long))
        self.register_buffer('internal_scheme', torch.tensor(0, dtype=torch.long))
        self._active_internal = False
        self.initial = nn.Parameter(torch.randn(1, 4, 64, 64) * .02)
        self.input_norm = nn.LayerNorm(4096)
        self.input_projection = nn.Linear(4096, 256)
        self.input_position = nn.Embedding(512, 256)
        self.input_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            256, 8, 768, dropout=0., batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        self.time_projection = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, 256))
        self.write_key = nn.Linear(256, 256)
        self.write_value = nn.Linear(256, 256)
        self.write_strength = nn.Linear(256, 4)
        self.read_queries = nn.Parameter(torch.randn(4, 8, 64) * .1)
        self.input_context = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 256), nn.Tanh())
        # One state summary can modulate perception, own-code feedback and
        # internal dynamics. New couplings start disabled for exact migration.
        self.context_gain = nn.Parameter(torch.tensor([1.,0.,0.]))
        # Zero initialization preserves legacy behavior until feedback is trained.
        nn.init.zeros_(self.input_context[1].weight)
        nn.init.zeros_(self.input_context[1].bias)
        self.translator = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 384), nn.GELU(), nn.Linear(384, 258))
        self.emission_embedding = nn.Embedding(258, 64)
        self.emission_write = nn.Sequential(nn.Linear(320, 384), nn.Tanh(), nn.Linear(384, 516))
        self.flow = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 512))
        self.flow_rate = nn.Parameter(torch.tensor([-9., -8., -7., -6.]))
        # Unassigned processing time scales. Disabled on legacy migration.
        self.processing_left = nn.Parameter(torch.eye(64).repeat(4,1,1))
        self.processing_right = nn.Parameter(torch.eye(64).repeat(4,1,1))
        self.processing_alignment = nn.Parameter(torch.zeros(4,64,64))
        self.processing_bias = nn.Parameter(torch.zeros(4,64,64))
        rates = torch.tensor([.02,.2,2.,20.])[:,None,None].expand(4,64,64)
        self.processing_rate = nn.Parameter(torch.log(torch.expm1(rates)).clone())
        self.processing_ceiling = nn.Parameter(torch.log(torch.expm1(torch.tensor([10.,10.,10.,10.]))))
        self.duration_head = nn.Linear(2048, 1)
        # The translator starts silent, not with a fabricated greeting.
        with torch.no_grad():
            self.translator[-1].bias[BLANK] = 3.
            self.flow[-1].weight.mul_(.01)
            self.flow[-1].bias.zero_()

    def start(self, batch=1):
        return self.initial.expand(batch, -1, -1, -1)

    def read(self, state):
        keys = F.normalize(self.read_queries, dim=-1)
        return torch.einsum('bhvk,hqk->bhqv', state, keys).flatten(1)

    def advance(self, state, dt):
        if isinstance(dt, Real):
            if not math.isfinite(dt) or dt < 0 or dt > torch.finfo(state.dtype).max:
                raise ValueError('Physical time must be finite and nonnegative.')
            # Clock scalars are known on the CPU. Validating a CUDA copy every
            # internal tick unnecessarily synchronizes the complete stream.
            interval = float(dt)
        else:
            interval = torch.as_tensor(dt, dtype=state.dtype, device=state.device).reshape(-1)
            if interval.numel() == 1:
                interval = interval.expand(state.shape[0])
            if not torch.isfinite(interval).all() or (interval < 0).any():
                raise ValueError('Physical time must be finite and nonnegative.')
            interval = interval[:, None, None, None]
        # A bounded state-dependent target prevents the old constant-flow drift.
        context = state.mean(-1).flatten(1) + self.context_gain[2]*self.input_context(self.read(state))
        values, keys = self.flow(context).chunk(2, -1)
        values = torch.tanh(values).reshape(-1, 4, 64)
        keys = F.normalize(keys.reshape(-1, 4, 64), dim=-1)
        target = values[..., None] * keys[..., None, :]
        amount = -torch.expm1(-F.softplus(self.flow_rate)[None, :, None, None] * interval)
        evolved = state + amount * (target - state)
        if self._active_internal:
            # Full-rank learned processing of the same state. The bilinear
            # term can compose stored associations without a symbolic traversal.
            features = self.processing_left @ evolved @ self.processing_right
            features = features + evolved @ self.processing_alignment @ evolved + self.processing_bias
            ceiling = F.softplus(self.processing_ceiling)[None,:,None,None]
            processing_target = ceiling*torch.tanh(features/ceiling)
            strength = -torch.expm1(-F.softplus(self.processing_rate)[None]*interval)
            evolved = evolved + strength*(processing_target-evolved)
        return evolved

    def enable_internal_writes(self):
        self.internal_scheme.fill_(3)
        self._active_internal = True
        self.context_gain.data[2] = 1.

    def observe(self, state, tokens, interval=.1, write_mode=None):
        if write_mode is None:
            scheme = int(self.observation_scheme)
            write_mode = {0: 'ordered', 1: 'simultaneous'}.get(scheme)
        if write_mode not in ['ordered', 'simultaneous']:
            raise ValueError('Unknown observation integration method.')
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(device=state.device, dtype=state.dtype)
        if not tokens.shape[1]:
            return state
        if tokens.shape[1] > 512:
            raise ValueError('Input exceeds numerical row capacity.')
        keys, content, strength, _ = self.input_writes(tokens, self.input_context(self.read(state)), interval, write_mode)
        if write_mode == 'simultaneous':
            return delta_flow_write(state, keys, content, strength, interval)
        return delta_chunk_write(state, keys, content, strength)

    def input_writes(self, tokens, context, interval=.1, write_mode='ordered'):
        """The shared input calculation, also used for derivative diagnostics."""
        values = self.input_projection(self.input_norm(tokens))
        values = values + self.context_gain[0]*context[:, None]
        positions = torch.arange(values.shape[1], device=context.device)
        values = self.input_encoder(values + self.input_position(positions)[None])
        dt = torch.as_tensor(interval, dtype=context.dtype, device=context.device).reshape(-1)
        if dt.numel() == 1:
            dt = dt.expand(context.shape[0])
        timing_dt = torch.full_like(dt, .1) if write_mode == 'simultaneous' else dt
        timing = torch.stack((timing_dt, torch.log1p(timing_dt), torch.expm1(-timing_dt)), -1)
        values = values + self.time_projection(timing)[:, None]
        batch, rows = values.shape[:2]
        keys = F.normalize(self.write_key(values).reshape(batch, rows, 4, 64).transpose(1, 2), dim=-1)
        content = torch.tanh(self.write_value(values).reshape(batch, rows, 4, 64).transpose(1, 2))
        strength = torch.sigmoid(self.write_strength(values)).transpose(1, 2)
        return keys, content, strength, values

    def logits(self, state):
        return self.translator(self.read(state))

    def predict_sensory(self, state):
        """Forecast native packet content through the existing shared pathway."""
        return self.input_context(self.read(state))

    def sensory_target(self, tokens):
        """Frozen numerical target, without captions, transcripts or answer labels."""
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if not tokens.shape[1]:
            raise ValueError('A sensory target requires at least one native row.')
        return torch.tanh(self.input_projection(self.input_norm(tokens)).mean(1))

    def feedback(self, state, code):
        codes = torch.as_tensor(code, device=state.device, dtype=torch.long).reshape(-1)
        if codes.numel() == 1:
            codes = codes.expand(state.shape[0])
        context = state.mean(-1).flatten(1) + self.context_gain[1]*self.input_context(self.read(state))
        output = self.emission_write(torch.cat((context, self.emission_embedding(codes)), -1))
        keys = F.normalize(output[:, :256].reshape(-1, 4, 64), dim=-1)
        values = torch.tanh(output[:, 256:512].reshape(-1, 4, 64))
        strength = torch.sigmoid(output[:, 512:]).reshape(-1, 4)
        return delta_write(state, keys, values, strength)

    def seed_sensory_weights(self, checkpoint):
        source = checkpoint['state_dict']
        names = ['input_norm', 'input_projection', 'input_position', 'input_encoder',
                 'time_projection', 'write_key', 'write_value', 'write_strength']
        for name in names:
            prefix = name + '.'
            getattr(self, name).load_state_dict({key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)})
        self.initial.data.copy_(source['initial'].reshape_as(self.initial))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def restore_weights(self, weights, architecture):
        # Older artifacts included a prediction head that was never trained or
        # used. Drop only that complete, known obsolete pair during migration.
        weights = dict(weights)
        obsolete = {'prediction.weight', 'prediction.bias'} & weights.keys()
        if obsolete and len(obsolete) != 2:
            raise ValueError('Incomplete obsolete prediction weights.')
        for name in obsolete:
            weights.pop(name)
        if architecture != self.architecture:
            for name in list(weights):
                if name=='internal_rate' or name.startswith('internal_value.'):
                    weights.pop(name)
        if architecture == self.architecture:
            self.load_state_dict(weights, strict=True)
        elif architecture in [self.legacy_architecture,self.contextual_architecture,self.shared_architecture,self.simultaneous_architecture,self.selective_architecture,self.dynamic_architecture]:
            if architecture in [self.selective_architecture,self.dynamic_architecture] and int(weights['internal_scheme'])!=0:
                raise ValueError('Retired experimental internal flow; use its disabled-flow encoder checkpoint.')
            self.internal_scheme.zero_()
            self.processing_left.data.copy_(torch.eye(64,device=self.processing_left.device).repeat(4,1,1))
            self.processing_right.data.copy_(torch.eye(64,device=self.processing_right.device).repeat(4,1,1))
            self.processing_alignment.data.zero_(); self.processing_bias.data.zero_()
            rates = torch.tensor([.02,.2,2.,20.],device=self.processing_rate.device)[:,None,None].expand(4,64,64)
            self.processing_rate.data.copy_(torch.log(torch.expm1(rates)))
            self.processing_ceiling.data.fill_(math.log(math.expm1(10.)))
            self.observation_scheme.zero_()
            self.context_gain.data.copy_(torch.tensor([1.,0.,0.],device=self.context_gain.device))
            # A legacy checkpoint must not inherit a previously trained context.
            nn.init.zeros_(self.input_context[1].weight)
            nn.init.zeros_(self.input_context[1].bias)
            nn.init.ones_(self.input_context[0].weight)
            nn.init.zeros_(self.input_context[0].bias)
            result = self.load_state_dict(weights, strict=False)
            expected={'processing_left','processing_right','processing_alignment','processing_bias','processing_rate','processing_ceiling'}
            if architecture not in [self.selective_architecture,self.dynamic_architecture]:
                expected.add('internal_scheme')
            if architecture not in [self.simultaneous_architecture,self.selective_architecture,self.dynamic_architecture]:
                expected.add('observation_scheme')
            if architecture in [self.legacy_architecture,self.contextual_architecture]:
                expected.add('context_gain')
            if architecture==self.legacy_architecture:
                expected.update({'input_context.0.weight','input_context.0.bias',
                                 'input_context.1.weight','input_context.1.bias'})
            if set(result.missing_keys) != expected or result.unexpected_keys:
                raise ValueError('Invalid legacy core weights.')
        else:
            raise ValueError('Unknown numerical core architecture.')
        # Static checkpoint configuration; avoid a CUDA synchronization per tick.
        scheme = int(self.internal_scheme)
        if scheme not in [0,3]:
            raise ValueError('Unknown internal computation method.')
        self._active_internal = scheme==3


def load_core(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['architecture'] not in [LiquidCore.architecture,LiquidCore.legacy_architecture,LiquidCore.contextual_architecture,LiquidCore.shared_architecture,LiquidCore.simultaneous_architecture,LiquidCore.selective_architecture,LiquidCore.dynamic_architecture]:
        raise ValueError('This checkpoint is not the single live core.')
    core = LiquidCore().to(device)
    core.restore_weights(checkpoint['state_dict'], checkpoint['architecture'])
    return core, checkpoint
