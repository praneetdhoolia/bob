"""Offline elapsed-time readout from the existing common numerical state.

No host clock, tick counter or separate recurrent timer reaches the readout.
Physical dt drives the same dynamics used by the live agent. Calibration is
supervision, not a claim of subjective time. Nothing is published to live.
"""
import argparse
import json
from pathlib import Path
import time

import psutil
import torch

from liquid_core import load_core
from train_liquid_core import write_report

ROOT = Path(__file__).resolve().parents[1]


def trace(core, cue, silent, ages, resolution, input_period=None, freeze_time=False, write_mode='ordered'):
    state = cue.expand(len(ages), -1, -1, -1).clone()
    elapsed = torch.zeros_like(ages)
    next_input = input_period
    steps = int(float(ages.max()) / resolution + 1.001)
    with torch.inference_mode():
        for step in range(steps):
            if step % 100 == 0 and psutil.virtual_memory().used > 50_000_000_000:
                raise MemoryError('Total system RAM exceeds the configured budget.')
            dt = (ages - elapsed).clamp(0., resolution)
            state = core.advance(state, dt * (0. if freeze_time else 1.))
            elapsed += dt
            clock = min((step + 1) * resolution, float(ages.max()))
            if next_input is not None and clock + 1e-6 >= next_input:
                # Finished samples remain frozen, including their sensory writes.
                due = ages + 1e-6 >= next_input
                observed = core.observe(state, silent.expand(len(ages), -1, -1), input_period, write_mode=write_mode)
                state = torch.where(due[:, None, None, None], observed, state)
                next_input += input_period
        return core.read(state).clone()


def score(predicted, ages):
    error = predicted - ages
    variance = ((ages - ages.mean()) ** 2).sum()
    return {'samples': len(ages), 'meanAbsoluteErrorSeconds': float(error.abs().mean()),
        'rootMeanSquareErrorSeconds': float(error.square().mean().sqrt()),
        'rSquared': float(1. - error.square().sum() / variance),
        'predictions': [round(float(value), 4) for value in predicted],
        'elapsedSeconds': [round(float(value), 4) for value in ages]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--write-mode', choices=['ordered', 'simultaneous'], default='ordered')
    args = parser.parse_args()
    torch.set_num_threads(4)
    pointer = json.loads((ROOT/'artifacts/liquid-core-training.json').read_text(encoding='utf8'))
    source = args.checkpoint or ROOT/pointer['directory']/'latest.pt'
    core, checkpoint = load_core(source, args.device)
    core.eval()
    folder = ROOT/'artifacts'/f'liquid-time-{time.time_ns()}'
    folder.mkdir()
    receipt = folder/'probe.json'
    report = {'state': 'preparing', 'sourceCheckpoint': str(source),
        'architecture': core.architecture, 'publishedToLive': False,
        'writeMode': args.write_mode,
        'readout': 'Existing duration_head, calibrated linear weights only',
        'limits': ['Offline interval decoding, not subjective time.',
            'No absolute host time or step count is an input to the readout.',
            'No learned event-reset or conversational time estimate is established.']}
    write_report(receipt, report)
    corpus = json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
    cues = {}
    with torch.inference_mode():
        for mode in ['held', 'changing']:
            item = next(row for row in corpus['cases'] if row['split'] == 'train' and row['mode'] == mode)
            case = torch.load(ROOT/item['file'], weights_only=False)
            state = core.start().clone()
            # The first twenty frames precede the scene switch and can be
            # identical. Use states after the actual recorded visual change.
            for event in case['events'][:90]:
                state = core.observe(core.advance(state, event['dt']), event['tokens'].to(args.device).float(), event['dt'])
            tokens = case['events'][90]['tokens'].to(args.device).float()
            cues[mode] = (state.clone(), {
                'quiet': torch.cat([tokens[:1], tokens[-1:]])[None],
                'audiovisual': tokens[None]})
        cue_difference = float((cues['held'][0] - cues['changing'][0]).abs().max())
        if cue_difference == 0.:
            raise ValueError('Recorded cue states must actually differ.')
        report['cueStateMaximumDifference'] = cue_difference
    train_ages = torch.arange(0., 20.01, 1., device=args.device)
    held_ages = torch.arange(.5, 20., 1., device=args.device)
    features = []; targets = []
    started = time.monotonic()
    for name, (cue, inputs) in cues.items():
        for kind, period in [('quiet', None), ('quiet', .1), ('audiovisual', .1)]:
            features.append(trace(core, cue, inputs[kind], train_ages, .05, period,
                                  write_mode=args.write_mode))
            targets.append(train_ages)
            print(json.dumps({'calibratedTrace': name, 'inputKind': kind, 'inputPeriodSeconds': period}), flush=True)
    x = torch.cat(features).double(); y = torch.cat(targets).double()
    center = x.mean(0); scale = x.std(0).clamp_min(.001)
    standardized = (x - center) / scale
    centered_y = y - y.mean()
    # Solve the small dual ridge system. Convert the result back to the actual
    # existing linear head; there is no extra decoder or persistent state.
    alpha = torch.linalg.solve(standardized @ standardized.T + torch.eye(len(y), device=args.device, dtype=torch.float64)*.01, centered_y)
    weight = (standardized.T @ alpha) / scale
    bias = y.mean() - center @ weight
    with torch.no_grad():
        core.duration_head.weight.copy_(weight[None].float())
        core.duration_head.bias.copy_(bias[None].float())
    def evaluate(values):
        with torch.inference_mode():
            return core.duration_head(values).flatten().clone()
    report.update(state='evaluating', train=score(evaluate(torch.cat(features)), torch.cat(targets)))
    write_report(receipt, report)
    report['conditions'] = []
    for name, (cue, inputs) in cues.items():
        for kind, period in [('quiet', None), ('quiet', .05), ('quiet', .1), ('quiet', .2),
                ('audiovisual', .05), ('audiovisual', .1), ('audiovisual', .2)]:
            values = trace(core, cue, inputs[kind], held_ages, .025, period,
                           write_mode=args.write_mode)
            row = dict(cue=name, inputKind=kind, inputPeriodSeconds=period, tickSeconds=.025,
                **score(evaluate(values), held_ages))
            report['conditions'].append(row)
            print(json.dumps({key: row[key] for key in ['cue', 'inputPeriodSeconds', 'meanAbsoluteErrorSeconds', 'rSquared']}), flush=True)
        # Exact fixed-count intervention: one hundred advances for every age.
        state = cue.expand(len(held_ages), -1, -1, -1).clone()
        with torch.inference_mode():
            for _ in range(100):
                state = core.advance(state, held_ages/100)
            report['conditions'].append(dict(cue=name, fixedAdvanceCount=100,
                **score(evaluate(core.read(state)), held_ages)))
        frozen = trace(core, cue, inputs['quiet'], held_ages, .025, None, freeze_time=True)
        report['conditions'].append(dict(cue=name, zeroPhysicalTimeControl=True,
            **score(evaluate(frozen), held_ages)))
        write_report(receipt, report)
    core.observation_scheme.fill_(1 if args.write_mode == 'simultaneous' else 0)
    torch.save({'architecture': core.architecture, 'state_dict': core.state_dict(),
        'trainingScope': 'Offline elapsed-time readout calibration only',
        'experimentalWriteMode': args.write_mode,
        'sourceCheckpoint': str(source)}, folder/'calibrated.pt')
    report.update(state='complete', elapsedSeconds=time.monotonic()-started,
        systemUsedBytes=psutil.virtual_memory().used,
        constantMeanBaseline=score(torch.full_like(held_ages, train_ages.mean()), held_ages))
    write_report(receipt, report)
    print(json.dumps({'receipt': str(receipt), 'state': 'complete'}), flush=True)


if __name__ == '__main__':
    from research_resources import research_slot
    with research_slot('interval readout calibration'):
        main()
