"""Test whether token-free internal processing adds multi-step computation.

One-hop sensory memory is trained first. Perception, read queries and byte
translation are then frozen. Only the shared state's internal processing
parameters learn the multi-hop task. Graph traversal computes offline labels only.
"""
import argparse
import json
from pathlib import Path
import random
import time

import psutil
import torch
import torch.nn.functional as F

from liquid_core import load_core
from research_resources import research_slot
from literal_stimuli import literal_rows
from train_liquid_core import write_report

ROOT = Path(__file__).resolve().parents[1]


def stimuli(nodes=8):
    rows = literal_rows([str(node) for node in range(nodes)])
    if any(len(row)!=1 for row in rows):
        raise ValueError('This numeric stimulus requires single-token node identifiers.')
    return torch.cat(rows).float()


def graph_cases(count, seed, excluded=None, nodes=8, hops=4):
    rng = random.Random(seed); occupied = set(excluded or []); cases = []
    while len(cases)<count:
        graph = list(range(nodes)); rng.shuffle(graph); signature = tuple(graph)
        if signature in occupied:
            continue
        occupied.add(signature)
        target = []
        for start in range(nodes):
            at = start
            for _ in range(hops):
                at = graph[at]
            target.append(at)
        cases.append(dict(graph=graph, target=target))
    return cases


def hop_baselines(cases, max_hops=3):
    """Answers available without four-hop computation. A four-hop result is only
    evidence of internal processing if it beats every one of these."""
    total = sum(len(case['target']) for case in cases); result = dict(total=total)
    for hops in range(max_hops+1):
        correct = 0
        for case in cases:
            for start, expected in enumerate(case['target']):
                at = start
                for _ in range(hops):
                    at = case['graph'][at]
                correct += at == expected
        result['start' if hops == 0 else f'hop{hops}'] = correct
    result['best'] = max(value for key, value in result.items() if key != 'total')
    return result


def cue(core, vectors, graph):
    # Ordered numerical packets share the native input space. Graph entries
    # arrive together; the next packet supplies each possible starting node.
    state = core.observe(core.start(), vectors[torch.tensor(graph, device=vectors.device)], .1)
    batch = len(graph)
    return core.observe(state.expand(batch,-1,-1,-1), vectors[:,None,:], .1)


def process(core, state, seconds, tick=.01):
    elapsed = 0.
    while elapsed < seconds-1e-8:
        dt = min(tick, seconds-elapsed)
        state = core.advance(state, dt); elapsed += dt
    return state


def assess(core, vectors, cases, seconds=.4, tick=.01, active=True):
    original = core._active_internal; core._active_internal = active
    result = []
    try:
        with torch.inference_mode():
            for case in cases:
                state = process(core, cue(core,vectors,case['graph']), seconds,tick)
                logits = core.logits(state)
                conditional = logits[:,48:56].argmax(-1)
                generated = logits.argmax(-1)-48
                expected = torch.tensor(case['target'],device=vectors.device)
                result.append(dict(correct=int((conditional==expected).sum()),
                    generatedCorrect=int((generated==expected).sum()), total=len(expected),
                    finite=bool(torch.isfinite(state).all())))
    finally:
        core._active_internal = original
    return dict(correct=sum(row['correct'] for row in result),
        generatedCorrect=sum(row['generatedCorrect'] for row in result),
        total=sum(row['total'] for row in result), finite=all(row['finite'] for row in result),
        seconds=seconds,tick=tick,internalWrites=active)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/liquid-core-1791832933421909500/latest.pt')
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.warmup<1 or args.updates<1:
        parser.error('Both training phases require positive update counts.')
    with research_slot('internal computation capacity experiment'):
        torch.set_num_threads(4); torch.manual_seed(1709); random.seed(1709)
        folder = ROOT/'artifacts'/f'liquid-recurrence-{time.time_ns()}'; folder.mkdir()
        receipt = folder/'training.json'
        core, checkpoint = load_core(args.checkpoint,args.device); core.eval()
        core.observation_scheme.fill_(1)
        vectors = stimuli().to(args.device)
        train = graph_cases(1024,1107)
        validation = graph_cases(64,1113,[tuple(row['graph']) for row in train])
        baselines = hop_baselines(validation)
        report = dict(state='one-hop-memory', architecture=core.architecture,
            pretrainedLanguageForwardCalls=0, publishedToLive=False,
            sourceCheckpoint=str(args.checkpoint), trainingGraphs=len(train), validationGraphs=len(validation),
            startsPerGraph=8, requestedHops=4,
            limits=['A controlled relational capacity experiment, not a test of consciousness.',
                'No sensory observations or emitted-code feedback occur during internal processing.',
                'Candidate does not replace the live speech model.'])
        write_report(receipt,report)
        optimizer = torch.optim.AdamW(core.parameters(),lr=.0005)
        started = time.monotonic()
        for update in range(1,args.warmup+1):
            if psutil.virtual_memory().used>50_000_000_000:
                raise MemoryError('Total system RAM exceeds the configured budget.')
            case = random.choice(train); optimizer.zero_grad(set_to_none=True)
            state = cue(core,vectors,case['graph'])
            loss = F.cross_entropy(core.logits(state),torch.tensor(case['graph'],device=args.device)+48)
            loss.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            if update%25==0 or update==args.warmup:
                report.update(update=update,loss=float(loss),elapsedSeconds=time.monotonic()-started)
                write_report(receipt,report); print(json.dumps({key:report[key] for key in ['state','update','loss','elapsedSeconds']}),flush=True)
        one_hop = [dict(row,target=row['graph']) for row in validation]
        report['oneHopValidation'] = assess(core,vectors,one_hop,seconds=0.,active=False)
        torch.save(dict(architecture=core.architecture,state_dict=core.state_dict(),update=args.warmup),folder/'one-hop.pt')
        if report['oneHopValidation']['correct'] < .9*report['oneHopValidation']['total']:
            report.update(state='insufficient-relational-memory',
                reason='Improve one-hop memory before attributing multi-hop failure to internal dynamics.')
            write_report(receipt,report)
            print(json.dumps({'state':report['state'],'oneHopValidation':report['oneHopValidation']}),flush=True)
            return
        # Isolate internal computation from changes in perceptual/English readout.
        for name,parameter in core.named_parameters():
            parameter.requires_grad_(name.startswith('processing_'))
        core.enable_internal_writes()
        optimizer = torch.optim.AdamW([parameter for parameter in core.parameters() if parameter.requires_grad],lr=.001)
        report.update(state='internal-processing',trainableParameters=sum(p.numel() for p in core.parameters() if p.requires_grad),
            initialMultiHop=assess(core,vectors,validation[:16],seconds=.4))
        for update in range(1,args.updates+1):
            if psutil.virtual_memory().used>50_000_000_000:
                raise MemoryError('Total system RAM exceeds the configured budget.')
            case = random.choice(train); optimizer.zero_grad(set_to_none=True)
            state = process(core,cue(core,vectors,case['graph']),random.choice([.3,.4,.5]))
            loss = F.cross_entropy(core.logits(state),torch.tensor(case['target'],device=args.device)+48)
            loss.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            if update%25==0 or update==args.updates:
                report.update(update=update,loss=float(loss),elapsedSeconds=time.monotonic()-started,
                    systemUsedBytes=psutil.virtual_memory().used,
                    internalRateRange=[float(F.softplus(core.processing_rate).min()),float(F.softplus(core.processing_rate).max())])
                if update%100==0 or update==args.updates:
                    report['validation'] = [assess(core,vectors,validation,seconds=t) for t in [0.,.1,.2,.4,.8]]
                    report['internalAblation'] = assess(core,vectors,validation,seconds=.4,active=False)
                    report['structuralBaselines'] = baselines
                temporary = folder/'latest.tmp'
                torch.save(dict(architecture=core.architecture,state_dict=core.state_dict(),update=update,
                    trainingScope='Relational internal-computation capacity, frozen perception/readout'),temporary)
                temporary.replace(folder/'latest.pt')
                write_report(receipt,report); print(json.dumps({key:report[key] for key in ['state','update','loss','elapsedSeconds']}),flush=True)
        report['resolutionValidation'] = [assess(core,vectors,validation,seconds=.4,tick=t) for t in [.005,.02]]
        report['state'] = 'complete'; write_report(receipt,report)


if __name__=='__main__':
    main()
