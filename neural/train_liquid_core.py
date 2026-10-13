"""Train the custom translator on actual cached raw sensory recordings.

Targets are bytes in an offline loss, never input captions or runtime rules.
No pretrained English generator is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import psutil
import torch
import torch.nn.functional as F
from liquid_core import LiquidCore, BLANK, BOUNDARY

ROOT = Path(__file__).resolve().parents[1]


def write_report(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def prepare(core, cases, device, recompute=False):
    sequences, idle = [], []
    with torch.inference_mode():
        for item in cases:
            case = torch.load(ROOT / item['file'], map_location='cpu', weights_only=False)
            state = core.start().clone()
            starts = {query['tick']: query for query in case['queries']}
            for tick, event in enumerate(case['events']):
                tokens = event['tokens'].to(device).float()
                state = core.observe(core.advance(state, event['dt']), tokens, event['dt'])
                if tick in starts:
                    query = starts[tick]
                    last = len(case['events']) - 1
                    future = [case['events'][min(tick + offset, last)]['tokens'].half() for offset in range(1, 10)]
                    row = {'state': state.detach().cpu().clone(), 'future': future,
                        'answer': query['answer'], 'kind': query['kind'], 'metadata': case['metadata']}
                    if recompute:
                        row.update(observationPrefix=[event['tokens'] for event in case['events'][:tick]],
                                   queryTokens=case['events'][tick]['tokens'])
                    sequences.append(row)
                elif tick % 4 == 0 and not any(0 < tick - start <= 10 for start in starts):
                    # Silence is supervised only away from the 1 s after a question,
                    # where an answer may legitimately be in progress.
                    idle.append(state.detach().cpu().clone())
    return sequences, idle


def starting_state(core,row,device):
    if 'observationPrefix' not in row:
        return row['state'].to(device).clone()
    state=core.start().clone();prefix=row['observationPrefix']
    # Recompute the whole trace with current weights; limit training's gradient
    # history while retaining all earlier evidence in the one shared matrix.
    with torch.no_grad():
        for tokens in prefix[:-8]:
            state=core.observe(core.advance(state,.1),tokens.to(device).float(),.1)
    for tokens in prefix[-8:]:
        state=core.observe(core.advance(state,.1),tokens.to(device).float(),.1)
    return core.observe(core.advance(state,.1),row['queryTokens'].to(device).float(),.1)


def quiet_states(core, sensory, device):
    """Current-model silence supervision, rather than stale cached states."""
    states = []
    for tokens in sensory:
        tokens = tokens.to(device).float()
        state = core.start().clone()
        length = random.randint(5, 15)
        with torch.no_grad():
            for _ in range(length-3):
                state = core.observe(core.advance(state, .1), tokens, .1)
        for _ in range(3):
            state = core.observe(core.advance(state, .1), tokens, .1)
        states.append(state)
    return torch.cat(states)


def rollout(core, row, device, teacher=False, sampling=0.):
    state = starting_state(core,row,device)
    codes = list(row['answer'].encode('utf8')) + [BOUNDARY]
    predictions, losses = [], []
    period=row.get('inputPeriod',.15);elapsed=0.;next_input=period;input_index=0
    for offset in range(len(codes) + 3):
        state = core.advance(state, .03)
        elapsed+=.03
        due=(offset and offset%5==0) if period==.15 else elapsed+1e-8>=next_input
        if due or offset == row.get('switchOffset', -1):
            selected=max(offset//5-1,0) if period==.15 else input_index
            tokens = row['future'][min(selected,len(row['future'])-1)].to(device).clone()
            if offset >= row.get('switchOffset', 100000):
                replacement = row['replacement'][min(selected,len(row['replacement'])-1)].to(device)
                tokens[2:66] = replacement[2:66]
            state = core.observe(state, tokens, period)
            if due:next_input+=period;input_index+=1
        logits = core.logits(state)
        target = codes[offset] if offset < len(codes) else BLANK
        losses.append(F.cross_entropy(logits, torch.tensor([target], device=device)))
        code = target if teacher and random.random() >= sampling else int(logits.argmax(-1))
        predictions.append(code)
        if code != BLANK:
            state = core.feedback(state, code)
    return torch.stack(losses).mean(), predictions


def evaluate(core, rows, idle, device):
    result = []
    with torch.inference_mode():
        for row in rows:
            _, codes = rollout(core, row, device)
            stop = codes.index(BOUNDARY) if BOUNDARY in codes else len(codes)
            actual = bytes(code for code in codes[:stop] if code < 256).decode('utf8', errors='replace')
            result.append({'kind': row['kind'], 'metadata': row['metadata'], 'expected': row['answer'],
                           'actual': actual, 'correct': actual == row['answer']})
        quiet = [int(core.logits(state.to(device)).argmax(-1)) == BLANK for state in idle[::max(1, len(idle)//100)]]
    return {'correct': sum(row['correct'] for row in result), 'total': len(result),
            'quietCorrect': sum(quiet), 'quietTotal': len(quiet), 'rows': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=800)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--live-switches', action='store_true')
    parser.add_argument('--simultaneous', action='store_true')
    parser.add_argument('--state-feedback', action='store_true')
    parser.add_argument('--input-period', type=float, default=.1)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--corpus', type=Path, default=ROOT/'artifacts/workspace-raw-corpus.json')
    parser.add_argument('--self-reports', action='store_true')
    parser.add_argument('--question-channel', choices=['spoken','text'])
    parser.add_argument('--case-limit', type=int)
    args = parser.parse_args()
    if args.input_period <= 0:
        parser.error('--input-period must be positive')
    if args.self_reports and not args.continuous:
        parser.error('--self-reports requires --continuous')
    if args.case_limit is not None and args.case_limit < 1:
        parser.error('--case-limit must be positive')
    from research_resources import research_slot
    manifest = json.loads(args.corpus.read_text(encoding='utf8'))
    train_items, validation_items = selected_cases(manifest, args)
    data_bytes = sum((ROOT/row['file']).stat().st_size for row in train_items+validation_items)
    if args.live_switches and args.continuous:
        # Conservative upper bound: two complete visual-intervention copies.
        data_bytes += 2*sum((ROOT/row['file']).stat().st_size for row in train_items)
    with research_slot('core training', reserve_bytes=data_bytes+3_000_000_000):
        train(args)


def selected_cases(corpus, args):
    cases = corpus['cases']
    if args.question_channel:
        cases = [row for row in cases if row.get('questionChannel')==args.question_channel]
    training = [row for row in cases if row['split']=='train']
    validation = [row for row in cases if row['split']!='train']
    if args.case_limit:
        training = training[:args.case_limit]; validation = validation[:args.case_limit]
    if not training or not validation:
        raise ValueError('Experiment requires both training and validation cases.')
    return training, validation


def train(args):
    torch.set_num_threads(4); torch.manual_seed(701); random.seed(701)
    root = ROOT / 'artifacts' / f'liquid-core-{time.time_ns()}'
    root.mkdir()
    # Experiments must not replace the running agent with an unfinished fit.
    receipt = root / 'training.json'
    report = {'state': 'preparing', 'directory': str(root.relative_to(ROOT)),
              'architecture': LiquidCore.architecture, 'pretrainedLanguageForwardCalls': 0,
              'publishedToLive': False, 'simultaneousInputIntegration': args.simultaneous,
              'stateConditionedInput': args.state_feedback,
              'inputPeriodSeconds': args.input_period,
              'recomputedQuietStates': args.state_feedback,
              'completeInteractionTraining': args.continuous,
              'corpus': str(args.corpus),
              'ownCodeFeedbackExperiment': args.self_reports,
              'questionChannel': args.question_channel, 'caseLimit': args.case_limit,
              'limits': ['Custom translator trained only on narrow recorded examples.',
                         'Numerical recurrence does not establish subjective experience.',
                         'Motor policy and general language remain untrained.']}
    write_report(receipt, report)
    try:
        source = ROOT / 'artifacts/continuous-workspace-1791807822052264200/latest.pt'
        seed = torch.load(source, map_location='cpu', weights_only=False)
        core = LiquidCore().to(args.device)
        core.seed_sensory_weights(seed)
        resumed = None
        if args.resume:
            resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
            core.restore_weights(resumed['state_dict'],resumed['architecture'])
            report['resumedFrom'] = str(args.resume)
        if args.simultaneous:
            core.observation_scheme.fill_(1)
        # Retain trained raw-sensory writes; new translation/feedback/flow learn.
        for name, parameter in core.named_parameters():
            if name.startswith(('input_', 'write_', 'time_projection', 'initial')):
                parameter.requires_grad_(False)
            if args.state_feedback and name.startswith('input_context'):
                parameter.requires_grad_(True)
        corpus = json.loads(args.corpus.read_text(encoding='utf8'))
        if corpus.get('state') != 'complete':
            raise ValueError('Finish building the sensory corpus before training.')
        train_items, validation_items = selected_cases(corpus, args)
        if args.continuous:
            from liquid_stream_training import examples, rollout as stream_rollout, evaluate as stream_evaluate
            train = examples(train_items, args.live_switches)
            validation = examples(validation_items)
            if args.self_reports:
                from liquid_stream_training import self_report_examples
                train = self_report_examples(train)
                validation = self_report_examples(validation)
            idle = []; validation_idle = []
        else:
            train, idle = prepare(core, train_items, args.device,
                                  recompute=args.state_feedback)
            validation, validation_idle = prepare(core, validation_items, args.device,
                                                recompute=args.state_feedback)
        for row in train + validation:
            row['inputPeriod'] = args.input_period
        quiet_sensory = [] if args.continuous else [next(row for row in train if row['metadata']['mode']==mode)['future'][0]
                                                   for mode in ['held','changing']]
        if args.live_switches and not args.continuous:
            visual = [row for row in train if row['kind']=='video']
            representatives = {mode:next(row for row in visual if row['metadata']['mode']==mode) for mode in ['changing','held']}
            for row in visual:
                for mode in ['changing','held']:
                    extra = dict(row)
                    extra.update(switchOffset=len('A person is '),replacement=representatives[mode]['future'],
                        answer='A person is riding a bicycle.' if mode=='changing' else 'A person is playing tennis.')
                    train.append(extra)
            report['matchedVisualSwitchSupervision'] = len(visual)*2
        # Materialize normal tensors outside inference mode before differentiation.
        for row in train + validation:
            if 'state' in row:
                row['state'] = row['state'].clone()
        idle = [state.clone() for state in idle]
        validation_idle = [state.clone() for state in validation_idle]
        optimizer = torch.optim.AdamW([parameter for parameter in core.parameters() if parameter.requires_grad], lr=.0005)
        optimizer_names = [name for name,parameter in core.named_parameters() if parameter.requires_grad]
        configuration = dict(continuous=args.continuous, simultaneous=args.simultaneous,
            stateFeedback=args.state_feedback, liveSwitches=args.live_switches, selfReports=args.self_reports,
            questionChannel=args.question_channel, caseLimit=args.case_limit, inputPeriod=args.input_period,
            corpusSha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest())
        restored_optimizer = bool(resumed and resumed.get('trainingConfiguration')==configuration
            and resumed.get('optimizerParameterNames')==optimizer_names and 'optimizerState' in resumed)
        if restored_optimizer:
            optimizer.load_state_dict(resumed['optimizerState'])
            random.setstate(resumed['pythonRandomState'])
            torch.set_rng_state(resumed['torchRandomState'].cpu())
            if args.device.startswith('cuda') and resumed.get('cudaRandomState') is not None:
                torch.cuda.set_rng_state_all(resumed['cudaRandomState'])
        report.update(optimizerRestored=restored_optimizer, trainingConfiguration=configuration)
        report.update(state='training', parameters=core.parameter_count(), seedSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      trainSequences=len(train), validationSequences=len(validation), update=0)
        write_report(receipt, report)
        started = time.monotonic()
        for update in range(1, args.updates + 1):
            if psutil.virtual_memory().used > 50_000_000_000:
                raise MemoryError('Total system RAM exceeds the configured budget.')
            core.train(); optimizer.zero_grad(set_to_none=True)
            row = random.choice(train)
            if args.continuous:
                total, _, _, _ = stream_rollout(core, row, args.device, teacher=True, sampling=.15)
            else:
                loss, _ = rollout(core, row, args.device, teacher=True, sampling=.15 if args.live_switches else 0.)
                quiet = quiet_states(core, quiet_sensory, args.device) if args.state_feedback else torch.cat(random.sample(idle, min(8, len(idle)))).to(args.device)
                quiet_loss = F.cross_entropy(core.logits(quiet), torch.full((len(quiet),), BLANK, device=args.device))
                total = loss + quiet_loss*.3
            total.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if update % 25 == 0 or update == args.updates:
                checkpoint = {'architecture':core.architecture,'state_dict':core.state_dict(), 'update':update,
                              'trainingScope':'raw cached recordings; custom byte translator', 'seed':str(source.relative_to(ROOT)),
                              'optimizerState':optimizer.state_dict(), 'optimizerParameterNames':optimizer_names,
                              'trainingConfiguration':configuration, 'pythonRandomState':random.getstate(),
                              'torchRandomState':torch.get_rng_state(),
                              'cudaRandomState':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None}
                temporary = root / 'latest.tmp'; torch.save(checkpoint, temporary); temporary.replace(root/'latest.pt')
                report.update(update=update,loss=float(total), elapsedSeconds=time.monotonic()-started)
                if update % 200 == 0 or update == args.updates:
                    core.eval()
                    report['train'] = stream_evaluate(core, train, args.device) if args.continuous else evaluate(core, train, idle, args.device)
                    report['validation'] = stream_evaluate(core, validation, args.device) if args.continuous else evaluate(core, validation, validation_idle, args.device)
                write_report(receipt, report)
                metric = 'exactUtterances' if args.continuous else 'correct'
                print(json.dumps({'update':update,'loss':report['loss'],'elapsed':report['elapsedSeconds'],
                                  'train':report.get('train',{}).get(metric),'validation':report.get('validation',{}).get(metric)}),flush=True)
        report['state']='complete';write_report(receipt,report)
    except Exception as error:
        report.update(state='failed',error=str(error));write_report(receipt,report)
        raise


if __name__ == '__main__':
    main()
