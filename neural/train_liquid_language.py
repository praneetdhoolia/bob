"""Human language supervision for the existing continuously observed byte core.

Loads only a frozen input embedding table, never a pretrained language decoder.
Candidate weights and receipts stay separate from the live checkpoint pointer.
"""
import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import psutil
from safetensors import safe_open
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from liquid_core import BLANK, BOUNDARY, load_core
from train_liquid_core import prepare, rollout, starting_state, quiet_states, write_report

ROOT = Path(__file__).resolve().parents[1]


def language_pairs(max_bytes):
    path = ROOT/'artifacts/english-corpus/2023-04-12_oasst_ready.messages.jsonl.gz'
    with gzip.open(path, 'rt', encoding='utf8') as source:
        messages = [json.loads(line) for line in source]
    lookup = {row['message_id']: row for row in messages}
    pairs = []
    for reply in messages:
        parent = lookup.get(reply.get('parent_id'))
        if not parent or parent.get('parent_id') is not None:
            continue
        if reply.get('role') != 'assistant' or reply.get('rank') != 0:
            continue
        if any(row.get('lang') != 'en' or row.get('deleted') or not row.get('review_result') for row in [parent, reply]):
            continue
        if not 2 <= len(reply['text'].encode('utf8')) <= max_bytes or len(parent['text'].encode('utf8')) > 400:
            continue
        tree = reply['message_tree_id']
        split = 'validation' if int(hashlib.sha256(tree.encode()).hexdigest()[:8], 16) % 5 == 0 else 'train'
        pairs.append({'input': parent['text'], 'answer': reply['text'], 'tree': tree, 'split': split})
    return pairs


def backgrounds(corpus):
    representative = {}
    for mode in ['held', 'changing']:
        item = next(row for row in corpus['cases'] if row['split'] == 'train' and row['mode'] == mode)
        case = torch.load(ROOT/item['file'], map_location='cpu', weights_only=False)
        representative['tennis' if mode == 'held' else 'cycling'] = case['events'][90]['tokens'].clone()
    # Unit marker plus actual silent acoustic row; omit the image marker/queries.
    tokens = representative['tennis']
    representative['quiet'] = torch.cat([tokens[:1], tokens[-1:]])
    return representative


def prepare_language(core, pairs, sensory, device):
    path = ROOT/'.runtime/minicpm-omni'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    index = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    key = 'llm.model.embed_tokens.weight'
    with safe_open(path/index[key], framework='pt', device='cpu') as source:
        embedding = source.get_tensor(key)
        prepared = []
        with torch.inference_mode():
            for number, pair in enumerate(pairs):
                ids = tokenizer.encode(pair['input'], add_special_tokens=False)
                if not ids or len(ids) > 128:
                    continue
                text = embedding[torch.tensor(ids)].to(device).float()
                for mode, background in sensory.items():
                    base = background.to(device).float()
                    state = core.start().clone()
                    prefix_length = 5 + pair.get('preparationIndex', number) % 11
                    for _ in range(prefix_length):
                        state = core.observe(core.advance(state, .1), base, .1)
                    state = core.observe(core.advance(state, .1), torch.cat([base, text]), .1)
                    prepared.append(dict(pair, kind='language', state=state.detach().cpu().clone(),
                        observationPrefix=[background]*prefix_length,
                        queryTokens=torch.cat([base,text]).detach().cpu().half(),
                        future=[background], inputPeriod=.1, metadata={'tree': pair['tree'], 'background': mode}))
                if number % 50 == 0:
                    print(json.dumps({'preparingPairs': number, 'totalPairs': len(pairs)}), flush=True)
    # Inference tensors must become normal tensors before backpropagation.
    for row in prepared:
        row['state'] = row['state'].clone()
        row['queryTokens'] = row['queryTokens'].clone()
    return prepared


def batched_rollout(core, rows, device, sampling=.05):
    state = torch.cat([starting_state(core,row,device) for row in rows])
    answers = [list(row['answer'].encode('utf8')) + [BOUNDARY] for row in rows]
    elapsed = 0.; next_input = .1; losses = []
    for offset in range(max(map(len, answers)) + 3):
        state = core.advance(state, .03); elapsed += .03
        if elapsed + 1e-8 >= next_input:
            parts = []
            for i, row in enumerate(rows):
                parts.append(core.observe(state[i:i+1], row['future'][0].to(device).float(), .1))
            state = torch.cat(parts); next_input += .1
        target = torch.tensor([answer[offset] if offset < len(answer) else BLANK for answer in answers], device=device)
        logits = core.logits(state)
        # Do not let short replies contribute hundreds of redundant blanks.
        relevant = torch.tensor([offset < len(answer) + 3 for answer in answers], device=device)
        losses.append(F.cross_entropy(logits[relevant], target[relevant]))
        code = logits.argmax(-1).detach() if random.random() < sampling else target
        changed = core.feedback(state, code)
        state = torch.where((code != BLANK)[:, None, None, None], changed, state)
    return torch.stack(losses).mean()


def assess(core, rows, device, limit=30):
    selected = rows[:limit]; outputs = []
    with torch.inference_mode():
        for row in selected:
            _, codes = rollout(core, row, device)
            end = codes.index(BOUNDARY) if BOUNDARY in codes else len(codes)
            raw = bytes(code for code in codes[:end] if code < 256)
            actual = raw.decode('utf8', errors='replace')
            outputs.append({'input': row.get('input'), 'expected': row['answer'], 'actual': actual,
                'exact': actual == row['answer'], 'ended': BOUNDARY in codes,
                'nonempty': bool(actual.strip()),
                'validUtf8': raw.decode('utf8', errors='replace').encode('utf8') == raw,
                'metadata': row['metadata']})
    return {'total': len(outputs), 'exact': sum(row['exact'] for row in outputs),
        'ended': sum(row['ended'] for row in outputs), 'validUtf8': sum(row['validUtf8'] for row in outputs),
        'nonempty': sum(row['nonempty'] for row in outputs),
        'distinctOutputs': len({row['actual'] for row in outputs}), 'rows': outputs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=600)
    parser.add_argument('--pairs', type=int, default=400)
    parser.add_argument('--max-answer-bytes', type=int, default=200)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train-input', action='store_true')
    parser.add_argument('--state-feedback', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--own-output-rate', type=float, default=.05)
    args = parser.parse_args()
    if not 0. <= args.own_output_rate <= 1.:
        parser.error('--own-output-rate must be between 0 and 1')
    torch.set_num_threads(4); torch.manual_seed(711); random.seed(711)
    folder = ROOT/'artifacts'/f'liquid-language-{time.time_ns()}'; folder.mkdir()
    receipt = folder/'training.json'
    info = json.loads((ROOT/'artifacts/liquid-core-training.json').read_text(encoding='utf8'))
    core, checkpoint = load_core(ROOT/info['directory']/'latest.pt', args.device)
    if args.resume:
        resumed=torch.load(args.resume,map_location='cpu',weights_only=False)
        core.restore_weights(resumed['state_dict'],resumed['architecture'])
    core.eval()
    for name, parameter in core.named_parameters():
        if name.startswith('initial') or (not args.train_input and name.startswith(('input_', 'write_', 'time_projection'))):
            parameter.requires_grad_(False)
        if args.state_feedback and name.startswith('input_context'):
            parameter.requires_grad_(True)
    report = {'state': 'preparing', 'directory': str(folder.relative_to(ROOT)),
        'architecture': core.architecture, 'sourceCheckpoint': info['directory'],
        'pretrainedLanguageForwardCalls': 0, 'publishedToLive': False,
        'inputPeriodSeconds': .1, 'bytePeriodSeconds': .03,
        'stateConditionedInput': args.train_input or args.state_feedback,
        'trainSensoryBackbone': args.train_input, 'recomputedObservationTraces': True,
        'ownOutputFeedbackRate': args.own_output_rate,
        'recomputedQuietStates': True,
        'resumedCheckpoint': str(args.resume) if args.resume else None,
        'limits': ['Short human-written English supervision; not general awareness.',
                   'Candidate is not installed in the live agent.', 'Human ratings do not certify factual correctness or absence of bias.']}
    write_report(receipt, report)
    try:
        corpus = json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
        pairs = language_pairs(args.max_answer_bytes); random.shuffle(pairs); pairs = pairs[:args.pairs]
        sensory = backgrounds(corpus)
        rows = prepare_language(core, pairs, sensory, args.device)
        train = [row for row in rows if row['split'] == 'train']
        validation = [row for row in rows if row['split'] != 'train']
        assert train and validation
        raw, _ = prepare(core, [row for row in corpus['cases'] if row['split'] == 'train'], args.device)
        raw_validation, _ = prepare(core, [row for row in corpus['cases'] if row['split'] != 'train'], args.device)
        for row in raw + raw_validation:
            row['state'] = row['state'].clone(); row['inputPeriod'] = .1
        for item in corpus['cases']:
            case=torch.load(ROOT/item['file'],map_location='cpu',weights_only=False)
            for row in raw+raw_validation:
                if row['metadata']==case['metadata']:
                    tick=next(query['tick'] for query in case['queries'] if query['kind']==row['kind'])
                    row.update(observationPrefix=[event['tokens'] for event in case['events'][:tick]],
                               queryTokens=case['events'][tick]['tokens'])
        optimizer = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=.0002)
        report.update(state='training', trainTrees=len({row['tree'] for row in train}),
            validationTrees=len({row['tree'] for row in validation}), trainSequences=len(train),
            validationSequences=len(validation), parameters=core.parameter_count(),
            initialValidation=assess(core, validation, args.device, 12))
        write_report(receipt, report); started = time.monotonic()
        losses={'language':deque(maxlen=25),'rehearsal':deque(maxlen=25)}
        for update in range(1, args.updates + 1):
            if psutil.virtual_memory().used > 50_000_000_000:
                raise MemoryError('Training stopped: total system RAM exceeds the configured budget.')
            core.train(); optimizer.zero_grad(set_to_none=True)
            if update % 5 == 0:
                kind='rehearsal'
                row = random.choice(raw)
                loss, _ = rollout(core, row, args.device, teacher=True, sampling=.1)
            else:
                kind='language'
                loss = batched_rollout(core, random.sample(train, min(args.batch, len(train))), args.device,
                                       sampling=args.own_output_rate)
            losses[kind].append(float(loss.detach()))
            quiet = quiet_states(core, list(sensory.values()), args.device)
            loss = loss + F.cross_entropy(core.logits(quiet), torch.full((len(quiet),), BLANK, device=args.device))*.15
            loss.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if update % 25 == 0 or update == args.updates:
                core.eval()
                temporary = folder/'latest.tmp'
                torch.save({'architecture': core.architecture, 'state_dict': core.state_dict(), 'update': update,
                    'trainingScope': 'Human English plus raw AV rehearsal, live input cadence',
                    'sourceCheckpoint': report['sourceCheckpoint']}, temporary)
                temporary.replace(folder/'latest.pt')
                report.update(update=update, loss=float(loss), elapsedSeconds=time.monotonic()-started,
                    lossKind=kind, languageLoss=sum(losses['language'])/max(1,len(losses['language'])),
                    rehearsalLoss=sum(losses['rehearsal'])/max(1,len(losses['rehearsal'])),
                    systemUsedBytes=psutil.virtual_memory().used, agentGpuAllocatedBytes=torch.cuda.memory_allocated())
                if update % 200 == 0 or update == args.updates:
                    report['train'] = assess(core, train, args.device)
                    report['validation'] = assess(core, validation, args.device)
                    report['rawRegression'] = assess(core, raw_validation, args.device)
                write_report(receipt, report)
                print(json.dumps({k: report.get(k) for k in ['update', 'languageLoss', 'rehearsalLoss', 'elapsedSeconds', 'systemUsedBytes']}), flush=True)
        report['state'] = 'complete'; write_report(receipt, report)
    except Exception as error:
        report.update(state='failed', error=str(error)); write_report(receipt, report); raise


if __name__ == '__main__':
    from research_resources import research_slot
    with research_slot('language training'):
        main()
