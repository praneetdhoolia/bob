"""Compare correct-prefix and own-output trajectories of a candidate core."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from liquid_core import BLANK, BOUNDARY, load_core
from train_liquid_core import starting_state, write_report
from train_liquid_language import language_pairs, backgrounds, prepare_language

ROOT = Path(__file__).resolve().parents[1]


def trajectory(core, row, device, teacher, observations):
    state = starting_state(core, row, device)
    target_codes = list(row['answer'].encode('utf8')) + [BOUNDARY]
    records = []; emitted = []; elapsed = 0.; next_input = .1
    with torch.inference_mode():
        for offset, target in enumerate(target_codes):
            state = core.advance(state, .03); elapsed += .03
            if observations and elapsed + 1e-8 >= next_input:
                state = core.observe(state, row['future'][0].to(device).float(), .1)
                next_input += .1
            logits = core.logits(state)
            predicted = int(logits.argmax(-1))
            records.append({'offset': offset, 'target': target, 'predicted': predicted,
                'negativeLogLikelihood': float(F.cross_entropy(logits, torch.tensor([target], device=device)))})
            code = target if teacher else predicted
            emitted.append(predicted)
            if code != BLANK:
                state = core.feedback(state, code)
    end = emitted.index(BOUNDARY) if BOUNDARY in emitted else len(emitted)
    raw = bytes(code for code in emitted[:end] if code < 256)
    return {'meanNegativeLogLikelihood': sum(row['negativeLogLikelihood'] for row in records)/len(records),
        'correctByteFraction': sum(row['predicted'] == row['target'] for row in records)/len(records),
        'firstIncorrectByte': next((row['offset'] for row in records if row['predicted'] != row['target']), None),
        'ended': BOUNDARY in emitted, 'actual': raw.decode('utf8', errors='replace'),
        'byteRecords': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pairs', type=int, default=160)
    parser.add_argument('--max-answer-bytes', type=int, default=160)
    args = parser.parse_args()
    torch.set_num_threads(4); random.seed(711)
    core, checkpoint = load_core(args.checkpoint, args.device); core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    corpus = json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
    pairs = language_pairs(args.max_answer_bytes); random.shuffle(pairs); pairs = pairs[:args.pairs]
    selected = []
    for split in ['train', 'validation']:
        selected += [dict(row, preparationIndex=index) for index, row in enumerate(pairs) if row['split'] == split][:2]
    rows = prepare_language(core, selected, backgrounds(corpus), args.device)
    report = {'checkpoint': str(args.checkpoint), 'checkpointUpdate': checkpoint.get('update'),
        'architecture': checkpoint['architecture'], 'publishedToLive': False,
        'limits': ['Four prompts, three background conditions each; diagnostic, not a benchmark.',
            'Correct-prefix predictions are not independently generated sentences.'], 'conditions': []}
    for row in rows:
        results = {}
        for label, teacher, observations in [('teacherObserved', True, True),
                ('ownObserved', False, True), ('ownWithoutNewObservations', False, False)]:
            results[label] = trajectory(core, row, args.device, teacher, observations)
        report['conditions'].append(dict(input=row['input'], expected=row['answer'],
            split=row['split'], observationPrefixCount=len(row['observationPrefix']),
            metadata=row['metadata'], **results))
        print(json.dumps({'split': row['split'], 'background': row['metadata']['background'],
            'teacherLoss': results['teacherObserved']['meanNegativeLogLikelihood'],
            'ownLoss': results['ownObserved']['meanNegativeLogLikelihood'],
            'ownWithoutObservationLoss': results['ownWithoutNewObservations']['meanNegativeLogLikelihood']}), flush=True)
    path = ROOT/'artifacts'/f'liquid-language-trajectories-{time.time_ns()}.json'
    write_report(path, report)
    print(json.dumps({'receipt': str(path)}), flush=True)


if __name__ == '__main__':
    from research_resources import research_slot
    with research_slot('language trajectory validation'):
        main()
