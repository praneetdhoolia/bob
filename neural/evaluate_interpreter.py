"""Verify the custom English interpreter and the urge gate on recorded episodes.

Part A — interpreter accuracy (prepared-question protocol used to train the
selected checkpoint). Reports are grouped by what is actually held out:
trained fruit words, unseen fruit words, the no-fact control and video reports
whose frames are identical to training frames. Controls follow decoder
verification practice (Tang et al. 2023; Hewitt and Liang 2019):

* state swap (null): the held state is replaced by another episode's state with
  a different answer, while the original later observations are kept;
* identification: is the produced report most likely under its own state among
  all candidate states of the same report type?
* visual intervention: only native visual rows of later observations change;
* calibration: mean emitted-code probability versus correctness (Brier, ECE).

Part B — urge onset on complete streams with runtime timing (10 ms numerical
ticks, 100 ms observations, 30 ms readout), gate disabled vs enabled, with the
surprise drive clamped to zero and with a doubled threshold.

"Accurate" is claimed only for a report type that passes every applicable
criterion; familiar-frame video cannot establish visual generalization.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from liquid_core import BLANK, BOUNDARY, load_core
from liquid_stream_training import decode, match_targets, targets
from train_liquid_core import prepare, starting_state, write_report
from urge import THRESHOLD, UrgeGate

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = dict(heldAccuracy=.9, nullHitRate=.2, identificationPercentile=.85, interventionFlipRate=.9, ece=.1)


_STARTS = {}


def start_state(core, row):
    """The query-time state depends only on the observed history; cache it."""
    key = (id(row['observationPrefix']), id(row['queryTokens']))
    if key not in _STARTS:
        _STARTS[key] = starting_state(core, row, 'cpu')
    return _STARTS[key].clone()


def greedy(core, row, replacement=None):
    """The training-protocol rollout, also returning each emitted code's probability."""
    state = start_state(core, row); codes = []; confidence = []
    future = row['future'] if replacement is None else replacement
    for offset in range(48):
        state = core.advance(state, .03)
        if offset and offset % 5 == 0:
            state = core.observe(state, future[min(offset//5-1, len(future)-1)].float(), .15)
        probabilities = core.logits(state).softmax(-1)[0]
        code = int(probabilities.argmax())
        if code != BLANK:
            codes.append(code); confidence.append(float(probabilities[code]))
            state = core.feedback(state, code)
            if code == BOUNDARY:
                break
    stop = codes.index(BOUNDARY) if BOUNDARY in codes else len(codes)
    text = bytes(code for code in codes[:stop] if code < 256).decode('utf8', errors='replace')
    return text, (sum(confidence)/len(confidence) if confidence else 0.)


def report_loss(core, row, text):
    """Teacher-forced loss of a given report under a row's state."""
    state = start_state(core, row); total = 0.; codes = list(text.encode('utf8'))+[BOUNDARY]
    for offset, code in enumerate(codes):
        state = core.advance(state, .03)
        if offset and offset % 5 == 0:
            state = core.observe(state, row['future'][min(offset//5-1, len(row['future'])-1)].float(), .15)
        total -= float(core.logits(state).log_softmax(-1)[0, code])
        state = core.feedback(state, code)
    return total/len(codes)


def category(row, trained_words):
    if row['kind'] == 'video':
        return 'video-familiar-frames'
    if row['answer'] == 'I did not hear a fruit.':
        return 'no-fact-control'
    return 'fruit-trained-word' if row['metadata']['fruit'] in trained_words else 'fruit-unseen-word'


def calibration(confidence, correct, bins=5):
    total = len(confidence); ece = 0.
    for index in range(bins):
        low, high = index/bins, (index+1)/bins
        members = [i for i, c in enumerate(confidence) if low <= c < high or (index == bins-1 and c == 1.)]
        if members:
            ece += len(members)/total*abs(sum(confidence[i] for i in members)/len(members)
                                           - sum(correct[i] for i in members)/len(members))
    brier = sum((c-k)**2 for c, k in zip(confidence, correct))/total
    return dict(expectedCalibrationError=ece, brier=brier, trials=total)


def interpreter_protocol(core, cases):
    trained_words = {case['fruit'] for case in cases if case['split'] == 'train'}
    rows, _ = prepare(core, cases, 'cpu', recompute=True)
    results = []
    for row in rows:
        text, confidence = greedy(core, row)
        results.append(dict(kind=row['kind'], category=category(row, trained_words), split=row['metadata']['split'],
                            fruit=row['metadata']['fruit'], mode=row['metadata']['mode'], expected=row['answer'],
                            actual=text, correct=text == row['answer'], confidence=confidence))
    # Null: another episode's held state, original later observations.
    nulls = []
    for index, row in enumerate(rows):
        for donor_index, donor in enumerate(rows):
            if donor['kind'] != row['kind'] or donor['answer'] == row['answer']:
                continue
            swapped = dict(row, observationPrefix=donor['observationPrefix'], queryTokens=donor['queryTokens'])
            text, _ = greedy(core, swapped)
            nulls.append(dict(kind=row['kind'], target=index, donor=donor_index,
                              keptOriginal=text == row['answer'], followedDonor=text == donor['answer']))
    # Identification among all same-kind states, for reports the interpreter produced.
    identification = []
    for index, row in enumerate(rows):
        text = results[index]['actual']
        if not text:
            continue
        candidates = [j for j, other in enumerate(rows) if other['kind'] == row['kind']]
        losses = {j: report_loss(core, rows[j], text) for j in candidates}
        better = sum(losses[j] < losses[index] for j in candidates if j != index)
        identification.append(dict(kind=row['kind'], target=index,
                                   percentile=1-better/max(1, len(candidates)-1), top1=better == 0))
    # Visual intervention: later observations take the other scene's visual rows.
    interventions = []
    representative = {mode: next(r for r in rows if r['kind'] == 'video' and r['metadata']['mode'] == mode)
                      for mode in ['held', 'changing']}
    answer = {'held': 'A person is playing tennis.', 'changing': 'A person is riding a bicycle.'}
    for index, row in enumerate(rows):
        if row['kind'] != 'video':
            continue
        other = 'changing' if row['metadata']['mode'] == 'held' else 'held'
        replacement = []
        for offset, tokens in enumerate(row['future']):
            changed = tokens.clone()
            changed[2:66] = representative[other]['future'][min(offset, len(representative[other]['future'])-1)][2:66]
            replacement.append(changed)
        text, _ = greedy(core, row, replacement)
        interventions.append(dict(target=index, original=results[index]['actual'], intervened=text,
                                  flippedToOtherScene=text == answer[other]))
    summary = {}
    for kind in ['fruit', 'video']:
        mine = [r for r in results if r['kind'] == kind]
        by_category = {}
        for r in mine:
            group = by_category.setdefault(r['category'], dict(correct=0, total=0))
            group['correct'] += r['correct']; group['total'] += 1
        kind_nulls = [n for n in nulls if n['kind'] == kind]
        kind_ident = [i for i in identification if i['kind'] == kind]
        entry = dict(categories=by_category,
                     nullHitRate=sum(n['keptOriginal'] for n in kind_nulls)/len(kind_nulls) if kind_nulls else None,
                     nullFollowedDonorRate=sum(n['followedDonor'] for n in kind_nulls)/len(kind_nulls) if kind_nulls else None,
                     nullTrials=len(kind_nulls),
                     identificationPercentile=sum(i['percentile'] for i in kind_ident)/len(kind_ident) if kind_ident else None,
                     identificationTop1=sum(i['top1'] for i in kind_ident)/len(kind_ident) if kind_ident else None,
                     calibration=calibration([r['confidence'] for r in mine], [float(r['correct']) for r in mine]))
        if kind == 'video':
            entry['interventionFlipRate'] = sum(i['flippedToOtherScene'] for i in interventions)/len(interventions)
        held = [g for name, g in by_category.items() if name in ['fruit-unseen-word']]
        held_accuracy = (sum(g['correct'] for g in held)/sum(g['total'] for g in held)) if held else None
        checks = dict(
            heldOutAvailable=held_accuracy is not None,
            heldAccuracy=held_accuracy is not None and held_accuracy >= CRITERIA['heldAccuracy'],
            nullHitRate=entry['nullHitRate'] is not None and entry['nullHitRate'] <= CRITERIA['nullHitRate'],
            identification=entry['identificationPercentile'] is not None and entry['identificationPercentile'] >= CRITERIA['identificationPercentile'],
            calibration=entry['calibration']['expectedCalibrationError'] <= CRITERIA['ece'])
        if kind == 'video':
            checks['intervention'] = entry['interventionFlipRate'] >= CRITERIA['interventionFlipRate']
        entry['heldAccuracy'] = held_accuracy; entry['checks'] = checks
        entry['verdict'] = ('accurate within tested scope' if all(checks.values()) else
                            'state-dependent on familiar inputs; not verified on held-out inputs' if not checks['heldOutAvailable'] or not checks['heldAccuracy'] else
                            'not verified')
        summary[kind] = entry
    return dict(criteria=CRITERIA, summary=summary, reports=results, nullTrials=nulls,
                identification=identification, interventions=interventions)


def stream(core, row, gate):
    """Runtime timing on a complete recorded episode, with the given urge gate."""
    _, observation_times, expected = targets(row)
    state = core.start().clone(); elapsed = 0.; event_index = 0; emitted = []
    end = observation_times[-1] + 1.5; next_read = .03
    while elapsed < end:
        state = core.advance(state, .01); elapsed += .01
        while event_index < len(observation_times) and elapsed+1e-8 >= observation_times[event_index]:
            tokens = row['events'][event_index]['tokens'].float()
            gate.observe_packet(core.sensory_target(tokens))
            state = core.observe(state, tokens, row['events'][event_index]['dt']); event_index += 1
        if elapsed+1e-8 < next_read:
            continue
        next_read += .03
        probabilities = core.logits(state).softmax(-1)[0]; code = int(probabilities.argmax())
        gate.advance(.03, probabilities)
        if not gate.permits(code):
            continue
        emitted.append(dict(seconds=elapsed, code=code))
        state = core.feedback(state, code)
        if code == BOUNDARY:
            gate.finished()
    actual = decode(emitted) if emitted else []
    matched, false_starts = match_targets(expected, actual)
    onsets = [(next((u['startSeconds'] for u in actual if t['questionSeconds'] <= u['startSeconds'] <= t['questionSeconds']+.5), None), t['questionSeconds'])
              for t in expected]
    latencies = [(start-question)*1000 for start, question in onsets if start is not None]
    return dict(exact=sum(t['exact'] is True for t in matched), targets=len(expected), falseStarts=false_starts,
                utterances=len(actual), latenciesMs=latencies, releases=gate.releases, finite=bool(torch.isfinite(state).all()))


def urge_protocol(core, cases):
    conditions = dict(ungated=dict(enabled=False), gated=dict(), surpriseClamped=dict(surprise_gain=0.),
                      doubledThreshold=dict(threshold=2*THRESHOLD))
    episodes = [dict(events=torch.load(ROOT/case['file'], map_location='cpu', weights_only=False)['events'],
                     queries=case['queries'], split=case['split']) for case in cases]
    summary = {}
    for name, settings in conditions.items():
        rows = [stream(core, episode, UrgeGate(**settings)) for episode in episodes]
        latencies = [value for row in rows for value in row['latenciesMs']]
        summary[name] = dict(exactUtterances=sum(r['exact'] for r in rows), targets=sum(r['targets'] for r in rows),
                             falseStarts=sum(r['falseStarts'] for r in rows), utterances=sum(r['utterances'] for r in rows),
                             onsetsInWindow=len(latencies), meanOnsetLatencyMs=sum(latencies)/len(latencies) if latencies else None,
                             releases=sum(r['releases'] for r in rows), finiteStates=all(r['finite'] for r in rows))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--skip-urge', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    if args.checkpoint:
        path = args.checkpoint
    else:
        info = json.loads((ROOT/'artifacts/liquid-core-training.json').read_text(encoding='utf8'))
        path = ROOT/info['directory']/'latest.pt'
    core, checkpoint = load_core(path); core.eval()
    corpus = json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
    begin = time.time()
    with torch.inference_mode():
        receipt = dict(checkpoint=str(path.relative_to(ROOT)).replace('\\', '/'), architecture=checkpoint['architecture'],
                       createdAtUnix=time.time(), corpus='artifacts/workspace-raw-corpus.json',
                       heldOutNote='Validation video rows are bit-identical to training episodes; only fruit words are held out.',
                       interpreter=interpreter_protocol(core, corpus['cases']))
        if not args.skip_urge:
            receipt['urge'] = urge_protocol(core, corpus['cases'])
    receipt['elapsedSeconds'] = time.time()-begin
    receipt['limits'] = ['Two recorded scenes and synthetic speech; familiar-frame video cannot establish visual generalization.',
                         'Passing these controls would show accurate state-to-English readout within scope, not consciousness.']
    destination = ROOT/'artifacts'/f'interpreter-verification-{time.time_ns()}.json'
    write_report(destination, receipt)
    print(json.dumps(dict(receipt=str(destination.relative_to(ROOT)),
                          interpreter={k: {key: v[key] for key in ['categories', 'heldAccuracy', 'nullHitRate', 'nullFollowedDonorRate',
                                                                  'identificationPercentile', 'interventionFlipRate', 'verdict'] if key in v}
                                       for k, v in receipt['interpreter']['summary'].items()},
                          urge=receipt.get('urge'), elapsedSeconds=receipt['elapsedSeconds']), indent=1))


if __name__ == '__main__':
    from research_resources import research_slot
    with research_slot('interpreter verification'):
        main()
