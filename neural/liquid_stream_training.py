"""Complete recorded interactions through one state, including own output."""
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from liquid_core import BLANK, BOUNDARY

ROOT = Path(__file__).resolve().parents[1]


def examples(items, matched_switches=False):
    result = []
    for item in items:
        case = torch.load(ROOT/item['file'], map_location='cpu', weights_only=False)
        result.append(dict(events=case['events'], queries=case['queries'], metadata=case['metadata']))
    if matched_switches:
        representatives = {mode: next(row for row in result if row['metadata']['mode']==mode)['events'][90]['tokens']
                           for mode in ['held', 'changing']}
        for base in list(result):
            question = next(row for row in base['queries'] if row['kind']=='video')
            # Offline intervention during the common unfinished prefix. Only
            # visual query rows change; no scene label reaches model inputs.
            switch_tick = question['tick'] + 3
            for mode in ['held', 'changing']:
                events = []
                for tick, event in enumerate(base['events']):
                    if tick >= switch_tick:
                        tokens = event['tokens'].clone()
                        tokens[2:66] = representatives[mode][2:66]
                        event = dict(event, tokens=tokens)
                    events.append(event)
                queries = [dict(row) for row in base['queries']]
                next(row for row in queries if row['kind']=='video')['answer'] = (
                    'A person is playing tennis.' if mode=='held' else 'A person is riding a bicycle.')
                result.append(dict(events=events, queries=queries, metadata=dict(base['metadata'],
                    interventionMode=mode, visualSwitchTick=switch_tick)))
    return result


def targets(row, byte_period=.03):
    observation_times = []; elapsed = 0.
    for event in row['events']:
        elapsed += event['dt']; observation_times.append(elapsed)
    labels = [BLANK] * math.ceil(elapsed/byte_period-1e-8)
    expected = []
    for query in row['queries']:
        at = observation_times[query['tick']]
        first = math.ceil(at/byte_period-1e-8)-1
        codes = list(query['answer'].encode('utf8'))+[BOUNDARY]
        labels += [BLANK]*max(0,first+len(codes)-len(labels))
        if any(code != BLANK for code in labels[first:first+len(codes)]):
            raise ValueError('Training utterances overlap.')
        labels[first:first+len(codes)] = codes
        expected.append(dict(kind=query['kind'], answer=query['answer'],
            forcedFeedback=query.get('forcedFeedback', False), startStep=first, codeCount=len(codes),
            questionSeconds=at, targetStartSeconds=(first+1)*byte_period,
            targetEndSeconds=(first+len(codes))*byte_period))
    return labels, observation_times, expected


def self_report_examples(rows, alternatives=('Apple.','Mango.'), clamp_prefix=False):
    """Causal source-memory experiment, never a runtime conversation rule.

    Only a first own-code feedback intervention and its report target vary
    within each pair. Every sensory input and external-fact target matches.
    The injected utterance is excluded from predictive loss and accuracy.
    """
    from literal_stimuli import literal_rows
    text = literal_rows(['What did you say first?', 'Which fruit did you hear earlier?'])
    result = []
    for base_index, base in enumerate(rows):
        _, _, expected = targets(base)
        after = max(target['targetEndSeconds'] for target in expected)
        self_tick = max(len(base['events'])-10, math.ceil((after+.5)/.1)-1)
        # Same-length alternatives control elapsed intervals and word length.
        for word in alternatives:
            events = list(base['events'])
            tail = events[-1]['tokens']
            external_tick = self_tick + math.ceil((max(len(item.encode('utf8')) for item in alternatives)+1)*.03/.1) + 6
            for _ in range(max(0, external_tick+20-len(events))):
                events.append(dict(dt=.1, tokens=tail))
            for tick, encoded in zip([self_tick, external_tick], text):
                events[tick] = dict(events[tick], tokens=torch.cat([events[tick]['tokens'], encoded]))
            queries = [dict(query) for query in base['queries']]
            queries[0].update(answer=word, forcedFeedback=True)
            if clamp_prefix:
                for query in queries:
                    query['forcedFeedback'] = True
            queries += [dict(tick=self_tick, kind='self-report', answer=word),
                dict(tick=external_tick, kind='external-report',
                     answer=base['metadata']['fruit'].capitalize()+'.' if base['metadata']['audio'] else 'I did not hear a fruit.')]
            result.append(dict(events=events, queries=queries,
                clampBeforeSeconds=(self_tick+1)*.1 if clamp_prefix else None,
                metadata=dict(base['metadata'], sourceControlId=str(base_index),selfReportIntervention=word,
                              interventionMeaning='Injected own-code feedback, not a voluntary generated statement.')))
    return result


def rollout(core, row, device, teacher=False, sampling=0., tick=.01, observer=None):
    labels, observation_times, expected = targets(row)
    forced = {step for target in expected if target['forcedFeedback']
              for step in range(target['startStep'],target['startStep']+target['codeCount'])}
    if row.get('clampBeforeSeconds') is not None:
        forced.update(range(math.ceil(row['clampBeforeSeconds']/.03-1e-8)-1))
    state = core.start().clone()
    label_tensor = torch.tensor(labels, device=device)
    write_mode = 'simultaneous' if int(core.observation_scheme) else 'ordered'
    event_index = 0; output_index = 0; elapsed = 0.; emitted = []
    voice_loss = []; quiet_loss = []; byte_period = .03
    while output_index < len(labels):
        dt = min(tick, (output_index+1)*byte_period-elapsed)
        state = core.advance(state, max(0.,dt)); elapsed += dt
        while event_index < len(observation_times) and elapsed+1e-8 >= observation_times[event_index]:
            event = row['events'][event_index]
            state = core.observe(state, event['tokens'].to(device).float(), event['dt'], write_mode=write_mode)
            if observer is not None:
                observer(state, event_index, elapsed)
            event_index += 1
        if elapsed+1e-8 < (output_index+1)*byte_period:
            continue
        target = labels[output_index]
        logits = core.logits(state)
        loss = F.cross_entropy(logits, label_tensor[output_index:output_index+1])
        if output_index not in forced:
            (quiet_loss if target==BLANK else voice_loss).append(loss)
        predicted = logits.argmax(-1).detach()
        intervened = output_index in forced
        code = target if intervened or (teacher and random.random() >= sampling) else predicted
        emitted.append(dict(seconds=elapsed, code=label_tensor[output_index:output_index+1] if intervened else predicted, intervened=intervened))
        if isinstance(code, int):
            if code != BLANK:
                state = core.feedback(state, label_tensor[output_index:output_index+1])
        else:
            changed = core.feedback(state, code)
            state = torch.where((code != BLANK)[:,None,None,None], changed, state)
        output_index += 1
    loss = torch.stack(voice_loss).mean() + .3*torch.stack(quiet_loss).mean()
    # Transfer generated codes once, after this offline numerical rollout.
    codes = torch.cat([row['code'] for row in emitted]).cpu().tolist()
    for row, code in zip(emitted, codes):
        row['code'] = code
    return loss, emitted, expected, state


def decode(emitted):
    utterances = []; pending = bytearray(); started = None; intervened = False
    for row in emitted:
        code = row['code']
        if code < 256:
            if started is None:
                started = row['seconds']
            pending.append(code)
            intervened = intervened or row.get('intervened', False)
        elif code == BOUNDARY and pending:
            utterances.append(dict(text=pending.decode('utf8', errors='replace'),
                startSeconds=started, endSeconds=row['seconds'], ended=True, intervened=intervened))
            pending.clear(); started = None; intervened = False
    if pending:
        utterances.append(dict(text=pending.decode('utf8', errors='replace'),
            startSeconds=started, endSeconds=emitted[-1]['seconds'], ended=False, intervened=intervened))
    return utterances


def source_control_summary(records):
    selected=[row for row in records if 'selfReportIntervention' in row['metadata']]
    pairs={};roles={}
    for index,row in enumerate(selected):
        key=row['metadata'].get('sourceControlId',f'legacy-pair-{index//2}')
        pairs.setdefault(key,[]).append(row)
        for target in row['expected']:
            if target['forcedFeedback']:continue
            counts=roles.setdefault(target['kind'],dict(total=0,exact=0))
            counts['total']+=1;counts['exact']+=int(target['exact'] is True)
    correct=0;invalid=0
    for pair in pairs.values():
        complete=(len(pair)==2 and len({row['metadata']['selfReportIntervention'] for row in pair})==2)
        complete=complete and all({target['kind'] for target in row['expected'] if not target['forcedFeedback']}
                                 >= {'self-report','external-report'} for row in pair)
        if not complete:invalid+=1;continue
        correct+=int(all(target['exact'] is True for row in pair for target in row['expected']
                         if target['kind'] in ['self-report','external-report'] and not target['forcedFeedback']))
    return dict(roleScores=roles,sourceControlPairs=len(pairs),sourceControlPairsCorrect=correct,
                invalidSourceControlGroups=invalid,injectedUtterancesScored=False)


ONSET_WINDOW_SECONDS = .5


def match_targets(expected, actual, window=ONSET_WINDOW_SECONDS):
    """Strict scoring: exactly one generated utterance may start in a target's
    onset window, it must be finished and its text must match. Generated
    utterances outside every window are counted as false starts. Extra or wrong
    alternatives therefore fail rather than being ignored."""
    generated = [output for output in actual if not output.get('intervened')]
    windows = [(target['questionSeconds']-1e-8, target['questionSeconds']+window) for target in expected]
    matched = []
    for target, (low, high) in zip(expected, windows):
        inside = [output for output in generated if low <= output['startSeconds'] <= high]
        first = inside[0] if inside else None
        exact = (len(inside) == 1 and first['ended'] and first['text'] == target['answer'])
        matched.append(dict(target, exact=exact if not target['forcedFeedback'] else None,
            firstText=first['text'] if first else None, utterancesInWindow=len(inside),
            latencyMilliseconds=(first['startSeconds']-target['questionSeconds'])*1000 if exact else None))
    false_starts = sum(not any(low <= output['startSeconds'] <= high for low, high in windows) for output in generated)
    return matched, false_starts


def evaluate(core, rows, device):
    records = []
    with torch.inference_mode():
        for row in rows:
            loss, emitted, expected, state = rollout(core, row, device)
            actual = decode(emitted)
            matched, false_starts = match_targets(expected, actual)
            records.append(dict(metadata=row['metadata'], expected=matched, actual=actual,
                loss=float(loss), finite=bool(torch.isfinite(state).all()), falseStarts=false_starts,
                exactUtterances=sum(target['exact'] is True for target in matched),
                completeConversation=false_starts == 0 and all(target['exact'] for target in matched if not target['forcedFeedback'])))
    report = dict(conversations=len(records), completeConversations=sum(row['completeConversation'] for row in records),
        utterances=sum(sum(not target['forcedFeedback'] for target in row['expected']) for row in records),
        exactUtterances=sum(row['exactUtterances'] for row in records),
        falseStarts=sum(row['falseStarts'] for row in records), scoring='strict-first-utterance-v2',
        finiteStates=all(row['finite'] for row in records), rows=records)
    if any(row['metadata'].get('selfReportIntervention') for row in records):
        report.update(source_control_summary(records))
    return report
