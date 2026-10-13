"""Separate completed-question timing from responses during ongoing speech.

Keeps the strict score and does not reinterpret a caption as proof of spoken
understanding. Onsets come from actual fixture lengths and recorded end ticks.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from build_workspace_raw_corpus import read_audio
from train_liquid_core import write_report

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_receipt',type=Path)
    args = parser.parse_args()
    run = json.loads(args.training_receipt.read_text(encoding='utf8'))
    corpus = json.loads((ROOT/'artifacts/spoken-stream-corpus.json').read_text(encoding='utf8'))
    audio = ROOT/corpus['directory']/'audio'
    rng = random.Random(972); annotations = {}
    for item in corpus['cases']:
        # Reproduce only the builder's three timing draws. Stored end ticks and
        # actual waveform lengths independently recover the question onsets.
        rng.randint(60,74);rng.randint(130,140);switch_tick=rng.randint(35,70)
        signature = json.dumps({key:value for key,value in item.items()
            if key not in ['file','ticks','queries','audioEncoderBarrierTick']},sort_keys=True)
        onsets = {}
        for query in item['queries']:
            if item['questionChannel']=='spoken':
                wave = read_audio(audio/f"{item['speaker']}-{query['kind']}-{item['phraseIndex']}.wav")
                onset_tick = query['tick']-int(np.ceil(len(wave)/1600))+1
            else:
                onset_tick = query['tick']
            onsets[query['kind']] = (onset_tick+1)*.1
        annotations[signature] = dict(onsets=onsets,switchSeconds=(switch_tick+1)*.1)
    groups = {}; detail = []
    for row in run['validation']['rows']:
        meta = row['metadata'];annotation=annotations[json.dumps(meta,sort_keys=True)]
        name = f"{meta['questionChannel']}/{meta['transfer']}"
        group = groups.setdefault(name,dict(total=0,strictCompletedQuestion=0,
                                            correctAfterQuestionOnset=0,correctDuringQuestion=0))
        for target in row['expected']:
            onset = annotation['onsets'][target['kind']]
            candidates = [output for output in row['actual'] if output['ended']
                and output['text']==target['answer']
                and onset-1e-8 <= output['startSeconds'] <= target['questionSeconds']+.5]
            # For these simple captions, require the eventual labelled scene to
            # be present throughout the utterance. A mid-utterance scene switch
            # needs a separate streaming-correction test, not a relaxed score.
            if target['kind']=='video' and meta['mode']=='changing':
                candidates = [output for output in candidates
                              if output['startSeconds']>=annotation['switchSeconds']-1e-8]
            group['total']+=1;group['strictCompletedQuestion']+=int(target['exact'])
            group['correctAfterQuestionOnset']+=bool(candidates)
            during = any(output['startSeconds']<target['questionSeconds']-1e-8 for output in candidates)
            group['correctDuringQuestion']+=during
            detail.append(dict(metadata=meta,kind=target['kind'],questionOnsetSeconds=onset,
                questionEndSeconds=target['questionSeconds'],strictExact=target['exact'],
                validMatchingUtterances=candidates))
    report = dict(trainingReceipt=str(args.training_receipt),groups=groups,rows=detail,
        limits=['Correct content during speech does not prove question comprehension.',
            'Counterfactual question-type and no-question controls are still needed.',
            'Original strict timing scores remain unchanged.',
            'Scene-switch times are reconstructed from the pinned builder seed; inputs are not re-encoded.'])
    path = args.training_receipt.parent/'spoken-timing-audit.json'
    write_report(path,report);print(json.dumps(dict(receipt=str(path),groups=groups)))


if __name__=='__main__':
    main()
