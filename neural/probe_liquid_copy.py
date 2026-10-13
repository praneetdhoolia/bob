"""Fixed-duration, terminated-output controls for own-feedback content recall."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from liquid_core import load_core
from literal_stimuli import literal_rows
from research_resources import research_slot
from train_liquid_core import write_report
from train_liquid_copy import assess, words
from train_liquid_language import backgrounds

ROOT = Path(__file__).resolve().parents[1]


def confidence_summary(rows):
    """Code probability calibration and its association with later word recall.

    Mean selected-code probability is only a diagnostic score, never a
    probability of knowing the entire word or a measure of self-awareness.
    """
    bins=[]
    for index in range(10):
        selected=[row for row in rows if min(9,int(row['firstCodeProbability']*10))==index]
        if selected:
            bins.append(dict(count=len(selected),
                meanProbability=sum(row['firstCodeProbability'] for row in selected)/len(selected),
                firstCodeAccuracy=sum(row['firstCodeCorrect'] for row in selected)/len(selected)))
    positive=[row['meanSelectedCodeProbability'] for row in rows if row['exact']]
    negative=[row['meanSelectedCodeProbability'] for row in rows if not row['exact']]
    auc=(sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
         if positive and negative else None)
    high=[row for row in rows if row['meanSelectedCodeProbability']>=.9]
    return dict(total=len(rows),
        firstCodeAccuracy=sum(row['firstCodeCorrect'] for row in rows)/len(rows),
        firstCodeBrier=sum((row['firstCodeProbability']-row['firstCodeCorrect'])**2 for row in rows)/len(rows),
        firstCodeCalibrationError=sum(row['count']*abs(row['meanProbability']-row['firstCodeAccuracy']) for row in bins)/len(rows),
        firstCodeCalibrationBins=bins,wordRecallDiagnosticAuc=auc,
        highMeanCodeProbabilityCount=len(high),highMeanCodeProbabilityWordFailures=sum(not row['exact'] for row in high),
        meanCodeProbabilityIsNotWordConfidence=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    with research_slot('strict own-feedback recall controls'):
        torch.set_num_threads(4)
        core, _ = load_core(args.checkpoint, args.device)
        core.eval()
        corpus = json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
        sensory = {name: value.to(args.device).float()[None] for name,value in backgrounds(corpus).items()}
        query = literal_rows(['What did you say first?'])[0].to(args.device).float()
        train = words(2048,119,'abcdef')
        validation = words(128,120,'abcdef',train)
        report = dict(checkpoint=str(args.checkpoint), architecture=core.architecture,
            publishedToLive=False, injectionSlots=9, outputSlots=12, bytePeriodSeconds=.03,
            strictTerminationRequired=True, validationTrainingOverlap=0,
            queryProtocol='Literal query joins the next ordinary 100 ms sensory packet.',
            sourceSha256=hashlib.sha256((ROOT/'neural/train_liquid_copy.py').read_bytes()).hexdigest(),
            limits=['Synthetic feedback content, not broad English or voluntary intent.',
                    'The feedback intervention is unscored and never supplied as input text.',
                    'Sensory inputs continue at 100 ms; there is no transcript/string lookup.'])
        report['delayValidation'] = [assess(core,validation,sensory['quiet'],query,args.device,delay=delay)
                                     for delay in [0.,.3,1.,3.]]
        report['noOwnFeedbackControl'] = assess(core,validation,sensory['quiet'],query,args.device,inject=False)
        batched = assess(core,validation[:16],sensory['quiet'],query,args.device)
        single = assess(core,validation[:16],sensory['quiet'],query,args.device,batch_size=1)
        report['singleStreamValidation'] = single
        report['batchEquivalentOutputs'] = all(
            (a['actual'],a['terminated']) == (b['actual'],b['terminated'])
            for a,b in zip(batched['rows'],single['rows']))
        report['visualDistractions'] = {name:assess(core,validation[:32],sensory[name],query,args.device)
                                         for name in ['tennis','cycling']}
        groups={'quiet':report['delayValidation'][1],**report['visualDistractions'],
                'removed-feedback':report['noOwnFeedbackControl']}
        report['outputUncertainty']={name:confidence_summary(group['rows']) for name,group in groups.items()}
        destination = ROOT/'artifacts'/f'liquid-copy-controls-{time.time_ns()}.json'
        write_report(destination,report)
        print(json.dumps(dict(receipt=str(destination),
            delays=[dict(seconds=row['delaySeconds'],exact=row['exact'],total=row['total'])
                    for row in report['delayValidation']],
            removedFeedback=report['noOwnFeedbackControl']['exact'],
            batchEquivalentOutputs=report['batchEquivalentOutputs'],
            visualDistractions={name:row['exact'] for name,row in report['visualDistractions'].items()},
            uncertainty={name:{key:row[key] for key in ['firstCodeAccuracy','firstCodeBrier',
                'wordRecallDiagnosticAuc','highMeanCodeProbabilityCount','highMeanCodeProbabilityWordFailures']}
                for name,row in report['outputUncertainty'].items()})),flush=True)


if __name__ == '__main__':
    main()
