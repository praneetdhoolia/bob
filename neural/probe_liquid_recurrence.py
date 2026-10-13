"""Paired internal-flow controls, separating unchanged and nontrivial answers."""
import argparse
import json
from pathlib import Path
import time

import torch

from liquid_core import load_core
from research_resources import research_slot
from train_liquid_core import write_report
from train_liquid_recurrence import cue, graph_cases, hop_baselines, process, stimuli

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--convergence',action='store_true',help='Add smaller integration steps and state-error comparisons.')
    args=parser.parse_args()
    with research_slot('paired recurrent-processing controls'):
        torch.set_num_threads(4)
        core,_=load_core(args.checkpoint,args.device);core.eval()
        vectors=stimuli().to(args.device)
        train=graph_cases(1024,1107)
        validation=graph_cases(64,1113,[tuple(row['graph']) for row in train])
        conditions=[('immediate',0.,.01,True),('disabled',.4,.01,False),
                    ('active',.4,.01,True),('finer',.4,.005,True),('coarser',.4,.02,True)]
        if args.convergence:
            conditions += [('fine-2.5ms',.4,.0025,True),('fine-1.25ms',.4,.00125,True),('reference-0.625ms',.4,.000625,True)]
        records=[];state_errors={}
        with torch.inference_mode():
            for case in validation:
                state=cue(core,vectors,case['graph'])
                predictions={};states={}
                for name,seconds,tick,active in conditions:
                    core._active_internal=active
                    states[name]=process(core,state,seconds,tick)
                    predictions[name]=(core.logits(states[name]).argmax(-1)-48).cpu().tolist()
                if args.convergence:
                    reference=states['reference-0.625ms']
                    for name,seconds,_,active in conditions:
                        if seconds and active:
                            errors=state_errors.setdefault(name,[])
                            difference=states[name]-reference
                            errors.append(dict(maximum=float(difference.abs().max()),
                                relativeRms=float(difference.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-8))))
                for start,expected in enumerate(case['target']):
                    records.append(dict(graph=case['graph'],start=start,expected=expected,
                        nontrivial=expected!=case['graph'][start],
                        predicted={name:row[start] for name,row in predictions.items()}))
        summary={}
        for name,_,_,_ in conditions:
            summary[name]={group:dict(total=len(selected),correct=sum(row['predicted'][name]==row['expected'] for row in selected))
                for group,selected in [('all',records),('nontrivial',[r for r in records if r['nontrivial']]),
                                       ('unchanged',[r for r in records if not r['nontrivial']])]}
        pairs=dict(improved=sum(r['predicted']['active']==r['expected'] and r['predicted']['disabled']!=r['expected'] for r in records),
                   damaged=sum(r['predicted']['active']!=r['expected'] and r['predicted']['disabled']==r['expected'] for r in records),
                   tickChangedPredictions=sum(r['predicted']['active']!=r['predicted']['finer'] for r in records))
        report=dict(checkpoint=str(args.checkpoint),publishedToLive=False,summary=summary,paired=pairs,rows=records,
            structuralBaselines=hop_baselines(validation),
            limits=['Eight starting-node decisions within each graph are correlated.',
                    'Held-out synthetic relations, not audiovisual deliberation or consciousness.'])
        if args.convergence:
            report['stateIntegrationError']={name:dict(maximum=max(row['maximum'] for row in values),
                meanRelativeRms=sum(row['relativeRms'] for row in values)/len(values))
                for name,values in state_errors.items()}
        destination=ROOT/'artifacts'/f'liquid-recurrence-controls-{time.time_ns()}.json'
        write_report(destination,report)
        print(json.dumps(dict(receipt=str(destination),summary=summary,paired=pairs,
            stateIntegrationError=report.get('stateIntegrationError'))),flush=True)


if __name__=='__main__':
    main()
