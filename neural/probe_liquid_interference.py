"""Isolate query-packet changes from pre/post-query sensory interference.

All phase controls are offline stimulus interventions, never runtime rules or
assigned cognitive slots. The model sees only native rows and elapsed time.
"""
import argparse
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
from probe_liquid_copy import confidence_summary

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    with research_slot('matched-query sensory-interference controls'):
        torch.set_num_threads(4)
        core,_=load_core(args.checkpoint,args.device);core.eval()
        manifest=json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
        sensory={name:value.to(args.device).float()[None] for name,value in backgrounds(manifest).items()}
        query=literal_rows(['What did you say first?'])[0].to(args.device).float()
        train=words(2048,119,'abcdef');validation=words(128,120,'abcdef',train)
        conditions=[('quiet','quiet','quiet','quiet')]
        for scene in ['tennis','cycling']:
            conditions += [(scene+'-throughout',scene,scene,scene),
                (scene+'-before',scene,'quiet','quiet'),
                (scene+'-query','quiet',scene,'quiet'),
                (scene+'-after','quiet','quiet',scene),
                (scene+'-known-query',scene,'quiet',scene)]
        results={}
        for name,before,cue,after in conditions:
            results[name]=assess(core,validation,sensory[before],query,args.device,
                query_sensory=sensory[cue],future_sensory=sensory[after])
        report=dict(checkpoint=str(args.checkpoint),publishedToLive=False,results=results,
            quietQueryTokenOffset=2,fullVisualQueryTokenOffset=68,
            conditions=[dict(name=name,before=before,query=cue,after=after) for name,before,cue,after in conditions],
            uncertainty={name:confidence_summary(row['rows']) for name,row in results.items()},
            limits=['Synthetic feedback recall, two recorded scenes and one familiar literal query.',
                    'After-only conditions keep all observations through the query identical.',
                    'Other conditions deliberately vary query packet and/or prior sensory history.',
                    'No transcript, source string lookup, subjective-awareness claim or live-model replacement.'])
        destination=ROOT/'artifacts'/f'liquid-interference-controls-{time.time_ns()}.json'
        write_report(destination,report)
        print(json.dumps(dict(receipt=str(destination),
            scores={name:dict(exact=row['exact'],total=row['total']) for name,row in results.items()})),flush=True)


if __name__=='__main__':
    main()
