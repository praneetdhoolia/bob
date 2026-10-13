"""Measure native-packet strength learning before authorizing a longer fit."""
import argparse
import json
from pathlib import Path
import time
import psutil
import torch
from liquid_core import load_core
from strength_credit import ExactStrengthLearner
from research_resources import research_slot
from train_liquid_core import write_report
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--packets',type=int,default=4)
    parser.add_argument('--chunk-size',type=int,default=64);args=parser.parse_args()
    if not 1<=args.packets<=40:parser.error('--packets must be from 1 to 40')
    if not 1<=args.chunk_size<=256:parser.error('--chunk-size must be from 1 to 256')
    with research_slot('structured strength-credit timing',reserve_bytes=1_000_000_000):
        torch.set_num_threads(2)
        path=ROOT/'artifacts'/f'strength-credit-timing-{time.time_ns()}.json'
        report=dict(state='preparing',device=args.device,chunkSize=args.chunk_size,publishedToLive=False,
            sourceCheckpoint=args.checkpoint,packets=[],limits=['Cached native embeddings; raw encoders excluded.',
                'Predictor freezes, only sensory strength weights learn.',
                'One output code per packet; timing is not the live 30 ms schedule.'])
        write_report(path,report)
        try:
            core,_=load_core(args.checkpoint,args.device);core.eval()
            learner=ExactStrengthLearner(core,chunk_size=args.chunk_size)
            corpus=json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text())
            row=torch.load(ROOT/corpus['cases'][0]['file'],weights_only=False,map_location='cpu')
            state=core.start().detach().clone()
            for event in row['events'][60:60+args.packets]:
                tokens=event['tokens'].to(args.device).float()
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                begin=time.perf_counter();state,code,metrics=learner.step(state,tokens,event['dt'])
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                metrics.update(seconds=time.perf_counter()-begin,rows=len(tokens),code=code,
                    systemUsedBytes=psutil.virtual_memory().used,
                    gpuAllocatedBytes=torch.cuda.memory_allocated() if args.device.startswith('cuda') else 0)
                report['packets'].append(metrics);report['state']='measuring';write_report(path,report)
                print(json.dumps(metrics),flush=True)
                if metrics['systemUsedBytes']>50_000_000_000:raise MemoryError('System RAM budget exceeded.')
            report.update(state='complete',stateFinite=bool(torch.isfinite(state).all()),
                finalEligibilityBytes=learner.eligibility.numel()*learner.eligibility.element_size())
            write_report(path,report);print(path,flush=True)
        except Exception as error:
            report.update(state='failed',error=str(error));write_report(path,report);raise


if __name__=='__main__':main()
