"""Meaningful numerical invariants and matched learned-output interventions."""
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from liquid_core import LiquidCore,delta_write,delta_chunk_write,load_core
from train_liquid_core import prepare,evaluate

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4);torch.manual_seed(902)
    receipt={'architecture':LiquidCore.architecture,'createdAtUnix':time.time(),'checks':{}}
    memory=torch.randn(1,4,64,64,requires_grad=True)
    keys=F.normalize(torch.randn(1,4,12,64),dim=-1);values=torch.randn(1,4,12,64);strength=torch.rand(1,4,12)
    sequential=memory
    for index in range(12):sequential=delta_write(sequential,keys[:,:,index],values[:,:,index],strength[:,:,index])
    parallel=delta_chunk_write(memory,keys,values,strength)
    difference=float((parallel-sequential).abs().max());assert difference<.00001
    grad1=torch.autograd.grad(parallel.square().sum(),memory,retain_graph=True)[0]
    grad2=torch.autograd.grad(sequential.square().sum(),memory)[0]
    gradient_difference=float((grad1-grad2).abs().max());assert gradient_difference<.0001
    receipt['checks']['orderedWrite']={'maximumDifference':difference,'gradientDifference':gradient_difference}
    info=json.loads((ROOT/'artifacts/liquid-core-training.json').read_text(encoding='utf8'))
    model,checkpoint=load_core(ROOT/info['directory']/'latest.pt');model.eval()
    state=model.start().detach().clone()
    with torch.inference_mode():
        unchanged=model.advance(state,0.);assert torch.equal(state,unchanged)
        long=state
        for _ in range(10000):long=model.advance(long,.1)
        assert torch.isfinite(long).all()
        maximum=float(long.abs().max());assert maximum<=max(1.,float(state.abs().max()))
        receipt['checks']['boundedPhysicalFlow']={'seconds':1000,'maximumCoordinate':maximum,'zeroTimeExact':True,
                                                 'stateChange':float((long-state).abs().max()),'interpretation':'Numerical stability; not autonomous thought.'}
    corpus=json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
    rows,idle=prepare(model,[row for row in corpus['cases'] if row['split']!='train'],'cpu')
    receipt['validation']=evaluate(model,rows,idle,'cpu')
    # Replace only image query rows at the same prior matrix. No caption or
    # prompt is supplied to either intervention.
    changing=torch.load(ROOT/corpus['cases'][0]['file'],weights_only=False)
    held=next(item for item in corpus['cases'] if item['mode']=='held' and item['fruit']=='mango' and item['speechPresent'])
    held=torch.load(ROOT/held['file'],weights_only=False)
    state=model.start().detach().clone()
    with torch.inference_mode():
        for tick in range(50):state=model.observe(model.advance(state,.1),changing['events'][tick]['tokens'],.1)
        query=next(row for row in changing['queries'] if row['kind']=='video')['tick']
        left=changing['events'][query]['tokens'].clone();right=left.clone()
        right[2:66]=held['events'][query]['tokens'][2:66]
        a=model.observe(state,left,.1);b=model.observe(state,right,.1)
        pa=model.logits(a).softmax(-1);pb=model.logits(b).softmax(-1)
        receipt['checks']['visualStateIntervention']={'otherRowsIdentical':torch.equal(left[66:],right[66:]) and torch.equal(left[:2],right[:2]),
            'stateMaximumDifference':float((a-b).abs().max()),'nextByteTotalVariation':float((pa-pb).abs().sum()/2)}
    assert receipt['checks']['visualStateIntervention']['stateMaximumDifference']>0
    receipt['pretrainedLanguageForwardCalls']=0
    path=ROOT/'artifacts/liquid-core-validation.json';path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps({'checks':receipt['checks'],'validation':{k:v for k,v in receipt['validation'].items() if k!='rows'}},indent=2))


if __name__=='__main__':
    from research_resources import research_slot
    with research_slot('core validation'):
        main()
