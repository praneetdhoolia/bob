"""Open-content memory of own-code feedback through the same numerical state.

Generated strings are offline controls. They are never supplied as text or
looked up during readout. Quiet native sensory observations continue throughout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import psutil
import torch
import torch.nn.functional as F

from liquid_core import BLANK, BOUNDARY, load_core
from research_resources import research_slot
from train_liquid_core import write_report
from train_liquid_language import backgrounds
from literal_stimuli import literal_rows
from liquid_stream_training import examples, rollout as stream_rollout, evaluate as stream_evaluate
from training_state import capture_training,restore_training

ROOT=Path(__file__).resolve().parents[1]


def words(count, seed, alphabet, excluded=()):
    rng=random.Random(seed);seen=set(excluded);result=[]
    while len(result)<count:
        word=''.join(rng.choice(alphabet) for _ in range(rng.randint(3,8)))
        if word not in seen:
            seen.add(word);result.append(word)
    return result


def feedback(core,state,codes):
    changed=core.feedback(state,codes)
    return torch.where((codes!=BLANK)[:,None,None,None],changed,state)


def run(core,samples,sensory,query,device,teacher=False,sampling=.1,delay=.3,inject=True,trace=False,
        query_sensory=None,future_sensory=None):
    state=core.start(len(samples)).clone();elapsed=0.;next_input=.1;query_pending=False;input_count=0;query_seen=False
    sequences=[list(word.encode('utf8'))+[BOUNDARY] for word in samples]
    def advance(dt):
        nonlocal state,elapsed,next_input,query_pending,input_count,query_seen
        state=core.advance(state,dt);elapsed+=dt
        if elapsed+1e-8>=next_input:
            stream=sensory
            if query_pending and query_sensory is not None:
                stream=query_sensory
            elif query_seen and future_sensory is not None:
                stream=future_sensory
            packet=stream[input_count%len(stream)][None]
            if query_pending:
                packet=torch.cat([packet,query[None]],1)
                query_seen=True
            state=core.observe(state,packet.expand(len(samples),-1,-1),.1,write_mode='simultaneous')
            query_pending=False
            input_count+=1
            next_input+=.1
    # Controlled actual feedback. No predictive loss or success score here.
    for at in range(9):
        advance(.03)
        codes=torch.tensor([seq[at] if at<len(seq) else BLANK for seq in sequences],device=device)
        if inject:
            state=feedback(core,state,codes)
    for _ in range(round(delay/.01)):
        advance(.01)
    # The query shares the next ordinary sensory packet. Do not insert an
    # extra write between observations or expose expected string length.
    query_pending=True
    while query_pending:
        advance(min(.01,next_input-elapsed))
    outputs=[];losses=[];scores=[]
    for at in range(12):
        advance(.03);logits=core.logits(state)
        target=torch.tensor([seq[at] if at<len(seq) else BLANK for seq in sequences],device=device)
        relevant=torch.tensor([at<len(seq)+3 for seq in sequences],device=device)
        per_row=F.cross_entropy(logits,target,reduction='none')
        losses.append((per_row*relevant).sum()/relevant.sum().clamp_min(1))
        predicted=logits.argmax(-1).detach();outputs.append(predicted)
        if trace:
            probabilities=logits.detach().softmax(-1)
            selected=probabilities.gather(1,predicted[:,None]).squeeze(1)
            entropy=-(probabilities*probabilities.clamp_min(1e-30).log2()).sum(-1)
            scores.append(torch.stack([selected,entropy],-1))
        codes=target if teacher and random.random()>=sampling else predicted
        state=feedback(core,state,codes)
    codes=torch.stack(outputs,1).cpu().tolist()
    score_rows=torch.stack(scores,1).cpu().tolist() if trace else None
    decoded=[]
    for index,row in enumerate(codes):
        end=row.index(BOUNDARY) if BOUNDARY in row else len(row)
        decoded.append(dict(text=bytes(code for code in row[:end] if code<256).decode('utf8',errors='replace'),
                            terminated=BOUNDARY in row))
        if trace:
            selected=score_rows[index][:end+1]
            decoded[-1].update(firstCode=row[0],firstCodeProbability=selected[0][0],
                firstCodeEntropyBits=selected[0][1],
                meanSelectedCodeProbability=sum(pair[0] for pair in selected)/len(selected),
                meanCodeEntropyBits=sum(pair[1] for pair in selected)/len(selected))
    return torch.stack(losses).mean(),decoded


def assess(core,samples,sensory,query,device,delay=.3,inject=True,batch_size=16,
           query_sensory=None,future_sensory=None):
    result=[]
    with torch.inference_mode():
        for at in range(0,len(samples),batch_size):
            selected=samples[at:at+batch_size];_,actual=run(core,selected,sensory,query,device,delay=delay,
                inject=inject,trace=True,query_sensory=query_sensory,future_sensory=future_sensory)
            result += [dict(expected=a,actual=b['text'],terminated=b['terminated'],
                            exact=a==b['text'] and b['terminated'],
                            firstCodeCorrect=b['firstCode']==a.encode('utf8')[0],
                            **{key:value for key,value in b.items() if key not in ['text','terminated']})
                       for a,b in zip(selected,actual)]
    return dict(total=len(result),exact=sum(row['exact'] for row in result),delaySeconds=delay,
                ownFeedbackPresent=inject,batchSize=batch_size,rows=result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--updates',type=int,default=1500)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/liquid-core-1791839440496083600/latest.pt')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--visual-inputs',action='store_true')
    parser.add_argument('--rehearsal-rate',type=float,default=0.)
    args=parser.parse_args()
    if args.updates<1:
        parser.error('Updates must be positive.')
    if not 0<=args.rehearsal_rate<=1:
        parser.error('Rehearsal rate must be between zero and one.')
    corpus=json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))
    selected=[row for row in corpus['cases'] if row['split']=='train'][:8]
    reserve=3_000_000_000+sum((ROOT/item['file']).stat().st_size for item in selected)
    with research_slot('own-code content-memory experiment',reserve_bytes=reserve):
        train(args,corpus,selected)


def train(args,corpus,selected):
    torch.set_num_threads(4);torch.manual_seed(1917);random.seed(1917)
    folder=ROOT/'artifacts'/f'liquid-copy-{time.time_ns()}';folder.mkdir();receipt=folder/'training.json'
    report=dict(state='preparing',publishedToLive=False)
    write_report(receipt,report)
    try:
        fit(args,corpus,selected,folder,receipt,report)
    except Exception as error:
        report.update(state='failed',error=str(error));write_report(receipt,report)
        raise


def fit(args,corpus,selected,folder,receipt,report):
    core,checkpoint=load_core(args.checkpoint,args.device);core.eval()
    for name,p in core.named_parameters():
        if name.startswith(('input_','write_','time_projection','initial','processing_')):
            p.requires_grad_(name.startswith('input_context'))
    views={name:value.to(args.device).float()[None] for name,value in backgrounds(corpus).items()}
    sensory=views['quiet']
    rehearsal=examples(selected) if args.rehearsal_rate or args.visual_inputs else []
    streams=[sensory]
    if args.visual_inputs:
        # Corpus-native packing is 68 rows before its optional literal text.
        # Recorded frames/audio vary over time; old question text is removed.
        for row in rehearsal[:2]:
            if any(len(event['tokens'])<68 for event in row['events'][60:90]):
                raise ValueError('Unexpected native sensory packet shape.')
            streams.append(torch.stack([event['tokens'][:68] for event in row['events'][60:90]]).to(args.device).float())
    query=literal_rows(['What did you say first?'])[0].to(args.device).float()
    train=words(2048,119,'abcdef');validation=words(128,120,'abcdef',train)
    report.update(state='training',architecture=core.architecture,publishedToLive=False,
        sourceCheckpoint=str(args.checkpoint),trainingStrings=len(train),validationStrings=len(validation),
        visualInputs=args.visual_inputs,rehearsalRate=args.rehearsal_rate,rehearsalEpisodes=len(rehearsal),
        inputPeriodSeconds=.1,injectionSlots=9,outputSlots=12,strictTerminationRequired=True,
        corpus='Random 3–8 character strings, no validation overlap',
        limits=['Content-memory capacity, not subjective self-awareness or voluntary intent.',
                'Injected strings are unscored; only subsequent generated readout is scored.',
                'No input transcript/cache/string lookup is available to the model.'])
    write_report(receipt,report)
    optimizer=torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],lr=.0003);started=time.monotonic()
    names=[name for name,p in core.named_parameters() if p.requires_grad]
    configuration=dict(visualInputs=args.visual_inputs,rehearsalRate=args.rehearsal_rate,
        corpusSha256=hashlib.sha256((ROOT/'artifacts/workspace-raw-corpus.json').read_bytes()).hexdigest(),
        protocol='fixed-9-feedback-12-output-aligned-query-v2')
    report['optimizerRestored']=restore_training(checkpoint,optimizer,names,configuration,args.device)
    report['trainingConfiguration']=configuration
    for update in range(1,args.updates+1):
        if psutil.virtual_memory().used>50_000_000_000:
            raise MemoryError('Total system RAM exceeds the configured budget.')
        optimizer.zero_grad(set_to_none=True)
        if random.random()<args.rehearsal_rate:
            task='audiovisual-conversation'
            loss,_,_,_=stream_rollout(core,random.choice(rehearsal),args.device,teacher=True,sampling=.15)
        else:
            task='own-content-memory'
            loss,_=run(core,random.sample(train,16),random.choice(streams),query,args.device,
                       teacher=True,delay=random.choice([.1,.3,.6]))
        loss.backward();torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update%25==0 or update==args.updates:
            report.update(update=update,task=task,loss=float(loss),elapsedSeconds=time.monotonic()-started,
                          systemUsedBytes=psutil.virtual_memory().used)
            if update%200==0 or update==args.updates:
                report['validation']=assess(core,validation,sensory,query,args.device)
                if args.visual_inputs:
                    report['visualValidation']={name:assess(core,validation[:32],views[name],query,args.device)
                                                for name in ['tennis','cycling']}
            temporary=folder/'latest.tmp';torch.save(dict(architecture=core.architecture,state_dict=core.state_dict(),update=update,
                trainingScope='Own-feedback content memory with configured sensory/rehearsal inputs',
                **capture_training(optimizer,names,configuration,args.device)),temporary);temporary.replace(folder/'latest.pt')
            write_report(receipt,report);print(json.dumps({key:report[key] for key in ['update','loss','elapsedSeconds']}),flush=True)
    report['delayValidation']=[assess(core,validation,sensory,query,args.device,delay=delay) for delay in [0.,1.,3.]]
    report['noOwnFeedbackControl']=assess(core,validation,sensory,query,args.device,inject=False)
    report['singleStreamValidation']=assess(core,validation[:16],sensory,query,args.device,batch_size=1)
    if args.rehearsal_rate:
        report['rehearsalValidation']=stream_evaluate(core,rehearsal,args.device)
    report['state']='complete';write_report(receipt,report)


if __name__=='__main__':
    main()
