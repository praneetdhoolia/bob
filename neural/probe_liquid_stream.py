"""Matched unfinished-byte and strict acoustic-memory interventions."""
import json
from pathlib import Path
import torch
from liquid_core import BLANK,BOUNDARY,load_core

ROOT=Path(__file__).resolve().parents[1]


def finish(model,state,future=None,changing=True,limit=48,phase=0):
    codes=[]
    for offset in range(limit):
        state=model.advance(state,.03)
        position=offset+phase
        if future is not None and ((position>0 and position%5==0) or (offset==0 and phase>0)):
            index=min(max(position//5-1,0),len(future[0])-1)
            tokens=future[0][index].clone()
            if not changing:tokens[2:66]=future[1][index][2:66]
            state=model.observe(state,tokens,.15)
        code=int(model.logits(state).argmax(-1))
        if code==BOUNDARY:break
        if code!=BLANK:
            codes.append(code);state=model.feedback(state,code)
    return bytes(codes).decode('utf8',errors='replace')


def main():
    torch.set_num_threads(4)
    info=json.loads((ROOT/'artifacts/liquid-core-training.json').read_text(encoding='utf8'))
    model,_=load_core(ROOT/info['directory']/'latest.pt');model.eval()
    metadata=json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text(encoding='utf8'))['cases']
    held=next(row for row in metadata if row['mode']=='held' and row['fruit']=='mango' and row['speechPresent'])
    held=torch.load(ROOT/held['file'],weights_only=False)
    result={'architecture':model.architecture,'unfinished':[],'strictSpeech':[],'agedQuery':[],'pretrainedLanguageForwardCalls':0}
    with torch.inference_mode():
        for record in [row for row in metadata if row['mode']=='changing' and row['speechPresent']][:6]:
            case=torch.load(ROOT/record['file'],weights_only=False)
            query=next(q for q in case['queries'] if q['kind']=='video');tick=query['tick']
            state=model.start().clone()
            for event in case['events'][:tick+1]:state=model.observe(model.advance(state,event['dt']),event['tokens'],event['dt'])
            prefix='A person is '
            current=[case['events'][min(tick+i+1,len(case['events'])-1)]['tokens'] for i in range(9)]
            steady=[held['events'][min(tick+i+1,len(held['events'])-1)]['tokens'] for i in range(9)]
            for position,code in enumerate(prefix.encode()):
                state=model.advance(state,.03)
                if position and position%5==0:state=model.observe(state,current[min(position//5-1,8)],.15)
                state=model.feedback(state,code)
            a=prefix+finish(model,state.clone(),(current,steady),True,phase=len(prefix))
            b=prefix+finish(model,state.clone(),(current,steady),False,phase=len(prefix))
            result['unfinished'].append({'fruit':record['fruit'],'changing':a,'held':b,'different':a!=b,
                'inputControl':'Only rows 2:66 replaced; all audio, text, feedback, starting state and elapsed intervals match.'})
        base=torch.load(ROOT/metadata[0]['file'],weights_only=False)
        neutral_record=next(row for row in metadata if not row['speechPresent'])
        neutral=torch.load(ROOT/neutral_record['file'],weights_only=False)
        query=next(q for q in base['queries'] if q['kind']=='fruit')
        for fruit in ['mango','apple','banana','pear','orange','grape']:
            source=next(row for row in metadata if row['fruit']==fruit and row['mode']=='changing' and row['speechPresent'])
            source=torch.load(ROOT/source['file'],weights_only=False)
            state=model.start().clone()
            for tick,event in enumerate(base['events'][:query['tick']+1]):
                tokens=event['tokens'].clone()
                if tick<52:tokens[67:68]=source['events'][tick]['tokens'][67:68]
                else:tokens[67:68]=neutral['events'][tick]['tokens'][67:68]
                state=model.observe(model.advance(state,.1),tokens,.1)
            future=[]
            for offset in range(9):
                tick=min(query['tick']+offset+1,len(base['events'])-1)
                tokens=base['events'][tick]['tokens'].clone();tokens[67:68]=neutral['events'][tick]['tokens'][67:68]
                future.append(tokens)
            text=finish(model,state,(future,future))
            result['strictSpeech'].append({'fruit':fruit,'actual':text,'correct':text==fruit.capitalize()+'.',
                                          'control':'Only early native acoustic row changed; every input after tick 52 is identical.'})
        question=next(q for q in base['queries'] if q['kind']=='video')['tick']
        state=model.start().clone()
        for event in base['events'][:question]:state=model.observe(model.advance(state,.1),event['tokens'],.1)
        for age in [0.,60.,300.,900.]:
            aged=model.advance(state.clone(),age)
            aged=model.observe(aged,base['events'][question]['tokens'],.1)
            result['agedQuery'].append({'extraQuietSeconds':age,'actual':finish(model,aged)})
    result['strictSpeechCorrect']=sum(row['correct'] for row in result['strictSpeech'])
    result['unfinishedDifferent']=sum(row['different'] for row in result['unfinished'])
    (ROOT/'artifacts/liquid-stream-controls.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
