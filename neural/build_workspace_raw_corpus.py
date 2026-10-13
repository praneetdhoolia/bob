"""Raw encoder-token sequences with answer labels kept out of model inputs.

Cache only deterministic vision encoding for repeated recording frames. Audio
remains sequential. Clear acoustic KV history after the spoken fact, so later
answers cannot obtain it from that encoder's short-term history.
"""
import hashlib
import json
from pathlib import Path
import random
import time
import types
import wave
import numpy as np
from PIL import Image
import torch
from omni_sensory_front import OmniSensoryFront


def read_audio(path):
    with wave.open(str(path),'rb') as wav:
        rate=wav.getframerate();channels=wav.getnchannels();assert wav.getsampwidth()==2
        samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
        samples=samples.reshape(-1,channels).mean(1)
    return np.interp(np.arange(int(len(samples)*16000/rate))/16000,np.arange(len(samples))/rate,samples).astype(np.float32)


def main():
    torch.set_num_threads(4)
    root=Path('artifacts')/f'workspace-raw-corpus-{time.time_ns()}';root.mkdir()
    path=Path('artifacts/workspace-raw-corpus.json')
    report={'state':'loading','architecture':OmniSensoryFront.architecture,'directory':str(root),'cases':[],
        'limits':['Offline cached vision encoding; not an end-to-end live latency test.',
            'Only two existing natural video sequences; broad visual transfer remains untested.',
            'Speech is synthesized; real human speech transfer remains untested.',
            'Answers are supervision and evaluation metadata, never sensory input.']}
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(path)
    save()
    try:
        front=OmniSensoryFront(chunk_ms=100);report.update(state='running',profile=front.profile());save()
        original=front.model.get_vision_embedding;vision_cache={};current_key=None;hits=0
        def cached(model,data):
            nonlocal hits
            if current_key in vision_cache:hits+=1;return vision_cache[current_key]
            value=original(data);vision_cache[current_key]=value;return value
        front.model.get_vision_embedding=types.MethodType(cached,front.model)
        rng=random.Random(412)
        cases=[]
        for split,fruits in [('train',['mango','apple','banana','pear']),('validation',['orange','grape'])]:
            for fruit in fruits:
                for mode in ['changing','held']:
                    cases.append({'split':split,'fruit':fruit,'mode':mode,'audio':True})
        for split in ['train','validation']:
            cases.append({'split':split,'fruit':'mango','mode':'changing','audio':False})
        frames=Path('artifacts/natural-stimuli-1791745722473036700')
        sequences={name:sorted((frames/name).glob('*.jpg')) for name in ['tennis','bmx-trees']}
        started=time.perf_counter()
        for number,case in enumerate(cases):
            front.reset();initial=front.initial_tokens.cpu().half();events=[];queries=[]
            audio_file=Path('artifacts/workspace-raw-speech')/(case['fruit']+'.wav');audio=read_audio(audio_file)
            first_query=rng.randint(63,78);second_query=rng.randint(96,108)
            order=['fruit','video'] if number%2==0 else ['video','fruit']
            query_ticks={first_query:order[0],second_query:order[1]}
            barrier=max(52,int(np.ceil(len(audio)/1600))+3)
            for tick in range(120):
                if tick==barrier:
                    front.model.audio_past_key_values=None
                    assert front.model.audio_past_key_values is None
                sequence='bmx-trees' if case['mode']=='changing' and tick>=50 else 'tennis'
                frame=sequences[sequence][(tick*3)%len(sequences[sequence])]
                image=Image.open(frame).convert('RGB')
                current_key=hashlib.sha256(image.tobytes()).hexdigest()
                chunk=np.zeros(1600,np.float32)
                if case['audio']:
                    part=audio[tick*1600:(tick+1)*1600];chunk[:len(part)]=part
                texts=[]
                if tick in query_ticks:
                    kind=query_ticks[tick]
                    if kind=='fruit':
                        question=('Which fruit did I say is my favourite?' if case['split']=='train' else 'Can you recall the fruit I mentioned earlier?')
                        answer=case['fruit'].capitalize()+'.' if case['audio'] else 'I did not hear a fruit.'
                    else:
                        question=('Describe what you see now.' if case['split']=='train' else 'What is happening in the scene at this moment?')
                        answer='A person is riding a bicycle.' if sequence=='bmx-trees' else 'A person is playing tennis.'
                    texts=[question];queries.append({'tick':tick,'kind':kind,'question':question,'answer':answer})
                tokens,event=front.step(audio=chunk,frames=[image],texts=texts)
                events.append({'tokens':tokens.cpu().half(),'dt':.1})
            file=root/f'case-{number:02d}.pt'
            # Correct labels appear only here, never in front.step(...).
            torch.save({'initial':initial,'events':events,'queries':queries,'metadata':case,
                'audioEncoderBarrierTick':barrier},file)
            item={'index':number,'file':str(file),'split':case['split'],'mode':case['mode'],
                'speechPresent':case['audio'],'fruit':case['fruit'],'ticks':len(events),'queries':queries,
                'audioEncoderBarrierTick':barrier,'speechSha256':hashlib.sha256(audio_file.read_bytes()).hexdigest()}
            report['cases'].append(item);report.update(visionCacheHits=hits,visionCacheEntries=len(vision_cache),
                elapsedSeconds=time.perf_counter()-started);save()
            print(json.dumps({'case':number,'split':case['split'],'ticks':120,'elapsedSeconds':report['elapsedSeconds']}),flush=True)
        report.update(state='complete',audioEncoderBarrierApplied=True);save()
    except Exception as error:
        report.update(state='failed',errorType=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
