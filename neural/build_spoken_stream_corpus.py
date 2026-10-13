"""Raw multimodal episodes with spoken questions and controlled transfer splits.

Native encoders see PCM/frame/literal-keyboard inputs. Spoken-question
annotations are offline targets and never passed as text to the frontend.
"""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import types

import numpy as np
from PIL import Image
import psutil
import torch

from build_workspace_raw_corpus import read_audio
from omni_sensory_front import OmniSensoryFront
from train_liquid_core import write_report

ROOT = Path(__file__).resolve().parents[1]
PHRASES = {
    'fruit': ['Which fruit did I say is my favourite?', 'What fruit did I mention earlier?',
              'Can you recall the fruit I mentioned earlier?'],
    'video': ['Describe what you see now.', 'What is happening in front of you?',
              'What is happening in the scene at this moment?']}


def fixtures(folder):
    items = []
    for speaker, voice in [('david','Microsoft David Desktop'), ('zira','Microsoft Zira Desktop')]:
        for fruit in ['mango','apple','banana','pear']:
            items.append(dict(voice=voice, text=f'My favourite fruit is {fruit}. Please remember it.',
                              path=str(folder/f'{speaker}-{fruit}.wav')))
        for kind, variants in PHRASES.items():
            for number, text in enumerate(variants):
                items.append(dict(voice=voice, text=text, path=str(folder/f'{speaker}-{kind}-{number}.wav')))
    manifest = folder/'speech-manifest.json'
    manifest.write_text(json.dumps(items, indent=2), encoding='utf8')
    subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',
                    str(ROOT/'scripts/make-spoken-fixtures.ps1'),'-Manifest',str(manifest)], check=True)


def specification():
    rows = []
    for speaker, split in [('david','train'), ('zira','validation')]:
        for fruit_index, fruit in enumerate(['mango','apple','banana','pear']):
            for mode_index, mode in enumerate(['held','changing']):
                for channel_index, channel in enumerate(['spoken','text']):
                    rows.append(dict(speaker=speaker, split=split, fruit=fruit, mode=mode,
                        questionChannel=channel, phraseIndex=(fruit_index+mode_index)%2, audio=True,
                        questionOrder=(fruit_index+mode_index+channel_index)%2,
                        transfer='held-speaker' if split=='validation' and channel=='spoken' else 'familiar-phrasing'))
        for channel in ['spoken','text']:
            rows.append(dict(speaker=speaker, split=split, fruit='mango', mode='changing',
                             questionChannel=channel, phraseIndex=0, questionOrder=len(rows)%2,
                             audio=False, transfer='absent-fact'))
    # Isolate wording transfer from voice transfer: training speaker, third phrase.
    for fruit in ['mango','apple','banana','pear']:
        for channel in ['spoken','text']:
            rows.append(dict(speaker='david', split='validation', fruit=fruit, mode='changing',
                             questionChannel=channel, phraseIndex=2, questionOrder=len(rows)%2,
                             audio=True, transfer='held-phrasing'))
    return rows


def main():
    torch.set_num_threads(4)
    folder = ROOT/'artifacts'/f'spoken-stream-corpus-{time.time_ns()}'
    folder.mkdir(); audio_folder = folder/'audio'; audio_folder.mkdir()
    receipt = ROOT/'artifacts/spoken-stream-corpus.json'
    report = dict(state='making-audio', directory=str(folder.relative_to(ROOT)), cases=[],
        pretrainedLanguageForwardCalls=0, sentSpokenQueryTranscript=False,
        limits=['Synthetic voices; real human speech remains untested.',
                'Two existing recorded scenes; broad visual transfer remains untested.',
                'Held speaker and held phrasing are separate conditions.',
                'Labels are offline supervision; no generated caption or ASR is an input.'])
    write_report(receipt, report); fixtures(audio_folder)
    if psutil.virtual_memory().used > 50_000_000_000:
        raise MemoryError('Total system RAM exceeds the configured budget.')
    front = OmniSensoryFront(chunk_ms=100)
    front.model.llm.model.register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('Language trunk execution is forbidden.')))
    front.duplex.processor.image_processor.scale_resolution = 224
    front.model.processor.image_processor.scale_resolution = 224
    original = front.model.get_vision_embedding; cache = {}; current_key = None
    def cached(model, data):
        if current_key not in cache:
            cache[current_key] = original(data)
        return cache[current_key]
    front.model.get_vision_embedding = types.MethodType(cached, front.model)
    frame_folder = ROOT/'artifacts/natural-stimuli-1791745722473036700'
    frames = {name: sorted((frame_folder/name).glob('*.jpg')) for name in ['tennis','bmx-trees']}
    rng = random.Random(972); started = time.monotonic()
    report.update(state='encoding', profile=front.profile()); write_report(receipt, report)
    for number, spec in enumerate(specification()):
        front.reset(); events = []; queries = []
        fact = read_audio(audio_folder/f"{spec['speaker']}-{spec['fruit']}.wav")
        streams = [(0,fact)] if spec['audio'] else []
        order = ['fruit','video'] if spec['questionOrder']==0 else ['video','fruit']
        question_ticks = {}; starts = [max(rng.randint(60,74),len(fact)//1600+5), rng.randint(130,140)]
        for kind, start in zip(order, starts):
            spoken = read_audio(audio_folder/f"{spec['speaker']}-{kind}-{spec['phraseIndex']}.wav")
            end = start+int(np.ceil(len(spoken)/1600))-1
            if spec['questionChannel']=='spoken':
                streams.append((start*1600,spoken))
            question_ticks[end] = kind
        barrier = max(52,int(np.ceil(len(fact)/1600))+3)
        switch_tick = rng.randint(35,70)
        for tick in range(200):
            if psutil.virtual_memory().used > 50_000_000_000:
                raise MemoryError('Total system RAM exceeds the configured budget.')
            if tick==barrier:
                front.model.audio_past_key_values = None
            scene = 'bmx-trees' if spec['mode']=='changing' and tick>=switch_tick else 'tennis'
            with Image.open(frames[scene][(tick*3)%len(frames[scene])]) as picture:
                image = picture.convert('RGB')
            current_key = hashlib.sha256(image.tobytes()).hexdigest()
            wave = np.zeros(1600,np.float32)
            for offset, samples in streams:
                source_start = tick*1600-offset
                if 0 <= source_start < len(samples):
                    part = samples[source_start:source_start+1600]; wave[:len(part)] += part
            texts = []
            if tick in question_ticks:
                kind = question_ticks[tick]
                answer = (spec['fruit'].capitalize()+'.' if spec['audio'] else 'I did not hear a fruit.') if kind=='fruit' else (
                    'A person is riding a bicycle.' if scene=='bmx-trees' else 'A person is playing tennis.')
                queries.append(dict(tick=tick, kind=kind, answer=answer))
                if spec['questionChannel']=='text':
                    texts = [PHRASES[kind][spec['phraseIndex']]]
            tokens, _ = front.step(audio=wave, frames=[image], texts=texts)
            events.append(dict(dt=.1,tokens=tokens.cpu().half()))
        path = folder/f'case-{number:03d}.pt'
        torch.save(dict(events=events,queries=queries,metadata=spec),path)
        report['cases'].append(dict(spec,file=str(path.relative_to(ROOT)),ticks=len(events),
                                    queries=queries,audioEncoderBarrierTick=barrier))
        report.update(elapsedSeconds=time.monotonic()-started,systemUsedBytes=psutil.virtual_memory().used)
        write_report(receipt, report)
        print(json.dumps(dict(case=number,split=spec['split'],channel=spec['questionChannel'],
                              transfer=spec['transfer'],elapsedSeconds=report['elapsedSeconds'])), flush=True)
    report['state'] = 'complete'; write_report(receipt, report)


if __name__ == '__main__':
    try:
        from research_resources import research_slot
        with research_slot('spoken corpus encoding'):
            main()
    except Exception as error:
        receipt = ROOT/'artifacts/spoken-stream-corpus.json'
        if receipt.exists():
            report = json.loads(receipt.read_text(encoding='utf8'))
            report.update(state='failed', error=str(error)); write_report(receipt, report)
        raise
