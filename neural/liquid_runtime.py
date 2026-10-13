"""Own one continuous state; numerical input and byte output share it."""
import base64
from collections import deque
import io
import json
from pathlib import Path
import queue
import os
import psutil
import threading
import time

import numpy as np
from PIL import Image
import torch

from liquid_core import LiquidCore, BLANK, BOUNDARY, load_core
from omni_sensory_front import OmniSensoryFront
from graphed_omni_vision import GraphedOmniVision
from urge import UrgeGate, stimulus_summary

MAX_MESSAGE_CHARACTERS = 2048


class LiquidRuntime:
    def __init__(self, root, state_path=None, checkpoint_override=None):
        self.root = root
        self.checkpoint_override = Path(checkpoint_override).resolve() if checkpoint_override else None
        if self.checkpoint_override and not self.checkpoint_override.is_relative_to((root/'artifacts').resolve()):
            raise ValueError('Candidate checkpoint is outside artifacts.')
        self.state_path = state_path or root/'artifacts/liquid-live-state.pt'
        self.ram_budget = 50_000_000_000
        self.process = psutil.Process(os.getpid())
        self.resource_at = 0.; self.compute = {}
        self.process.cpu_percent(); psutil.cpu_percent()
        self.lock = threading.RLock()
        self.save_lock = threading.Lock()
        self.running = True; self.ready = False; self.active = True
        self.source = 'quiet'; self.audio = deque(); self.frames = queue.Queue(16); self.text = queue.Queue(32)
        self.events = deque(maxlen=4096); self.event_id = 0; self.messages = deque(maxlen=100)
        self.state = None; self.model = None; self.latest_jpeg = None; self.last_frame = None
        self.clock_at = time.monotonic(); self.demo_at = self.clock_at
        self.utterance = ''; self.utterance_id = 0; self.pending_bytes = bytearray()
        self.urge = UrgeGate()
        self.stats = {'state':'loading', 'physicalSeconds':0., 'ticks':0, 'observations':0,
                      'emittedBytes':0,'deadlineMisses':0,'audioSamplesReceived':0,'framesReceived':0,
                      'rejectedAudioSamples':0,'rejectedFrames':0,'supersededFrames':0,'focus':[],
                      'pretrainedLanguageForwardCalls':0,'weightsUpdate':0,
                      'limits':['Custom translator trained on narrow recordings; broad English is untrained.',
                                'Autonomous numerical activity is not verified deliberation or consciousness.',
                                'No motor policy is connected. Physical camera/microphone transfer needs validation.']}
        self.worker = threading.Thread(target=self.perceive, daemon=True, name='liquid-input')
        self.worker.start()

    def event(self, kind, **data):
        with self.lock:
            self.event_id += 1
            self.events.append({'id':self.event_id,'type':kind,'data':data,'time':time.time()})

    def snapshot(self, full=False, include_messages=True):
        with self.lock:
            result = dict(self.stats)
            result.update(ready=self.ready,active=self.active,source=self.source,
                          videoAgeSeconds=time.time()-self.stats['lastVideoObservedAtUnix'] if self.stats.get('lastVideoObservedAtUnix') else None,
                          queues={'frames':self.frames.qsize(),'audioSamples':len(self.audio),'text':self.text.qsize()},
                          lastEvent=self.event_id)
            if include_messages:result['messages']=list(self.messages)
            memory=psutil.virtual_memory()
            if time.monotonic()-self.resource_at >= 1:
                self.compute.update(agentCpuPercent=self.process.cpu_percent()/max(1,psutil.cpu_count()),
                                    systemCpuPercent=psutil.cpu_percent())
                self.resource_at=time.monotonic()
            result['compute']=dict(self.compute)
            result['memory']={'agentRssBytes':self.process.memory_info().rss,'systemUsedBytes':memory.used,
                              'systemTotalBytes':memory.total,
                              'systemAvailableBytes':memory.available,'systemBudgetBytes':self.ram_budget,
                              'withinSystemBudget':memory.used<=self.ram_budget}
            if self.state is not None:
                values = self.state.detach().float().cpu().numpy()
                result['matrixShape'] = list(values.shape[1:])
                result['matrixProjection'] = values.reshape(256,-1).mean(-1).tolist()
                result['matrixRms'] = float(np.sqrt(np.mean(values**2)))
                result['matrixRange'] = [float(values.min()),float(values.max())]
                result['matrixBytes'] = values.nbytes
                if full:
                    result['matrixFloat32'] = base64.b64encode(values.astype('<f4').tobytes()).decode()
            return result

    def receive_audio(self, samples):
        if not np.isfinite(samples).all() or not 0 < len(samples) <= 16000:
            raise ValueError('Expected at most one second of finite audio.')
        with self.lock:
            if len(self.audio)+len(samples)>48000:
                self.stats['rejectedAudioSamples'] += len(samples)
                raise queue.Full('Audio queue full.')
            self.audio.extend(samples.clip(-1,1).tolist())
            self.stats['audioSamplesReceived'] += len(samples)

    def receive_frame(self, image):
        if image.width*image.height>4_000_000:
            raise ValueError('Frame exceeds four million pixels.')
        frame=(image.convert('RGB').copy(),time.monotonic())
        with self.lock:
            try:self.frames.put_nowait(frame)
            except queue.Full:
                self.frames.get_nowait()
                self.frames.put_nowait(frame)
                self.stats['supersededFrames']=self.stats.get('supersededFrames',0)+1
            self.stats['framesReceived'] += 1

    def latest_frame(self, now):
        with self.lock:
            consumed=0
            while True:
                try:self.last_frame=self.frames.get_nowait();consumed+=1
                except queue.Empty:break
            self.stats['supersededFrames']+=max(0,consumed-1)
            return self.last_frame[0] if self.last_frame and now-self.last_frame[1]<.5 else None

    def message(self, text):
        if not isinstance(text,str) or not 0<len(text.strip())<=4000:
            raise ValueError('Enter 1 to 4000 characters.')
        self.text.put_nowait(text.strip())
        row={'source':'user','text':text.strip(),'time':time.time()}
        with self.lock:self.messages.append(row)
        self.event('message',**row)

    def set_source(self, source):
        if source not in ['quiet','recorded','camera','video']:
            raise ValueError('Unknown source.')
        with self.lock:
            if source=='recorded' and (not self.ready or not all(self.images.values())):
                raise ValueError('Test recording unavailable. Choose a camera or local video.')
            while True:
                try:self.frames.get_nowait()
                except queue.Empty:break
            self.last_frame=None;self.latest_jpeg=None;self.stats['focus']=[]
            self.source=source;self.demo_at=time.monotonic()
        self.event('source',source=source)

    def set_active(self, active):
        if not isinstance(active,bool):raise ValueError('Active must be boolean.')
        with self.lock:
            if active and self.stats.get('state')=='failed':
                raise ValueError('The numerical runtime failed. Resolve the reported error and restart.')
            if active and self.ready and self.source=='recorded' and not all(self.images.values()):
                raise ValueError('Test recording unavailable. Choose a camera or local video.')
            if active and psutil.virtual_memory().used>self.ram_budget:
                raise ValueError('Total system RAM exceeds the configured budget. Free memory to resume.')
            self.advance_locked(time.monotonic());self.active=active

    def advance_locked(self, now):
        elapsed=max(0.,now-self.clock_at);self.clock_at=now
        if self.active and self.state is not None:
            self.state=self.model.advance(self.state,elapsed)
            self.stats['physicalSeconds']+=elapsed;self.stats['ticks']+=1

    def clock(self):
        due=time.monotonic();translated=due
        try:
            with torch.inference_mode():
                while self.running:
                    now=time.monotonic()
                    if now<due:time.sleep(min(.005,due-now));continue
                    due=now+.01
                    with self.lock:
                        self.advance_locked(now)
                        if not self.active or now-translated<.03:continue
                        since=now-translated;translated=now
                        probabilities=self.model.logits(self.state).softmax(-1)[0]
                        code=int(probabilities.argmax())
                        self.stats['translationCode']=code;self.stats['translationConfidence']=float(probabilities[code])
                        self.stats['translationTop']=[{'code':int(i),'probability':float(probabilities[i])} for i in probabilities.topk(5).indices]
                        self.urge.advance(min(since,1.),probabilities)
                        onset=not self.urge.speaking
                        permitted=self.urge.permits(code)
                        self.stats.update(self.urge.measurement())
                        if not permitted:continue
                        if onset:
                            # The released stimulus is the readout the translator interprets.
                            self.event('stimulus',utteranceId=self.utterance_id,urge=self.urge.urge,
                                       readiness=self.urge.readiness,surprise=self.urge.surprise,
                                       **stimulus_summary(self.model,self.state,probabilities))
                        self.emit_code_locked(code)
        except Exception as error:self.fail(error)

    def finish_output_locked(self, reason):
        if self.utterance:
            row={'id':self.utterance_id,'source':'core','text':self.utterance,'time':time.time()}
            self.messages.append(row);self.event('output',**row,ended=True,framing=reason)
        self.utterance='';self.pending_bytes.clear();self.utterance_id+=1
        # Transport framing splits a long utterance; only a learned boundary spends the urge.
        if reason!='transport-limit':self.urge.finished()

    def emit_code_locked(self, code):
        self.state=self.model.feedback(self.state,code)
        if code==BOUNDARY:
            if self.pending_bytes:
                self.utterance+=self.pending_bytes.decode('utf8',errors='replace')
            self.finish_output_locked('model-boundary')
            return
        self.stats['emittedBytes']+=1
        self.pending_bytes.append(code)
        try:part=self.pending_bytes.decode('utf8')
        except UnicodeDecodeError as error:
            if error.reason=='unexpected end of data' and len(self.pending_bytes)<4:return
            part=self.pending_bytes.decode('utf8',errors='replace')
        self.pending_bytes.clear();self.utterance+=part
        self.event('output',id=self.utterance_id,source='core',text=self.utterance,part=part,ended=False)
        if len(self.utterance)>=MAX_MESSAGE_CHARACTERS:
            # Transport framing only: no synthetic boundary feeds into the model.
            self.finish_output_locked('transport-limit')

    def fail(self, error):
        with self.lock:self.stats.update(state='failed',error=str(error));self.active=False;self.ready=False
        self.event('error',error=str(error))

    def checkpoint_path(self):
        if self.checkpoint_override:
            return self.checkpoint_override, {'update': 0}
        info=json.loads((self.root/'artifacts/liquid-core-training.json').read_text(encoding='utf8'))
        path=(self.root/info['directory']/'latest.pt').resolve()
        if not path.is_relative_to((self.root/'artifacts').resolve()):raise ValueError('Checkpoint is outside artifacts.')
        return path,info

    def weights_path(self):
        return self.state_path.with_name(self.state_path.stem+'.weights.pt')

    @staticmethod
    def durable_save(payload, path):
        temporary=path.with_suffix('.tmp')
        with open(temporary,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)

    def save_state(self):
        """Save the 64 KB state each time; rewrite weights only when they change."""
        if self.model is None or self.state is None:return
        with self.save_lock:
            with self.lock:
                model=self.model
                identity=(getattr(self,'model_generation',0),self.stats['weightsUpdate'],model.architecture,int(model.internal_scheme))
                weights=None
                if identity!=getattr(self,'saved_weights_identity',None):
                    weights={'architecture':model.architecture,
                             'modelState':{key:value.detach().cpu().clone() for key,value in model.state_dict().items()}}
                payload={'architecture':model.architecture,'state':self.state.detach().cpu().clone(),
                         'physicalSeconds':self.stats['physicalSeconds'],'weightsUpdate':self.stats['weightsUpdate'],
                         'weightsFile':self.weights_path().name,
                         'utterance':self.utterance,'utteranceId':self.utterance_id,'pendingBytes':bytes(self.pending_bytes),
                         'messages':list(self.messages),'source':self.source,'demoElapsed':max(0.,time.monotonic()-self.demo_at),
                         'stats':dict(self.stats),'eventId':self.event_id,'savedAtUnix':time.time()}
            # Weights are written before the state that refers to them.
            if weights is not None:
                self.durable_save(weights,self.weights_path());self.saved_weights_identity=identity
            self.durable_save(payload,self.state_path)

    def save_quietly(self):
        """Autosave must never stop the agent; report the I/O error instead."""
        try:
            self.save_state()
            with self.lock:self.stats.pop('saveError',None)
        except (OSError,RuntimeError) as error:
            with self.lock:self.stats['saveError']=str(error)
            self.event('input-error',path='autosave',error=f'Autosave failed: {error}')

    def restore_state(self, resume):
        source=resume.get('source','quiet')
        state=resume['state']
        if source not in ['quiet','recorded','camera','video']:
            raise ValueError('Saved source is invalid.')
        if not isinstance(state,torch.Tensor) or state.shape!=self.model.start().shape or not torch.isfinite(state).all():
            raise ValueError('Saved numerical state is invalid.')
        if 'modelState' in resume:
            weights=resume  # Older single-file saves embed their weights.
        else:
            weights=torch.load(self.state_path.parent/Path(resume['weightsFile']).name,map_location='cpu',weights_only=False)
        self.model.restore_weights(weights['modelState'],weights['architecture'])
        if self.model._active_internal:
            raise ValueError('Experimental internal processing has not passed deployment checks.')
        self.state=state.clone();self.active=False;self.source=source
        self.demo_at=time.monotonic()-max(0.,resume.get('demoElapsed',0.))
        self.utterance=resume['utterance'][-MAX_MESSAGE_CHARACTERS:];self.utterance_id=resume['utteranceId']
        self.pending_bytes=bytearray(resume['pendingBytes']);self.messages.clear()
        self.messages.extend(dict(row,text=row['text'][-(4000 if row['source']=='user' else MAX_MESSAGE_CHARACTERS):])
                             for row in resume['messages'][-100:])
        self.event_id=resume.get('eventId',0)
        self.urge.speaking=bool(self.utterance or self.pending_bytes)
        self.stats.update(resume.get('stats',{}));self.stats['physicalSeconds']=resume['physicalSeconds']

    def close(self):
        self.set_active(False);self.running=False
        if threading.current_thread() is not self.worker:self.worker.join(timeout=5)
        self.save_state()

    def perceive(self):
        try:
            torch.set_num_threads(4)
            while self.running and psutil.virtual_memory().used>self.ram_budget:
                # Wait rather than fail: memory used by other programs can fall.
                with self.lock:self.stats['state']='waiting-for-memory'
                time.sleep(1)
            while self.running:
                try:
                    path,info=self.checkpoint_path()
                    if path.exists():break
                except (OSError,ValueError,KeyError):pass
                time.sleep(.2)
            if not self.running:return
            self.model,checkpoint=load_core(path)
            if self.model._active_internal:
                raise ValueError('Experimental internal processing has not passed deployment checks.')
            self.model.eval().requires_grad_(False)
            self.front=OmniSensoryFront();self.graphs=GraphedOmniVision(self.front.model,fast_attention=True)
            self.compute['gpuTotalBytes']=torch.cuda.get_device_properties(0).total_memory
            self.front.duplex.processor.image_processor.scale_resolution=224
            self.front.model.processor.image_processor.scale_resolution=224
            self.front.model.llm.model.register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('Language trunk execution is forbidden.')))
            folder=self.root/'artifacts/natural-stimuli-1791745722473036700'
            self.images={name:sorted((folder/name).glob('*.jpg')) for name in ['tennis','bmx-trees']}
            from build_workspace_raw_corpus import read_audio
            speech_path=self.root/'artifacts/workspace-raw-speech/mango.wav'
            try:self.speech=read_audio(speech_path) if speech_path.exists() else np.empty(0,np.float32)
            except (OSError,ValueError,RuntimeError) as error:
                self.speech=np.empty(0,np.float32)
                self.stats['demoAudioError']=str(error)
            self.front.step(audio=np.zeros(1600,np.float32));self.front.reset()
            with self.lock:
                self.state=self.model.start().detach().clone();self.clock_at=time.monotonic()
                saved=self.state_path
                resume=None
                if saved.exists():
                    try:
                        resume=torch.load(saved,map_location='cpu',weights_only=False)
                        self.restore_state(resume)
                    except Exception as error:
                        # Keep the unreadable file for inspection; start from the trained initial state.
                        quarantine=saved.with_name(f'{saved.stem}.corrupt-{int(time.time())}.pt')
                        saved.replace(quarantine);resume=None
                        self.model.restore_weights(checkpoint['state_dict'],checkpoint['architecture'])
                        self.state=self.model.start().detach().clone();self.active=False;self.source='quiet'
                        # Discard anything partially restored from the unreadable file.
                        self.utterance='';self.utterance_id=0;self.pending_bytes=bytearray()
                        self.messages.clear();self.event_id=max(self.event_id,0);self.urge.speaking=False
                        self.stats['restoreError']=f'{error} (moved to {quarantine.name})'
                        self.event('input-error',path='restore',error=self.stats['restoreError'])
                self.stats.pop('error',None)
                self.ready=True;self.stats.update(state='running',parameters=self.model.parameter_count(),
                    weightsUpdate=resume['weightsUpdate'] if resume else checkpoint['update'],architecture=LiquidCore.architecture,
                    profile=self.front.profile(),checkpoint=resume.get('stats',{}).get('checkpoint',str(path.relative_to(self.root))) if resume else str(path.relative_to(self.root)))
            threading.Thread(target=self.clock,daemon=True,name='liquid-clock').start()
            with torch.inference_mode():self.run_inputs()
        except Exception as error:self.fail(error)

    def run_inputs(self):
        due=time.monotonic();last_input=due;loaded_at=due;pending=deque()
        while self.running:
            if not self.active:time.sleep(.02);due=time.monotonic();last_input=due;continue
            now=time.monotonic()
            if now<due:time.sleep(min(.005,due-now));continue
            due=max(due+.1,now);begin=now;interval=now-last_input;last_input=now
            with self.lock:
                source=self.source;elapsed=now-self.demo_at
                samples=np.array([self.audio.popleft() for _ in range(min(1600,len(self.audio)))],np.float32)
            wave=np.zeros(1600,np.float32);wave[:len(samples)]=samples
            picture=None
            if source=='recorded' and not all(self.images.values()):
                self.set_source('quiet')
                self.event('input-error',path='source',error='Test recording unavailable; switched to quiet input.')
                continue
            if source=='recorded':
                name='tennis' if int(elapsed/2)%2==0 else 'bmx-trees';paths=self.images[name]
                try:picture=Image.open(paths[int(elapsed*30)%len(paths)]).convert('RGB')
                except OSError as error:
                    self.set_source('quiet')
                    self.event('input-error',path='source',error=f'Test recording frame unreadable ({error}); switched to quiet input.')
                    continue
                part=self.speech[int(elapsed*16000):int(elapsed*16000)+1600]
                if not len(samples):wave[:len(part)]=part
            elif source in ['camera','video']:
                picture=self.latest_frame(now)
            while len(pending)<4096 and not self.text.empty():pending.extend(self.front.duplex.tokenizer.encode(self.text.get_nowait(),add_special_tokens=False))
            ids=[pending.popleft() for _ in range(min(128,len(pending)))]
            texts=[self.front.duplex.tokenizer.decode(ids)] if ids else []
            if self.stats['observations'] and self.stats['observations']%256==0:
                self.front.reset()
                self.stats['encoderCacheResets']=self.stats.get('encoderCacheResets',0)+1
            tokens,encoding=self.front.step(audio=wave,frames=[picture] if picture else [],texts=texts)
            with self.lock:
                if not self.active:continue
                self.advance_locked(time.monotonic());previous=self.state.clone()
                packet=tokens.cpu().float()
                if packet.shape[-2]:self.urge.observe_packet(self.model.sensory_target(packet))
                self.state=self.model.observe(self.state,packet,interval)
                code=int(self.model.logits(self.state).argmax(-1))
            focus=[]
            if picture and self.graphs.last_patch_attention is not None:
                from liquid_focus import measure_focus
                try:focus=measure_focus(self.model,previous,tokens,interval,code,self.graphs.last_patch_attention)
                except (RuntimeError,ValueError,IndexError) as error:
                    # The focus overlay is diagnostic; it must not stop perception.
                    with self.lock:self.stats['focusError']=str(error)
            if picture:
                stream=io.BytesIO();picture.save(stream,format='JPEG',quality=78)
                with self.lock:self.latest_jpeg=stream.getvalue()
            duration=time.monotonic()-begin
            with self.lock:
                self.stats.update(observations=self.stats['observations']+1,focus=focus,
                    audioRms=float(np.sqrt(np.mean(wave**2))),
                    audioEnvelope=np.sqrt(np.mean(wave.reshape(40,40)**2,axis=1)).tolist(),
                    latestObservationMs=duration*1000,inputIntervalMs=interval*1000,
                    encoderMs=encoding['seconds']*1000,
                    encodedRows=int(tokens.shape[-2]),
                    latestModalities={'audioSamples':1600,'videoFrames':int(picture is not None),'textRows':len(ids)},
                    gpuAllocatedBytes=torch.cuda.memory_allocated(),gpuReservedBytes=torch.cuda.memory_reserved())
                if picture:self.stats['lastVideoObservedAtUnix']=time.time()
                if duration>.1:self.stats['deadlineMisses']+=1
            self.event('sample',**self.snapshot(include_messages=False))
            if psutil.virtual_memory().used>self.ram_budget:
                self.set_active(False)
                self.event('error',error='Paused: total system RAM exceeds the configured budget. Free memory to resume.')
            if now-loaded_at>5:
                loaded_at=now
                try:
                    path,info=self.checkpoint_path()
                    if info.get('update',0)>self.stats['weightsUpdate'] or str(path.relative_to(self.root))!=self.stats['checkpoint']:
                        if not path.exists():raise FileNotFoundError('Selected checkpoint is missing.')
                        candidate,newer=load_core(path)
                        if candidate._active_internal:
                            raise ValueError('Experimental internal processing has not passed deployment checks.')
                        candidate.eval().requires_grad_(False)
                        update=newer['update'];selected=str(path.relative_to(self.root))
                        with self.lock:
                            self.model=candidate;self.model_generation=getattr(self,'model_generation',0)+1
                            self.stats['weightsUpdate']=update;self.stats['checkpoint']=selected
                            self.stats.pop('checkpointError',None)
                except (OSError,ValueError,RuntimeError,KeyError) as error:
                    with self.lock:
                        self.stats['checkpointError']=str(error)
                    self.event('input-error',path='checkpoint',error=str(error))
                self.save_quietly()
