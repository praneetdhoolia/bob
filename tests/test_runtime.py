"""Observable recovery, concurrency and framing without sensory-weight loading."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import sys
import threading
import time
import unittest
import uuid

import numpy as np
from PIL import Image
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'neural'))
from liquid_core import LiquidCore,BOUNDARY
from liquid_runtime import LiquidRuntime,MAX_MESSAGE_CHARACTERS
from liquid_focus import rectangles
from graphed_omni_vision import GraphedOmniVision


def fixture():
    runtime=LiquidRuntime.__new__(LiquidRuntime)
    runtime.root=ROOT;runtime.lock=threading.RLock();runtime.save_lock=threading.Lock()
    runtime.model=LiquidCore().eval().requires_grad_(False);runtime.state=runtime.model.start().detach().clone()
    runtime.process=psutil.Process();runtime.resource_at=0.;runtime.compute={}
    runtime.clock_at=time.monotonic();runtime.demo_at=runtime.clock_at-24.
    runtime.running=False;runtime.ready=True;runtime.active=False;runtime.source='video'
    runtime.ram_budget=50_000_000_000;runtime.frames=queue.Queue(2);runtime.last_frame=None;runtime.latest_jpeg=None
    runtime.images={'tennis':[],'bmx-trees':[]};runtime.audio=deque();runtime.text=queue.Queue(32)
    runtime.events=deque(maxlen=4096);runtime.event_id=0;runtime.messages=deque(maxlen=100)
    runtime.utterance='';runtime.utterance_id=0;runtime.pending_bytes=bytearray()
    from urge import UrgeGate
    runtime.urge=UrgeGate()
    runtime.stats=dict(physicalSeconds=123.25,ticks=0,framesReceived=0,supersededFrames=0,
        emittedBytes=0,weightsUpdate=0,rejectedAudioSamples=0,audioSamplesReceived=0,focus=[])
    directory=ROOT/'artifacts/runtime-tests';directory.mkdir(parents=True,exist_ok=True)
    runtime.state_path=directory/(uuid.uuid4().hex+'.pt')
    return runtime


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_eager_vision_fallback_clears_previous_patch_map(self):
        graphs=GraphedOmniVision.__new__(GraphedOmniVision)
        graphs.last_patch_attention='old';graphs.original=lambda data:'native-fallback'
        self.assertEqual(graphs.encode({'pixel_values':[[],[]]}),'native-fallback')
        self.assertIsNone(graphs.last_patch_attention)

    def test_resume_rejects_unvalidated_internal_processing(self):
        runtime=fixture();runtime.model.enable_internal_writes();runtime.save_state()
        saved=torch.load(runtime.state_path,map_location='cpu',weights_only=False)
        with self.assertRaisesRegex(ValueError,'deployment checks'):fixture().restore_state(saved)
        runtime.state_path.unlink()

    def test_source_and_matrix_survive_concurrent_saves(self):
        runtime=fixture();runtime.messages.append(dict(source='user',text='A saved message',time=1.))
        expected=runtime.state.clone()
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(lambda _:runtime.save_state(),range(8)))
        saved=torch.load(runtime.state_path,map_location='cpu',weights_only=False)
        recovered=fixture();recovered.restore_state(saved)
        self.assertTrue(torch.equal(expected,recovered.state))
        self.assertEqual(recovered.source,'video');self.assertFalse(recovered.active)
        self.assertEqual(recovered.stats['physicalSeconds'],123.25)
        self.assertAlmostEqual(time.monotonic()-recovered.demo_at,24.,delta=1.)
        self.assertEqual(recovered.messages[0]['text'],'A saved message')
        self.assertFalse(runtime.state_path.with_suffix('.tmp').exists())
        runtime.state_path.unlink()

    def test_autosave_failure_is_reported_without_stopping_the_agent(self):
        runtime=fixture();runtime.active=True
        runtime.state_path=runtime.state_path.parent/'missing-directory'/'state.pt'
        runtime.save_quietly()
        self.assertIn('saveError',runtime.stats);self.assertTrue(runtime.active)
        self.assertNotEqual(runtime.stats.get('state'),'failed')
        self.assertEqual(runtime.events[-1]['data']['path'],'autosave')

    def test_weights_are_rewritten_only_when_they_change(self):
        runtime=fixture();written=[]
        original=runtime.durable_save
        runtime.durable_save=lambda payload,path:(written.append(path.name),original(payload,path))
        runtime.save_state();runtime.save_state()
        self.assertEqual(written,[runtime.weights_path().name,runtime.state_path.name,runtime.state_path.name])
        saved=torch.load(runtime.state_path,map_location='cpu',weights_only=False)
        self.assertNotIn('modelState',saved)
        self.assertLess(runtime.state_path.stat().st_size,1_000_000)
        recovered=fixture();recovered.state_path=runtime.state_path;recovered.restore_state(saved)
        self.assertTrue(torch.equal(runtime.state,recovered.state))
        runtime.stats['weightsUpdate']=1;runtime.save_state()
        self.assertEqual(written.count(runtime.weights_path().name),2)
        runtime.state_path.unlink();runtime.weights_path().unlink()

    def test_resume_and_recorded_source_respect_guards(self):
        runtime=fixture();runtime.ram_budget=0
        with self.assertRaisesRegex(ValueError,'budget'):runtime.set_active(True)
        self.assertFalse(runtime.active)
        runtime.ready=False
        with self.assertRaisesRegex(ValueError,'Test recording unavailable'):runtime.set_source('recorded')

    def test_newest_frame_with_concurrent_producers_and_consumer(self):
        runtime=fixture();stop=threading.Event()
        def consume():
            while not stop.is_set():runtime.latest_frame(time.monotonic());time.sleep(.0001)
        reader=threading.Thread(target=consume);reader.start()
        try:
            with ThreadPoolExecutor(max_workers=4) as workers:
                list(workers.map(lambda index:runtime.receive_frame(Image.new('RGB',(2,2),(index%255,0,0))),range(300)))
        finally:stop.set();reader.join(timeout=2)
        runtime.receive_frame(Image.new('RGB',(2,2),(0,255,0)))
        self.assertEqual(runtime.latest_frame(time.monotonic()).getpixel((0,0)),(0,255,0))
        self.assertEqual(runtime.stats['framesReceived'],301)
        self.assertLessEqual(runtime.frames.qsize(),2)

    def test_source_switch_clears_stale_picture_without_resetting_state(self):
        runtime=fixture();expected=runtime.state.clone()
        runtime.receive_frame(Image.new('RGB',(2,2)));runtime.latest_jpeg=b'old picture'
        runtime.set_source('quiet')
        self.assertIsNone(runtime.latest_frame(time.monotonic()));self.assertIsNone(runtime.latest_jpeg)
        self.assertTrue(torch.equal(expected,runtime.state))
        with self.assertRaisesRegex(ValueError,'Test recording unavailable'):runtime.set_source('recorded')

    def test_unicode_and_transport_framing_preserve_actual_feedback(self):
        runtime=fixture();codes=[]
        class Feedback:
            def feedback(self,state,code):codes.append(code);return state
        runtime.model=Feedback()
        for code in 'λ'.encode('utf8'):runtime.emit_code_locked(code)
        self.assertEqual(runtime.stats['emittedBytes'],2);self.assertEqual(runtime.utterance,'λ')
        runtime.emit_code_locked(BOUNDARY)
        self.assertEqual(runtime.messages[-1]['text'],'λ')
        for _ in range(MAX_MESSAGE_CHARACTERS+3):runtime.emit_code_locked(ord('x'))
        self.assertEqual(codes.count(BOUNDARY),1)
        self.assertEqual(runtime.utterance,'xxx')
        self.assertEqual(len(runtime.messages[-1]['text']),MAX_MESSAGE_CHARACTERS)
        self.assertTrue(any(event['data'].get('framing')=='transport-limit' for event in runtime.events))
        self.assertLessEqual(max(len(event['data'].get('text','')) for event in runtime.events),MAX_MESSAGE_CHARACTERS)

    def test_invalid_utf8_is_bounded_and_sample_omits_chat_history(self):
        runtime=fixture()
        runtime.emit_code_locked(255)
        self.assertEqual(runtime.stats['emittedBytes'],1);self.assertEqual(runtime.utterance,'�')
        runtime.messages.append(dict(source='user',text='Private chat',time=1.))
        self.assertNotIn('messages',runtime.snapshot(include_messages=False))
        self.assertEqual(runtime.snapshot()['messages'][-1]['text'],'Private chat')

    def test_uniform_or_invalid_focus_does_not_claim_localization(self):
        for heat in [np.zeros((4,4)),np.ones((4,4)),np.array([[np.nan]]),np.empty((0,0))]:
            self.assertEqual(rectangles(heat),[])
        heat=np.zeros((4,4));heat[1,2]=1.
        self.assertEqual(rectangles(heat)[0],dict(x=.5,y=.25,width=.25,height=.25))


if __name__=='__main__':unittest.main()
