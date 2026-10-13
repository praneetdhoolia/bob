"""Isolated active run of the real runtime: urge, stimulus, timing, save/restore.

Uses the selected checkpoint and real frozen sensory encoders with the built-in
recorded scenes and spoken sample. It opens no port, touches no camera or
microphone, and writes its own state file, so the user's live state is untouched.
"""
import json
from pathlib import Path
import time

import psutil
import torch

from liquid_runtime import LiquidRuntime
from research_resources import research_slot

ROOT = Path(__file__).resolve().parents[1]


def wait_ready(runtime, seconds=120):
    deadline = time.monotonic()+seconds
    while not runtime.ready and runtime.stats.get('state') != 'failed' and time.monotonic() < deadline:
        time.sleep(.1)
    if not runtime.ready:
        raise RuntimeError(runtime.stats.get('error', 'Startup did not become ready.'))


def main(listen_seconds=14., answer_seconds=8.):
    with research_slot('isolated live runtime probe', reserve_bytes=4_000_000_000):
        folder = ROOT/'artifacts'/f'runtime-live-{time.time_ns()}'; folder.mkdir()
        selection = json.loads((ROOT/'artifacts/liquid-core-training.json').read_text(encoding='utf8'))
        checkpoint = ROOT/selection['directory']/'latest.pt'
        result = dict(state='starting', checkpoint=str(checkpoint.relative_to(ROOT)).replace('\\', '/'),
                      media='Built-in recorded scenes and spoken sample; no camera, microphone or port')
        runtime = LiquidRuntime(ROOT, state_path=folder/'state.pt', checkpoint_override=checkpoint)
        try:
            began = time.monotonic(); wait_ready(runtime); result['startupSeconds'] = time.monotonic()-began
            runtime.set_source('recorded')
            start_ticks, start_observations = runtime.stats['ticks'], runtime.stats['observations']
            time.sleep(listen_seconds)
            runtime.message('Which fruit did I say is my favourite?')
            time.sleep(answer_seconds)
            runtime.set_active(False)
            with runtime.lock:
                events = list(runtime.events); stats = dict(runtime.stats)
            samples = [event['data'] for event in events if event['type'] == 'sample']
            latencies = [row['latestObservationMs'] for row in samples if 'latestObservationMs' in row]
            stimuli = [event['data'] for event in events if event['type'] == 'stimulus']
            outputs = [event['data']['text'] for event in events if event['type'] == 'output' and event['data'].get('ended')]
            elapsed = listen_seconds+answer_seconds
            result.update(
                numericalTicksPerSecond=(stats['ticks']-start_ticks)/elapsed,
                observationsPerSecond=(stats['observations']-start_observations)/elapsed,
                observationMsMean=sum(latencies)/len(latencies) if latencies else None,
                observationMsMax=max(latencies) if latencies else None,
                deadlineMisses=stats['deadlineMisses'], observations=stats['observations'],
                stimulusReleases=len(stimuli), urgeReleasesStat=stats.get('urgeReleases'),
                firstStimulus={key: stimuli[0][key] for key in ['urge', 'readiness', 'surprise', 'dimensions', 'rms', 'firstReading']} if stimuli else None,
                finishedUtterances=outputs, errors=[event['data'] for event in events if event['type'] in ['error', 'input-error']],
                agentRssBytes=runtime.process.memory_info().rss, systemUsedBytes=psutil.virtual_memory().used,
                gpuAllocatedBytes=stats.get('gpuAllocatedBytes'), pretrainedLanguageForwardCalls=stats['pretrainedLanguageForwardCalls'])
            runtime.save_state()
            saved_state = runtime.state.clone(); saved_seconds = runtime.stats['physicalSeconds']
            result['stateFileBytes'] = (folder/'state.pt').stat().st_size
            result['weightsFileBytes'] = runtime.weights_path().stat().st_size
        finally:
            runtime.close(); runtime.worker.join(timeout=5)
        restored = LiquidRuntime(ROOT, state_path=folder/'state.pt', checkpoint_override=checkpoint)
        try:
            wait_ready(restored)
            result.update(restoredPaused=not restored.active,
                          restoredMaximumDifference=float((restored.state-saved_state).abs().max()),
                          restoredPhysicalSeconds=restored.stats['physicalSeconds'], savedPhysicalSeconds=saved_seconds)
        finally:
            restored.running = False; restored.worker.join(timeout=5)
        result['state'] = 'complete'
        result['limits'] = ['One short run on recorded fixtures; not a soak test or hardware-capture test.',
                            'Utterance content is the narrow trained translator; urge releases are mechanism readings, not felt desire.']
        (folder/'validation.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        print(json.dumps(result, indent=1)); print(folder/'validation.json')


if __name__ == '__main__':
    main()
