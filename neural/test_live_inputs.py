"""Replay actual waveform/video through an isolated instance of the same core."""
import argparse
import json
from pathlib import Path
import time

from build_workspace_raw_corpus import read_audio
from liquid_runtime import LiquidRuntime

ROOT = Path(__file__).resolve().parents[1]


def feed(runtime, wave, report, save):
    started = time.monotonic()
    for offset in range(0, len(wave), 1600):
        chunk = wave[offset:offset+1600].astype('<f4')
        runtime.receive_audio(chunk)
        due = started+(offset+len(chunk))/16000
        while time.monotonic() < due:
            time.sleep(max(0., min(.01, due-time.monotonic())))
        if offset % 16000 == 0:
            report['observations'].append(runtime.snapshot()); save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--audio', type=Path, default=ROOT/'artifacts/workspace-raw-speech/apple.wav')
    parser.add_argument('--spoken-query', type=Path)
    parser.add_argument('--question', default='Which fruit did I say is my favourite?')
    args = parser.parse_args()
    identifier = time.time_ns()
    runtime = LiquidRuntime(ROOT, state_path=ROOT/'artifacts'/f'liquid-isolated-test-{identifier}.pt',
                            checkpoint_override=args.checkpoint)
    path = ROOT/'artifacts'/f'liquid-live-input-test-{identifier}.json'
    report = {'state': 'loading', 'sentTranscript': False, 'observations': [],
              'candidateCheckpoint': str(args.checkpoint) if args.checkpoint else None}
    def save():
        path.write_text(json.dumps(report, indent=2), encoding='utf8')
    try:
        save(); deadline = time.monotonic()+90
        while not runtime.ready and time.monotonic() < deadline:
            if runtime.stats.get('error'):
                raise RuntimeError(runtime.stats['error'])
            time.sleep(.1)
        if not runtime.ready:
            raise RuntimeError('Numerical model did not become ready.')
        before = runtime.snapshot()
        runtime.set_source('recorded'); runtime.set_active(True)
        # PCM samples, not a fruit label or transcript in a request.
        wave = read_audio(args.audio)
        report.update(state='running', samples=len(wave), before=before,
                      spokenQuery=bool(args.spoken_query))
        wall_started = time.time(); feed(runtime, wave, report, save)
        time.sleep(.3); question_at = time.time()
        if args.spoken_query:
            query = read_audio(args.spoken_query)
            report['querySamples'] = len(query)
            feed(runtime, query, report, save)
        else:
            runtime.message(args.question)
        deadline = time.monotonic()+5; responses = []
        while time.monotonic() < deadline:
            actual = runtime.snapshot()
            responses = [row for row in actual['messages'] if row['source']=='core' and row['time']>=question_at]
            if responses:
                break
            if actual.get('error'):
                raise RuntimeError(actual['error'])
            time.sleep(.1)
        report.update(state='complete', after=actual, postQuestionOutputs=responses,
            streamOutputs=[row for row in actual['messages'] if row['source']=='core' and row['time']>=wall_started],
            ticksAdvanced=actual['ticks']-before['ticks'],
            observationsAdvanced=actual['observations']-before['observations'],
            audioSamplesAdvanced=actual['audioSamplesReceived']-before['audioSamplesReceived'],
            prerecordedVideo=True, isolatedModelInstance=True, extraListeningPorts=0,
            userCameraAndLiveStateUnchanged=True)
        print(json.dumps({key: report[key] for key in ['state','samples','postQuestionOutputs',
            'ticksAdvanced','observationsAdvanced','audioSamplesAdvanced']}), flush=True)
    except Exception as error:
        report.update(state='failed', error=str(error)); raise
    finally:
        runtime.close(); save()
        print(json.dumps({'receipt': str(path)}), flush=True)


if __name__ == '__main__':
    from research_resources import research_slot
    with research_slot('raw replay validation'):
        main()
