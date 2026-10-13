"""End-to-end optional-demo startup with real native weights, no UI port/media."""
import json
from pathlib import Path
import shutil
import time
import torch
from liquid_runtime import LiquidRuntime
from research_resources import research_slot
ROOT=Path(__file__).resolve().parents[1]


def main():
    with research_slot('runtime without optional demonstration assets',reserve_bytes=3_000_000_000):
        directory=ROOT/'artifacts'/f'runtime-startup-{time.time_ns()}'
        assets=directory/'artifacts';assets.mkdir(parents=True)
        selection=json.loads((ROOT/'artifacts/liquid-core-training.json').read_text())
        checkpoint=assets/'selected.pt';shutil.copy2(ROOT/selection['directory']/'latest.pt',checkpoint)
        runtime=LiquidRuntime(directory,state_path=assets/'state.pt',checkpoint_override=checkpoint)
        result=dict(state='starting',usesRealSensoryWeights=True,media='Silent waveform only; no camera/microphone',
                    extraUiPorts=0,missingOptionalImages=True,missingOptionalSpeech=True)
        try:
            deadline=time.monotonic()+60
            while not runtime.ready and runtime.stats.get('state')!='failed' and time.monotonic()<deadline:
                time.sleep(.1)
            if not runtime.ready:raise RuntimeError(runtime.stats.get('error','Startup did not become ready.'))
            runtime.set_active(False);runtime.set_source('camera')
            try:runtime.set_source('recorded')
            except ValueError:result['missingRecordingRejected']=True
            else:raise AssertionError('Missing test recording unexpectedly accepted.')
            result.update(state='complete',readyWithoutDemos=runtime.ready,
                source=runtime.source,numericalStateFinite=bool(torch.isfinite(runtime.state).all()),
                speechSamples=len(runtime.speech),imageCounts={name:len(paths) for name,paths in runtime.images.items()},
                systemWithinBudget=runtime.snapshot()['memory']['withinSystemBudget'])
            assert result['systemWithinBudget']
        except Exception as error:
            result.update(state='failed',error=str(error));raise
        finally:
            runtime.close();runtime.worker.join(timeout=5)
            result['workerStopped']=not runtime.worker.is_alive()
            path=directory/'validation.json';path.write_text(json.dumps(result,indent=2),encoding='utf8')
            print(json.dumps(result),flush=True);print(path,flush=True)


if __name__=='__main__':main()
