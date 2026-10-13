"""Actual copy-trainer updates survive interruption; failure receipts terminate."""
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'neural'))
import train_liquid_copy as trainer
from liquid_core import LiquidCore


class CopyTrainingTests(unittest.TestCase):
    def test_actual_fit_continuation_and_terminal_failure(self):
        torch.set_num_threads(2)
        directory=ROOT/'artifacts/runtime-tests'/uuid.uuid4().hex
        (directory/'artifacts').mkdir(parents=True)
        (directory/'artifacts/workspace-raw-corpus.json').write_text('{}')
        core=LiquidCore();source=directory/'source.pt'
        torch.save(dict(architecture=core.architecture,state_dict=core.state_dict()),source)
        sensory=torch.randn(2,4096);query=torch.randn(2,4096)
        folders=[]
        def fit(checkpoint,updates):
            folder=directory/f'fit-{len(folders)}';folder.mkdir();folders.append(folder)
            args=SimpleNamespace(checkpoint=checkpoint,device='cpu',visual_inputs=False,rehearsal_rate=0.,updates=updates)
            torch.manual_seed(1917);random.seed(1917)
            trainer.fit(args,{},[],folder,folder/'training.json',{})
            return folder/'latest.pt'
        original_words=trainer.words
        try:
            with patch.object(trainer,'ROOT',directory),patch.object(trainer,'backgrounds',return_value={'quiet':sensory}),patch.object(trainer,'literal_rows',return_value=[query]),patch.object(trainer,'words',side_effect=lambda count,*a,**k:original_words(32 if count==2048 else 4,*a,**k)),patch.object(trainer,'assess',return_value={'syntheticEvaluationSkipped':True}):
                uninterrupted=fit(source,2);first=fit(source,1);resumed=fit(first,1)
                expected=torch.load(uninterrupted,weights_only=False)
                actual=torch.load(resumed,weights_only=False)
                self.assertTrue(json.loads((resumed.parent/'training.json').read_text())['optimizerRestored'])
                self.assertTrue(all(torch.equal(value,actual['state_dict'][name]) for name,value in expected['state_dict'].items()))
                with patch.object(trainer,'fit',side_effect=RuntimeError('controlled failure')):
                    with self.assertRaisesRegex(RuntimeError,'controlled failure'):trainer.train(SimpleNamespace(),{},[])
                failed=list((directory/'artifacts').glob('liquid-copy-*/training.json'))
                self.assertEqual(len(failed),1)
                self.assertEqual(json.loads(failed[0].read_text())['state'],'failed')
        finally:
            # Delete only enumerated test files; never a recursive shell command.
            if not directory.resolve().is_relative_to((ROOT/'artifacts/runtime-tests').resolve()):
                raise RuntimeError('Test cleanup target escaped its workspace.')
            for path in directory.rglob('*'):
                if path.is_file():path.unlink()
            for path in sorted([p for p in directory.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):path.rmdir()
            directory.rmdir()


if __name__=='__main__':unittest.main()
