"""Read-only local dependency/asset checks; never load private live state."""
import argparse
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def collect(hash_weights=False):
    problems=[];packages=[];assets=[]
    lock_path=ROOT/'config/sensory-assets.json'
    lock=json.loads(lock_path.read_text()) if lock_path.exists() else {}
    expected_assets={row['file']:row for row in lock.get('files',[])}
    for line in (ROOT/'config/runtime-requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,expected=line.split('==')
        try:actual=version(name)
        except PackageNotFoundError:actual=None
        packages.append(dict(name=name,expected=expected,actual=actual))
        if actual!=expected:problems.append(f'{name}: expected {expected}, found {actual}')
    paths=[ROOT/'artifacts/liquid-core-training.json',ROOT/'.runtime/minicpm-omni/config.json',
           ROOT/'.runtime/minicpm-omni/model.safetensors.index.json']
    model=ROOT/'.runtime/minicpm-omni'
    if paths[-1].exists():
        weights=json.loads(paths[-1].read_text())['weight_map']
        prefixes=('vpm.','resampler.','apm.','audio_projection_layer.','llm.model.embed_tokens.')
        paths.extend(model/shard for shard in sorted({shard for key,shard in weights.items() if key.startswith(prefixes)}))
        paths.extend(sorted(model.glob('*.py')))
        paths.extend(sorted(model.glob('*token*')))
        paths.extend(sorted(model.glob('*processor*json')))
    if paths[0].exists():
        selected=json.loads(paths[0].read_text())
        checkpoint=(ROOT/selected['directory']/'latest.pt').resolve()
        if not checkpoint.is_relative_to((ROOT/'artifacts').resolve()):problems.append('Checkpoint outside artifacts')
        else:paths.append(checkpoint)
    for path in dict.fromkeys(paths):
        exists=path.is_file();record=dict(path=str(path.relative_to(ROOT)),exists=exists)
        if exists:
            record['bytes']=path.stat().st_size
            if path.suffix not in ['.safetensors','.pt'] or hash_weights:record['sha256']=digest(path)
            if path.parent==model and path.name in expected_assets:
                expected=expected_assets[path.name]
                if record['bytes']!=expected['bytes'] or ('sha256' in record and record['sha256']!=expected['sha256']):
                    problems.append(f'Sensory identity mismatch: {path.name}')
        else:problems.append(f'Missing required asset: {record["path"]}')
        assets.append(record)
    return dict(createdAt=datetime.now(timezone.utc).isoformat(),python=sys.version.split()[0],
        packages=packages,assets=assets,problems=problems,ok=not problems,
        upstreamModel='https://huggingface.co/openbmb/MiniCPM-o-4_5',
        upstreamRevision=lock.get('revision'),limitation='Small sensory assets are checked against the pinned identity; use --hash-weights for shard checks. Cold-install reproducibility remains unverified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path)
    parser.add_argument('--hash-weights',action='store_true');args=parser.parse_args()
    result=collect(args.hash_weights)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        args.manifest.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(ok=result['ok'],problems=result['problems'],assets=len(result['assets']))))
    sys.exit(0 if result['ok'] else 1)
