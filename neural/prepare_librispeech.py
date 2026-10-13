"""Fetch an identity-checked public speech archive and a bounded speaker split.

Transcripts are offline supervision/evaluation metadata. They must never be
given to the sensory frontend during a raw-speech test. No recognizer is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
URL='https://openslr.trmal.net/resources/12/dev-clean.tar.gz'
EXPECTED_MD5='42e2234ba48799c1f50f24a7926300a1'


def hashes(path):
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda:source.read(1024*1024),b''):md5.update(data);sha.update(data)
    return md5.hexdigest(),sha.hexdigest()


def prepare(speakers=12,per_speaker=6):
    folder=ROOT/'artifacts/librispeech-dev-clean';folder.mkdir(exist_ok=True)
    receipt=folder/'manifest.json'
    report=dict(state='preparing',sourceUrl=URL,license='CC BY 4.0',
        attribution='Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur; LibriSpeech, ICASSP 2015',
        resourcePage='https://www.openslr.org/12/',checksumSource='https://openslr.trmal.net/resources/12/md5sum.txt',
        expectedArchiveMd5=EXPECTED_MD5,
        splitProtocol='First requested sorted speakers; every fourth speaker held out. No speaker crosses split.',
        limits=['Read audiobook speech; not spontaneous conversation.',
            'Subset taken from official development set, not the official training/test protocol.',
            'Transcripts remain offline labels and do not enter raw sensory input.'])
    def save():
        temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2),encoding='utf8');temporary.replace(receipt)
    save()
    try:
        archive=folder/'dev-clean.tar.gz'
        if not archive.exists():
            temporary=archive.with_suffix('.download')
            print('Downloading identity-checked LibriSpeech dev-clean archive.',flush=True)
            with urlopen(URL,timeout=60) as response,temporary.open('wb') as target:
                shutil.copyfileobj(response,target,1024*1024)
            temporary.replace(archive)
        md5,sha=hashes(archive)
        if md5!=EXPECTED_MD5:raise ValueError('LibriSpeech archive checksum mismatch; retain file for diagnosis.')
        report.update(archiveSha256=sha,archiveBytes=archive.stat().st_size,state='extracting');save()
        with tarfile.open(archive,'r:gz') as source:
            members={member.name:member for member in source if member.isfile()}
            audio=[name for name in members if name.endswith('.flac')]
            speaker_ids=sorted({name.split('/')[-3] for name in audio},key=int)[:speakers]
            examples=[]
            for index,speaker in enumerate(speaker_ids):
                selected=sorted(name for name in audio if name.split('/')[-3]==speaker)[:per_speaker]
                for name in selected:
                    member=members[name];identifier=Path(name).stem
                    # Only selected regular files are copied; archive names do
                    # not control filesystem paths or create links/directories.
                    target=folder/(identifier+'.flac')
                    if not target.resolve().is_relative_to(folder.resolve()):raise ValueError('Invalid archive identifier.')
                    stream=source.extractfile(member)
                    if stream is None:raise ValueError('Missing audio archive entry.')
                    with stream,target.open('wb') as output:shutil.copyfileobj(stream,output,1024*1024)
                    transcript_name=str(Path(name).parent).replace('\\','/')+'/'+ '-'.join(identifier.split('-')[:2])+'.trans.txt'
                    transcript=source.extractfile(members[transcript_name])
                    if transcript is None:raise ValueError('Missing transcript archive entry.')
                    with transcript:text=transcript.read().decode('utf8')
                    label=next(line.split(' ',1)[1] for line in text.splitlines() if line.startswith(identifier+' '))
                    examples.append(dict(id=identifier,speaker=speaker,split='held' if index%4==3 else 'train',
                        file=str(target.relative_to(ROOT)),sha256=hashes(target)[1],transcript=label))
                print(f'Prepared speaker {speaker}: {len(selected)} recordings.',flush=True)
        report.update(state='complete',speakers=speaker_ids,examples=examples,
            trainRecordings=sum(row['split']=='train' for row in examples),heldRecordings=sum(row['split']=='held' for row in examples))
        save();print(receipt,flush=True)
    except Exception as error:report.update(state='failed',error=str(error));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--speakers',type=int,default=12)
    parser.add_argument('--per-speaker',type=int,default=6);args=parser.parse_args()
    if not 4<=args.speakers<=40:parser.error('--speakers must be from 4 to 40')
    if not 1<=args.per_speaker<=20:parser.error('--per-speaker must be from 1 to 20')
    prepare(args.speakers,args.per_speaker)
