"""Copy the small receipts cited by the documentation into version control.

`artifacts/` is ignored because it holds checkpoints, corpora and media. The
documentation still cites measured receipts there. This script mirrors every
cited JSON receipt under `docs/evidence/receipts/` and records the SHA-256 of
cited binaries (checkpoints) without copying them. `--check` verifies that the
tracked copies still match the manifest and, when present, the local originals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/evidence'
RECEIPTS = EVIDENCE/'receipts'
MANIFEST = EVIDENCE/'manifest.json'
CITING = ['README.md', 'USAGE.md', 'docs/*.md', 'docs/reports/recommendations.json',
          'docs/reports/reference/*.json']
CITATION = re.compile(r'artifacts/[A-Za-z0-9_./-]+')
MAX_COPY_BYTES = 1_000_000
# Private, constantly rewritten live state: never hashed or copied.
EXCLUDED = {'artifacts/liquid-live-state.pt'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def cited_paths(root=ROOT):
    found = set()
    for pattern in CITING:
        for document in sorted(root.glob(pattern)):
            for match in CITATION.findall(document.read_text(encoding='utf8')):
                found.add(match.rstrip('.,)'))
    return sorted(found - EXCLUDED)


def collect(root=ROOT):
    entries, missing, skipped = [], [], []
    for relative in cited_paths(root):
        path = root/relative
        if not path.is_file():
            (missing if not path.exists() else skipped).append(relative)
            continue
        size = path.stat().st_size
        row = dict(original=relative, bytes=size, sha256=digest(path))
        if path.suffix == '.json' and size <= MAX_COPY_BYTES:
            row['copy'] = str((RECEIPTS/relative.removeprefix('artifacts/')).relative_to(root)).replace('\\', '/')
        else:
            row['copy'] = None
            row['note'] = 'Hash only: binary or larger than the tracked receipt limit.'
        entries.append(row)
    return entries, missing, skipped


def write(root=ROOT):
    entries, missing, skipped = collect(root)
    for row in entries:
        if row['copy']:
            target = root/row['copy']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/row['original'], target)
    manifest = dict(
        purpose='Tracked copies and hashes of receipts cited by the documentation.',
        limits='Copies preserve measured outputs; they do not include checkpoints, corpora or media needed to rerun them.',
        entries=entries,
        citedDirectoriesOrPrefixes=skipped + missing)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    return manifest


def check(root=ROOT):
    manifest = json.loads((root/'docs/evidence/manifest.json').read_text(encoding='utf8'))
    problems, warnings = [], []
    recorded = {row['original'] for row in manifest['entries']} | set(manifest['citedDirectoriesOrPrefixes'])
    for relative in cited_paths(root):
        if relative not in recorded and (root/relative).is_file():
            problems.append(f'cited but not snapshotted (run scripts/snapshot_evidence.py): {relative}')
    for row in manifest['entries']:
        if row['copy']:
            copy = root/row['copy']
            if not copy.is_file() or digest(copy) != row['sha256']:
                problems.append(f"tracked copy differs: {row['copy']}")
        original = root/row['original']
        if original.is_file() and digest(original) != row['sha256']:
            # Some preflight receipts are regenerated locally; report, do not fail.
            warnings.append(f"local original changed since snapshot: {row['original']}")
    return problems, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='verify instead of writing')
    args = parser.parse_args()
    if args.check:
        problems, warnings = check()
        for line in warnings + problems:
            print(line)
        if not problems:
            print('evidence manifest verified')
        sys.exit(1 if problems else 0)
    manifest = write()
    copied = sum(1 for row in manifest['entries'] if row['copy'])
    print(f"{copied} receipts copied, {len(manifest['entries'])-copied} hashed only, "
          f"{len(manifest['citedDirectoriesOrPrefixes'])} directory/prefix citations")


if __name__ == '__main__':
    main()
