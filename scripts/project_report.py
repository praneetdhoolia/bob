"""Shared, project-agnostic collection/rendering for both agent frameworks.

Standard library only. Collection reads source metadata and optional filtered
loopback status; rendering writes a new standalone report without overwriting.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from urllib.request import urlopen

EXCLUDED = {'.git','.runtime','.venv','venv','node_modules','__pycache__',
            '.local','.pytest_cache','.playwright-mcp','artifacts','data','logs','dist','build'}
TEXT_SUFFIXES = {'.py','.js','.mjs','.cjs','.ts','.tsx','.jsx','.md','.txt','.json',
                 '.yaml','.yml','.toml','.css','.html','.ps1','.sh','.cfg','.ini','.xml'}
MAX_TEXT_BYTES = 2_000_000


def git(root, *arguments):
    try:
        result = subprocess.run(['git','-C',str(root),*arguments],capture_output=True,
                                encoding='utf8',errors='replace',timeout=20)
        return result.stdout if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        return None


def source_paths(root):
    listed = git(root,'ls-files','-z','--cached','--others','--exclude-standard')
    if listed is not None:
        candidates = sorted(set(filter(None,listed.split('\0'))))
        for name in candidates:
            path = Path(name)
            if not any(part in EXCLUDED for part in path.parts):
                yield path
    else:
        for directory,children,files in os.walk(root,followlinks=False):
            children[:] = sorted(name for name in children if name not in EXCLUDED
                                 and not (Path(directory)/name).is_symlink()
                                 and not getattr(Path(directory)/name,'is_junction',lambda:False)())
            for name in sorted(files):
                yield (Path(directory)/name).relative_to(root)


def collect(root, status_url=None):
    rows=[];gaps=[]
    for relative in source_paths(root):
        path=root/relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        try:
            size=path.stat().st_size
            if not path.is_file():
                continue
            row=dict(path=relative.as_posix(),bytes=size,lines=None)
            if size<=MAX_TEXT_BYTES and (path.suffix.lower() in TEXT_SUFFIXES or path.name=='.gitignore'):
                # Count only: file contents and secrets never enter the output.
                row['lines']=len(path.read_bytes().splitlines())
            rows.append(row)
        except OSError as error:
            gaps.append(dict(path=relative.as_posix(),reason=type(error).__name__))
    head=git(root,'rev-parse','--verify','HEAD')
    branch=git(root,'symbolic-ref','--short','HEAD')
    status=git(root,'status','--short')
    history=git(root,'log','-100','--format=%H%x09%aI%x09%s') if head else None
    if not head:
        gaps.append(dict(area='history',reason='No readable HEAD; commit/PR history is not established.'))
    if status is None:
        gaps.append(dict(area='git',reason='Git worktree status unavailable.'))
    result=dict(generatedAt=datetime.now(timezone.utc).isoformat(),root=str(root),
        scope='Source metadata, including Git-untracked files; not a code review or full artifact inventory.',
        exclusions=sorted(EXCLUDED),textLineLimitBytes=MAX_TEXT_BYTES,
        repository=dict(head=head.strip() if head else None,branch=branch.strip() if branch else None,
                        status=status.splitlines() if status is not None else None,
                        recentCommits=history.splitlines() if history else [],historyLimit=100),
        inventory=dict(files=len(rows),bytes=sum(row['bytes'] for row in rows),
                       countedLines=sum(row['lines'] or 0 for row in rows),
                       byExtension=dict(sorted(Counter(Path(row['path']).suffix or '(none)' for row in rows).items())),
                       rows=rows),gaps=gaps)
    if status_url:
        address=urlparse(status_url)
        if address.scheme not in ['http','https'] or address.hostname not in ['localhost','127.0.0.1','::1']:
            raise ValueError('Status collection requires an explicitly supplied loopback HTTP(S) URL.')
        try:
            with urlopen(status_url,timeout=5) as response:
                raw=json.loads(response.read(1_000_001))
            permitted=['state','ready','active','source','physicalSeconds','ticks','observations',
                       'deadlineMisses','pretrainedLanguageForwardCalls','weightsUpdate','latestObservationMs']
            result['liveStatus']={key:raw[key] for key in permitted if key in raw}
            result['liveStatus']['memory']={key:raw.get('memory',{}).get(key) for key in
                ['agentRssBytes','systemUsedBytes','systemTotalBytes','systemAvailableBytes','withinSystemBudget']}
            result['liveStatus']['compute']={key:raw.get('compute',{}).get(key) for key in
                ['agentCpuPercent','systemCpuPercent','gpuAllocatedBytes','gpuReservedBytes','gpuTotalBytes']}
        except (OSError,ValueError) as error:
            gaps.append(dict(area='liveStatus',reason=type(error).__name__))
    return result


def escaped(value):
    return html.escape(str(value),quote=True)


def cell(value):
    if isinstance(value,dict) and 'url' in value:
        url=str(value['url']);label=value.get('label',url)
        if urlparse(url).scheme in ['http','https']:
            return f'<a href="{escaped(url)}">{escaped(label)}</a>'
    if isinstance(value,(dict,list)):
        return escaped(json.dumps(value,ensure_ascii=False))
    return escaped('Unverified' if value is None else value)


def table(data):
    columns=data['columns'];rows=data['rows']
    if not isinstance(columns,list) or not all(isinstance(column,str) for column in columns):
        raise ValueError('Table columns must be a list of strings.')
    if not isinstance(rows,list) or any(not isinstance(row,(dict,list)) for row in rows):
        raise ValueError('Table rows must be dictionaries or lists.')
    values=[[row.get(column) for column in columns] if isinstance(row,dict) else row for row in rows]
    if any(len(row)!=len(columns) for row in values):
        raise ValueError('Table row width differs from its columns.')
    numeric=[any(isinstance(row[index],(int,float)) and not isinstance(row[index],bool) for row in values)
             for index in range(len(columns))]
    attributes=[' class="number"' if number else '' for number in numeric]
    headers=''.join(f'<th scope="col"{attributes[index]}>{escaped(column)}</th>' for index,column in enumerate(columns))
    body=''.join('<tr>'+''.join(f'<td{attributes[index]}>{cell(value)}</td>' for index,value in enumerate(row))+'</tr>' for row in values)
    return f'<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable report table"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>'


def render(data):
    title=data.get('title');sections=data.get('sections')
    if not isinstance(title,str) or not title.strip() or not isinstance(sections,list) or not sections:
        raise ValueError('A report needs a title and nonempty sections.')
    parts=[]
    for section in sections:
        if not isinstance(section,dict) or not isinstance(section.get('heading'),str):
            raise ValueError('Every report section needs a heading.')
        if not isinstance(section.get('paragraphs',[]),list) or not isinstance(section.get('bullets',[]),list):
            raise ValueError('Section paragraphs and bullets must be lists.')
        body=''.join(f'<p>{cell(paragraph)}</p>' for paragraph in section.get('paragraphs',[]))
        if section.get('bullets'):
            body+='<ul>'+''.join(f'<li>{cell(item)}</li>' for item in section['bullets'])+'</ul>'
        if 'table' in section:
            body+=table(section['table'])
        parts.append(f'<section><h2>{escaped(section["heading"])}</h2>{body}</section>')
    if not isinstance(data.get('sources',[]),list):
        raise ValueError('Report sources must be a list.')
    if data.get('sources'):
        parts.append('<section><h2>Sources</h2><ul>'+''.join(f'<li>{cell(source)}</li>' for source in data['sources'])+'</ul></section>')
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    styles='''
    :root{color-scheme:light dark;--back:#e3e3e3;--panel:#f2f2f2dd;--ink:#202020;--muted:#505050;--line:#7b7b7b44;--link:#164b8a}
    @media(prefers-color-scheme:dark){:root{--back:#1d1d1d;--panel:#303030dd;--ink:#efefef;--muted:#bcbcbc;--line:#b0b0b038;--link:#9bc9ff}}
    *{box-sizing:border-box}html{font:16px/1.6 system-ui,sans-serif;background:var(--back);color:var(--ink)}
    body{max-width:1200px;margin:0 auto;padding:clamp(18px,4vw,48px)}header,section{padding:24px;border:1px solid var(--line);border-radius:18px;background:var(--panel);margin-bottom:18px}
    h1{font-size:clamp(24px,3vw,38px);line-height:1.2}h2{font-size:22px}h1,h2{margin:0 0 14px}p,li,td{overflow-wrap:anywhere}p:last-child{margin-bottom:0}
    .stamp{color:var(--muted);font-size:14px}a{color:var(--link);text-underline-offset:3px}:focus-visible{outline:2px solid var(--link);outline-offset:3px;border-radius:6px}
    .table-scroll{overflow-x:auto;scrollbar-width:thin;scrollbar-color:var(--muted) transparent}table{border-collapse:collapse;width:100%;font-size:14px;min-width:560px}
    th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th+th,td+td{border-left:1px solid var(--line)}th{font-weight:600}
    .number{text-align:right;font-variant-numeric:tabular-nums}ul{padding-left:22px}@media print{html{background:white;color:black}body{max-width:none;padding:0}section,header{background:white;border-color:#aaa;break-inside:avoid}.table-scroll{overflow:visible}table{min-width:0}}
    '''
    return ('<!doctype html><html lang="en-AU"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escaped(title)}</title><style>{styles}</style></head><body><main>'
        f'<header><h1>{escaped(title)}</h1><p class="stamp">{escaped(data.get("generatedAt","Timestamp unverified"))}</p>'
        f'<p>{cell(data.get("summary","Assessment summary not supplied."))}</p></header>'
        +''.join(parts)+f'</main><script type="application/json" id="report-data">{payload}</script></body></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    inventory=commands.add_parser('collect');inventory.add_argument('output',type=Path)
    inventory.add_argument('--root',type=Path,default=Path.cwd());inventory.add_argument('--status-url')
    report=commands.add_parser('render');report.add_argument('input',type=Path);report.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.command=='collect':
        data=collect(args.root.resolve(),args.status_url)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    else:
        data=json.loads(args.input.read_text(encoding='utf8'))
        document=render(data)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf8') as output:
            output.write(document)
    print(args.output.resolve())


if __name__=='__main__':
    main()
