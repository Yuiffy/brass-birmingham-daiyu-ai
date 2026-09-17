"""Merge complete replay datasets without changing game-level holdout splits."""
import argparse, shutil
from pathlib import Path
from model import read_json, write_json


def merge(sources, out):
    sources=[Path(s) for s in sources];out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty replay output')
    metas=[read_json(s/'schema.json') for s in sources]
    games=[read_json(s/'games.json') for s in sources]
    seen=set()
    for source,meta,records in zip(sources,metas,games):
        if any(meta[k]!=metas[0][k] for k in ['fields','stateDim','rowWidth']):raise ValueError('Schema mismatch')
        if (source/'states.f32').stat().st_size!=meta['rows']*meta['rowWidth']*4:raise ValueError('Incomplete replay')
        cursor=0
        for g in records:
            if g['seed'] in seen:raise ValueError('Duplicate game seed')
            if g['start']!=cursor or g['end']<=cursor:raise ValueError('Invalid row ranges')
            cursor=g['end'];seen.add(g['seed'])
        if cursor!=meta['rows']:raise ValueError('Unclaimed replay rows')
    out.mkdir(parents=True,exist_ok=True);rows=0;combined=[]
    with (out/'states.f32').open('xb') as target:
        for source,meta,records in zip(sources,metas,games):
            with (source/'states.f32').open('rb') as stream:shutil.copyfileobj(stream,target)
            combined.extend(dict(g,start=g['start']+rows,end=g['end']+rows,source=str(source)) for g in records)
            rows+=meta['rows']
    metadata={k:v for k,v in metas[0].items() if k not in ['config','artifacts','rows']}
    metadata.update(rows=rows,sources=[dict(path=str(s),config=m.get('config'),artifacts=m.get('artifacts')) for s,m in zip(sources,metas)])
    write_json(out/'schema.json',metadata);write_json(out/'games.json',combined)
    print(dict(games=len(combined),rows=rows,out=str(out)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();merge(a.sources,a.out)
