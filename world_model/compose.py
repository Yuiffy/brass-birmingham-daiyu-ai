"""Freeze separately selected dynamics, terminal value and action-policy weights."""
import argparse,hashlib,shutil
from pathlib import Path
from model import read_json,write_json


def compose(args):
    dynamics=Path(args.dynamics);value=Path(args.value);policy=Path(args.policy);out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty output directory')
    model=read_json(dynamics/'world-model.json');critic=read_json(value/'value.json')
    if model['kind']!='gated-world-model' or critic['featureVersion'] not in ['brass-value-v1','brass-value-v2']:raise ValueError('Incompatible artifacts')
    out.mkdir(parents=True,exist_ok=True)
    model.update(valueModel=critic,planningValue='learned')
    if args.shortlist:model['planningShortlist']=args.shortlist
    if args.candidates:model['planningCandidates']=args.candidates
    if args.continuation:model['planningContinuation']=args.continuation
    write_json(out/'world-model.json',model);shutil.copyfile(policy,out/'neural-policy.json')
    for directory,files in [(dynamics,['training.json','delta-optimizer.npz','gate-optimizer.npz','control-optimizer.npz']),
                            (value,['value-training.json','value-fixture.json','value-optimizer.npz'])]:
        for name in files:
            if (directory/name).exists():shutil.copyfile(directory/name,out/name)
    sources=[dynamics/'world-model.json',value/'value.json',policy]
    write_json(out/'composition.json',dict(sources=[dict(path=p.as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        method='Dynamics selected by validation prediction loss; critic selected by validation terminal-value loss; action policy frozen.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dynamics',required=True);p.add_argument('--value',required=True)
    p.add_argument('--policy',default='world_model/experiments/expanded-10k/models/neural-policy.json');p.add_argument('--out',required=True)
    p.add_argument('--shortlist',choices=['legacy','score-diverse-v2'])
    p.add_argument('--candidates',choices=['learned-pool-v1'])
    p.add_argument('--continuation',choices=['legacy','diverse-pool-12','own-diverse-pool-12'])
    compose(p.parse_args())
