"""Held-out terminal-score calibration and multi-position Python/JS fixtures."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
from pathlib import Path
import numpy as np
from dataset import Dataset
from model import MLP,read_json,write_json
from train_value import replay_samples,targets
from value_model import features


def evaluate(args):
    ds=Dataset(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    artifact=read_json(Path(args.models)/'world-model.json')['valueModel'];m=MLP.load(artifact);weight=artifact.get('opponentWeight',.25)
    income_weight=artifact.get('incomeBonusWeight',0);ix={f['name']:i for i,f in enumerate(ds.schema['fields'])}
    games=[g for g in ds.games if g['split']=='test'][200:300];states=[];labels=[]
    for g in games:
        ids=np.arange(g['start'],g['end'],8)
        income=np.array([ds.data[g['end']-1,ds.input_dim+ix[f'p{p}.income']]*30 for p in range(4)])
        states.extend(ds.data[ids,:ds.state_dim]);labels.extend([np.array([p['vp'] for p in g['finalScores']])-income_weight*income]*len(ids))
    version=artifact['featureVersion']
    states=np.array(states);x=features(states,ds.schema,version).reshape(-1,artifact['inputDim']);y=targets(np.array(labels),weight).reshape(-1,1)
    rx,ry,rg=replay_samples(args.replay,'test',ds.schema,weight,version,income_weight)
    def metrics(x,y):
        pred=m.predict(x);error=(pred-y)*100
        return dict(examples=len(x),rmse=float(np.sqrt(np.square(error).mean())),mae=float(np.abs(error).mean()))
    cases=[]
    for index in [0,5,10,15,160,320]:
        v=states[index:index+1];f=features(v,ds.schema,version)[0]
        cases.append(dict(state=v[0].tolist(),features=f.tolist(),expected=m.predict(f).reshape(-1).tolist()))
    report=dict(objective=artifact['target'],epoch=artifact['epoch'],base=dict(games=len(games),seeds=[g['seed'] for g in games],**metrics(x,y)),
        replay=dict(games=len(rg),seeds=[g['seed'] for g in rg],**metrics(rx,ry)),
        notes='Final outcomes are held-out labels, never inference features. These metrics measure value calibration, not playing strength.')
    write_json(out/'value-evaluation.json',report);write_json(out/'value-cases.json',dict(cases=cases));print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--out',required=True)
    p.add_argument('--data',default='world_model/data-10k');p.add_argument('--replay',default='world_model/data-score-league');evaluate(p.parse_args())
