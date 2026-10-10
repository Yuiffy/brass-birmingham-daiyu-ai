"""Report exact critical successor fields, stratified by action; no engine repair."""
import argparse,hashlib
from pathlib import Path
import numpy as np
from dataset import Dataset
from model import read_json,write_json
from structured import StructuredWorld

def evaluate(data,models,split='test'):
    ds=Dataset(data);ids=np.array(ds.splits[split]);x,d,_=ds.batch(ids);truth=x[:,:ds.state_dim]+d
    units=np.array([f['scale'] for f in ds.schema['fields']]);names=[f['name'] for f in ds.schema['fields']];reports={}
    groups={'currentPlayer':[i for i,n in enumerate(names) if n.startswith('current.')],
            'round':[names.index('round')],'actionsThisTurn':[names.index('actionsThisTurn')],
            'actionsPerTurn':[names.index('actionsPerTurn')],'deckSize':[names.index('deckSize')],
            'supply':[i for i,f in enumerate(ds.schema['fields']) if f['group']=='supply'],
            'links':[i for i,n in enumerate(names) if n.startswith('link.')],
            'cardCounts':[i for i,n in enumerate(names) if '.card.' in n]}
    money=[i for i,n in enumerate(names) if n.endswith('.money')]
    for label,path in models.items():
        artifact=read_json(path)
        if artifact.get('actionEncoding','legacy')!=ds.schema.get('actionEncoding','legacy'):raise ValueError('Action encoding mismatch')
        model=StructuredWorld.load(artifact);pred=np.concatenate([model.predict(x[i:i+512]) for i in range(0,len(x),512)])
        def metrics(mask):
            if not mask.any():return {'rows':0}
            result={k:float((np.rint(pred[mask][:,ix]*units[ix])==np.rint(truth[mask][:,ix]*units[ix])).all(1).mean()) for k,ix in groups.items()}
            return dict(rows=int(mask.sum()),exact=result,moneyMAE=float(np.abs((pred[mask][:,money]-truth[mask][:,money])*units[money]).mean()))
        reports[label]=dict(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),all=metrics(np.ones(len(ids),bool)),
            byAction={action:metrics(x[:,ds.state_dim+i]>.5) for i,action in enumerate(ds.schema['actions'])})
    return dict(data=data,split=split,games=[g['seed'] for g in ds.games if g['split']==split],models=reports,
                limitation='Future draw identities are stochastic. Card-count exactness is descriptive, not a fully reducible target.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--incumbent',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);p.add_argument('--split',default='test')
    a=p.parse_args();result=evaluate(a.data,dict(incumbent=a.incumbent,candidate=a.candidate),a.split);write_json(a.out,result);print(result)
