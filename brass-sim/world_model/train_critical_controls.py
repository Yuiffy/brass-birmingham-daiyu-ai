"""Train categorical controls while preserving learned board deltas and value weights.

All targets come from synthetic engine trajectories. No engine transition is used
at inference. The original prediction holdout is development data for this round.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,copy,hashlib,time
from pathlib import Path
import numpy as np
from dataset import Dataset
from model import MLP,read_json,write_json
from structured import StructuredWorld,control_input

def prepare(artifact,schema,seed):
    model=StructuredWorld.load(artifact);spec=copy.deepcopy(model.spec)
    public_supply=[i for i,f in enumerate(schema['fields']) if f['group']=='supply']
    spec['inputIndices']=list(dict.fromkeys(spec['inputIndices']+public_supply+list(range(model.state_dim,model.state_dim+model.action_dim))))
    spec['version']='complete-action-control-v2'
    model.spec=spec;model.control=MLP(len(spec['inputIndices'])+16,sum(h['classes'] for h in spec['heads']),128,seed)
    return model

def targets(x,delta,model):
    truth=x[:,:model.state_dim]+delta
    return [truth[:,h['indices']].argmax(1) if h.get('oneHot') else np.rint(truth[:,h['indices'][0]]*h['scale']).astype(int) for h in model.spec['heads']]

def control_metrics(model,x,delta):
    logits=model.control.predict(control_input(x,model.spec));labels=targets(x,delta,model);result={};start=0
    for h,y in zip(model.spec['heads'],labels):
        result[h['name']]=float((logits[:,start:start+h['classes']].argmax(1)==y).mean());start+=h['classes']
    result['selectionError']=sum((1-result[n])*w for n,w in [('currentPlayer',4),('round',2),('actionsThisTurn',3),('actionsPerTurn',3),('era',4),('gameOver',4),('deckSize',1)])
    return result

def train(args):
    ds=Dataset(args.data);vds=Dataset(args.validation_data);source=read_json(args.model)
    if ds.schema['fields']!=vds.schema['fields'] or ds.schema.get('actionEncoding')!=source.get('actionEncoding') or vds.schema.get('actionEncoding')!=source.get('actionEncoding'):raise ValueError('Incompatible critical data')
    if {g['seed'] for g in ds.games if g['split']=='train'}&{g['seed'] for g in vds.games if g['split']!='train'}:raise ValueError('Heldout leakage')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose fresh output')
    out.mkdir(parents=True)
    train_ids=np.array(ds.splits['train']);val_ids=np.array(vds.splits['validation']);vx,vd,_=vds.batch(val_ids)
    model=prepare(source,ds.schema,args.seed);initial=control_metrics(StructuredWorld.load(source),vx,vd)
    rng=np.random.default_rng(args.seed);best=float('inf');history=[];started=time.time()
    for epoch in range(args.epochs):
        lr=args.lr*(.1+.9*(1+np.cos(np.pi*epoch/max(1,args.epochs-1)))/2)
        for start in range(0,len(train_ids),512):
            if start==0:order=rng.permutation(train_ids)
            x,d,_=ds.batch(order[start:start+512]);cx=control_input(x,model.spec);z=model.control.predict(cx);grad=np.zeros_like(z);offset=0
            for h,y in zip(model.spec['heads'],targets(x,d,model)):
                stop=offset+h['classes'];p=z[:,offset:stop]-z[:,offset:stop].max(1,keepdims=True);p=np.exp(p);p/=p.sum(1,keepdims=True)
                p[np.arange(len(x)),y]-=1
                importance={'currentPlayer':4,'actionsThisTurn':3,'actionsPerTurn':3,'era':4,'gameOver':4,'round':2}.get(h['name'],1)
                grad[:,offset:stop]=importance*p/(len(x)*28);offset=stop
            model.control.update_gradient(cx,grad,lr)
        result=dict(epoch=epoch+1,**control_metrics(model,vx,vd));history.append(result)
        if result['selectionError']<best:
            best=result['selectionError'];artifact=copy.deepcopy(source)
            artifact.update(controlSpec=model.spec,controlLayers=model.control.export()['layers'],criticalControls=dict(epoch=epoch+1,seed=args.seed,trainingRows=len(train_ids),validationRows=len(val_ids),sourceModelSHA256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),selection='weighted categorical control error; no test selection'))
            write_json(out/'world-model.json',artifact);model.control.save_optimizer(out/'control-optimizer.npz')
        if epoch%10==0 or epoch==args.epochs-1:print(dict(**result,seconds=round(time.time()-started,2)),flush=True)
    write_json(out/'training.json',dict(config=vars(args),initialValidation=initial,history=history,bestSelectionError=best))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--validation-data',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=90331);p.add_argument('--epochs',type=int,default=100);p.add_argument('--lr',type=float,default=.002)
    train(p.parse_args())
