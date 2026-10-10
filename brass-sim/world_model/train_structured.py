import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,time,shutil
from pathlib import Path
import numpy as np
from dataset import Dataset,TrainingMixture
from model import read_json,write_json
from structured import StructuredWorld


def statistics(ds,ids):
    count=np.zeros(ds.state_dim);squares=count.copy()
    for start in range(0,len(ids),4096):
        _,d,_=ds.batch(ids[start:start+4096])
        count+=(np.abs(d)>1e-7).sum(0);squares+=np.square(d).sum(0,dtype='float64')
    return np.maximum(np.sqrt(squares/np.maximum(count,1)),.03).astype('float32'),(count/len(ids)).astype('float32')


def validate(model,ds,game_limit=0):
    games=[g for g in ds.games if g['split']=='validation']
    if game_limit:games=games[:game_limit]
    if not games:raise ValueError('No validation games')
    ids=np.concatenate([np.arange(g['start'],g['end']) for g in games]);error=0.;correct=0;changed_count=0;tp=0;claimed=0
    units=np.array([f['scale'] for f in ds.schema['fields']]);current=model.spec['heads'][0]['indices']
    for i in range(0,len(ids),2048):
        x,d,_=ds.batch(ids[i:i+2048]);s=x[:,:ds.state_dim];truth=s+d;p=model.predict(x)
        error+=np.square(p-truth).sum(dtype='float64');correct+=(p[:,current].argmax(1)==truth[:,current].argmax(1)).sum()
        changed=np.abs(d*units)>.1;predicted=np.abs((p-s)*units)>.5
        changed_count+=changed.sum();tp+=(changed&predicted).sum();claimed+=predicted.sum()
    starts=np.array([g['start']+k for g in games for k in range(0,g['end']-g['start']-2,24)])
    state=np.array(ds.data[starts,:ds.state_dim]);rollout=[]
    for depth in range(3):
        rows=ds.data[starts+depth];x=np.concatenate([state,rows[:,ds.state_dim:ds.input_dim]],1)
        state=model.predict(x);rollout.append(float(np.square(state-rows[:,ds.input_dim:-1]).mean()))
    mse=error/(len(ids)*ds.state_dim)
    return dict(validationMSE=float(mse),currentPlayerAccuracy=float(correct/len(ids)),
        changeRecall=float(tp/max(1,changed_count)),changePrecision=float(tp/max(1,claimed)),
        rolloutMSE=rollout,selectionScore=float(mse+.1*rollout[-1]))


def train(args):
    ds=Dataset(args.data);vds=Dataset(args.validation_data);out=Path(args.out)
    if (out/'training.json').exists() or (out/'world-model.json').exists():raise ValueError('Choose a fresh output directory')
    if args.epochs<1 or args.batch_size<1 or args.lr<=0:raise ValueError('Positive epochs, batch size and learning rate required')
    out.mkdir(parents=True,exist_ok=True)
    games=[g for g in ds.games if g['split']=='train'];ids=np.array(ds.splits['train'])
    replay=Dataset(args.replay) if args.replay else None
    reference=Dataset(args.reference_data) if args.reference_data else None
    if not 0<=args.reference_weight<=1 or args.reference_games<0:raise ValueError('Invalid reference validation configuration')
    mixture=TrainingMixture(ds,replay,args.replay_stride,args.seed)
    if replay:games += [g for g in replay.games if g['split']=='train']
    heldout={g['seed'] for g in vds.games if g['split']!='train'}
    if heldout.intersection(g['seed'] for g in games):raise ValueError('Held-out seed leakage')
    if ds.schema['fields']!=vds.schema['fields']:raise ValueError('Validation schema mismatch')
    if ds.schema.get('actionEncoding','legacy')!=vds.schema.get('actionEncoding','legacy'):raise ValueError('Validation action encoding mismatch')
    if reference:
        if ds.schema['fields']!=reference.schema['fields']:raise ValueError('Reference schema mismatch')
        if ds.schema.get('actionEncoding','legacy')!=reference.schema.get('actionEncoding','legacy'):raise ValueError('Reference action encoding mismatch')
        if {g['seed'] for g in reference.games if g['split']!='train'}&{g['seed'] for g in games}:raise ValueError('Reference held-out seed leakage')
    inherited={};base_epoch=0;restored={}
    if args.resume:
        source=Path(args.resume);artifact=read_json(source/'world-model.json')
        if artifact['kind']!='gated-world-model' or artifact['schemaVersion']!=ds.schema['version'] or artifact['stateDim']!=ds.state_dim or artifact['actionDim']!=ds.action_dim:raise ValueError('Resume requires a compatible structured model')
        model=StructuredWorld.load(artifact);model.schema=ds.schema;base_epoch=artifact.get('epoch',0)
        if model.change_weight is None:
            _,freq=statistics(ds,ids);model.change_weight=(1/np.maximum(freq,.02)).astype('float32')
        inherited={k:artifact[k] for k in ['valueModel','planningValue','planningShortlist','planningCandidates','planningContinuation'] if k in artifact}
        for name,component in [('delta',model.delta),('gate',model.gate),('control',model.control)]:
            checkpoint=source/(name+'-optimizer.npz');restored[name]=checkpoint.exists()
            if checkpoint.exists():component.load_optimizer(checkpoint)
    else:
        scale,freq=statistics(ds,ids);model=StructuredWorld(ds.schema,scale,freq,args.hidden,args.seed)
    rng=np.random.default_rng(args.seed);history=[];started=time.time()
    def validation():
        metrics=validate(model,vds)
        if reference:
            ref=validate(model,reference,args.reference_games)
            metrics['primarySelectionScore']=metrics['selectionScore']
            metrics['referenceValidation']=ref
            metrics['selectionScore']=(1-args.reference_weight)*metrics['selectionScore']+args.reference_weight*ref['selectionScore']
        return metrics
    initial=validation() if args.resume else None
    best=initial['selectionScore'] if initial else float('inf')
    def save(epoch):
        metadata=dict(inherited,epoch=epoch,seed=args.seed,observation=ds.schema['observation'],
            datasetGames=ds.schema['games']+(replay.schema['games'] if replay else 0),trainingGames=len(games),trainingRows=mixture.size,
            selectionMetric='Weighted validationMSE + 0.1 * three-step validation MSE' if reference else 'validationMSE + 0.1 * three-step validation MSE',
            dynamicsReplay=dict(primary=args.data,replay=args.replay,replayStride=args.replay_stride,rows=[len(rows) for rows in mixture.rows]) if replay else None,
            actionEncoding=ds.schema.get('actionEncoding','legacy'))
        if args.resume and epoch==base_epoch:metadata=dict(artifact)
        write_json(out/'world-model.json',model.export(**metadata))
        for name,component in [('delta',model.delta),('gate',model.gate),('control',model.control)]:component.save_optimizer(out/(name+'-optimizer.npz'))
    if args.resume:save(base_epoch)
    for epoch in range(args.epochs):
        lr=args.lr*(.1+.9*(1+np.cos(np.pi*epoch/max(1,args.epochs-1)))/2)
        shuffled=rng.permutation(mixture.size);loss=[]
        for i in range(0,mixture.size,args.batch_size):
            x,d,_=mixture.batch(shuffled[i:i+args.batch_size]);loss.append(model.update(x,d,lr))
        metrics=validation()
        record=dict(epoch=base_epoch+epoch+1,lr=float(lr),loss=np.mean(loss,0).tolist(),**metrics,seconds=round(time.time()-started,2))
        history.append(record);print(record,flush=True)
        if metrics['selectionScore']<best:
            best=metrics['selectionScore'];save(base_epoch+epoch+1)
        write_json(out/'training.json',dict(config=vars(args),history=history,initialValidation=initial,baseEpoch=base_epoch,optimizerRestored=restored,bestSelectionScore=best,
            bestValidationMSE=min([r['validationMSE'] for r in history]+([initial['validationMSE']] if initial else [])),architecture='Learned change gates + conditional deltas + categorical controls'))
    if args.policy:shutil.copyfile(args.policy,out/'neural-policy.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='world_model/data');p.add_argument('--validation-data',default='world_model/data')
    p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=24);p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--policy',default='world_model/experiments/expanded-10k/models/neural-policy.json')
    p.add_argument('--resume',help='Directory containing a structured checkpoint and matching optimizers; write to a fresh --out')
    p.add_argument('--replay',help='Older transition dataset; use its training split only')
    p.add_argument('--replay-stride',type=int,default=8,help='Use a seeded random 1/N subset of old training rows')
    p.add_argument('--reference-data',help='Additional validation distribution to protect against forgetting')
    p.add_argument('--reference-weight',type=float,default=.25);p.add_argument('--reference-games',type=int,default=100)
    train(p.parse_args())
