import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,time
from pathlib import Path
import numpy as np
from dataset import Dataset
from model import MLP,read_json,write_json
from value_model import features,score_anchors

def targets(labels,opponent_weight,counts=None):
    if counts is None:counts=np.full(len(labels),4)
    active=np.arange(4)[None,:]<np.asarray(counts)[:,None]
    return np.stack([labels[:,p]-opponent_weight*np.where(active[:,[k for k in range(4) if k!=p]],
        labels[:,[k for k in range(4) if k!=p]],-np.inf).max(1) for p in range(4)],1)/100


def active_samples(states,labels,schema,opponent_weight,version,residual_score=False):
    """Padding preserves the export schema, but absent seats are never examples."""
    index=next(i for i,f in enumerate(schema['fields']) if f['name']=='players')
    counts=np.rint(states[:,index]*schema['fields'][index]['scale']).astype(int)
    if np.any((counts<2)|(counts>4)):raise ValueError('Invalid observed player count')
    mask=np.arange(4)[None,:]<counts[:,None]
    x=features(states,schema,version);y=targets(labels,opponent_weight,counts)
    if residual_score:
        if opponent_weight!=0:raise ValueError('Residual score requires an own-VP objective')
        y-=score_anchors(states,schema)/100
    return x[mask],y[mask,None]


def replay_samples(directory,split,schema,opponent_weight,version='brass-value-v1',income_bonus_weight=0,residual_score=False):
    path=Path(directory);meta=read_json(path/'schema.json');games=read_json(path/'games.json')
    if meta['fields']!=schema['fields'] or meta['stateDim']!=schema['stateDim']:raise ValueError('Replay schema mismatch')
    if (path/'states.f32').stat().st_size!=meta['rows']*meta['rowWidth']*4:raise ValueError('Incomplete replay')
    data=np.memmap(path/'states.f32',mode='r',dtype='<f4',shape=(meta['rows'],meta['rowWidth']))
    selected=[g for g in games if g['split']==split]
    ids=np.concatenate([np.arange(g['start'],g['end']) for g in selected]);rows=np.array(data[ids])
    labels=rows[:,-4:].copy()
    ix={f['name']:i for i,f in enumerate(schema['fields'])};cursor=0
    for g in selected:
        income=np.array([data[g['end']-1,ix[f'p{p}.income']]*30 for p in range(4)])
        count=g['end']-g['start'];labels[cursor:cursor+count]-=income_bonus_weight*income;cursor+=count
    x,y=active_samples(rows[:,:meta['stateDim']],labels,schema,opponent_weight,version,residual_score)
    return x,y,selected

def samples(ds,split,stride,opponent_weight=.25,version='brass-value-v1',income_bonus_weight=0,residual_score=False):
    games=[g for g in ds.games if g['split']==split];ids=[];scores=[]
    ix={f['name']:i for i,f in enumerate(ds.schema['fields'])}
    def outcome(g):
        scores=np.zeros(4,'float32')
        for p in g['finalScores']:scores[p['id']]=p['vp']
        return scores-income_bonus_weight*np.array([ds.data[g['end']-1,ds.input_dim+ix[f'p{p}.income']]*30 for p in range(4)])
    for g in games:
        ii=list(range(g['start'],g['end'],stride));ids.extend(ii)
        scores.extend([outcome(g)]*len(ii))
    s=np.array(ds.data[ids,:ds.state_dim]);labels=np.array(scores,'float32')
    terminal=np.array(ds.data[[g['end']-1 for g in games],ds.input_dim:-1])
    final=np.array([outcome(g) for g in games],'float32')
    s=np.concatenate([s,terminal],0);labels=np.concatenate([labels,final],0)
    x,target=active_samples(s,labels,ds.schema,opponent_weight,version,residual_score)
    return x,target,s

def train(args):
    artifact=None
    if args.resume:
        source=Path(args.resume);artifact=read_json(source/'value.json') if (source/'value.json').exists() else read_json(source/'world-model.json')['valueModel']
    if args.opponent_weight is None:args.opponent_weight=artifact.get('opponentWeight',.25) if artifact else .25
    if args.income_bonus_weight is None:args.income_bonus_weight=artifact.get('incomeBonusWeight',0) if artifact else 0
    version=args.feature_version or (artifact['featureVersion'] if artifact else 'brass-value-v1')
    if min(args.epochs,args.batch_size,args.stride,args.hidden,args.replay_weight)<1 or args.lr<=0 or not 0<=args.opponent_weight<=1 or not 0<=args.income_bonus_weight<=1:raise ValueError('Invalid training parameters')
    ds=Dataset(args.data);vds=Dataset(args.validation_data);out=Path(args.out)
    if (out/'value.json').exists():raise ValueError('Choose a fresh value-model output')
    out.mkdir(parents=True,exist_ok=True)
    if artifact and artifact.get('schemaVersion','brass-wm-v1')!=ds.schema['version']:raise ValueError('Value resume rules/schema mismatch')
    if ds.schema.get('rulesVersion','legacy-v1')=='economy-v2' and args.income_bonus_weight:raise ValueError('Corrected rules have no income bonus to subtract')
    if ds.schema['fields']!=vds.schema['fields']:raise ValueError('Validation schema mismatch')
    train_seeds={g['seed'] for g in ds.games if g['split']=='train'}
    if train_seeds.intersection(g['seed'] for g in vds.games if g['split']!='train'):raise ValueError('Held-out overlap')
    x,y,_=samples(ds,'train',args.stride,args.opponent_weight,version,args.income_bonus_weight,args.residual_score);vx,vy,vs=samples(vds,'validation',args.stride,args.opponent_weight,version,args.income_bonus_weight,args.residual_score)
    replay_info=None;rvx=None
    if args.replay:
        rx,ry,rg=replay_samples(args.replay,'train',ds.schema,args.opponent_weight,version,args.income_bonus_weight,args.residual_score)
        rvx,rvy,rvg=replay_samples(args.replay,'validation',ds.schema,args.opponent_weight,version,args.income_bonus_weight,args.residual_score)
        all_replay=read_json(Path(args.replay)/'games.json')
        replay_train={g['seed'] for g in rg};heldout={g['seed'] for g in ds.games+vds.games+all_replay if g['split']!='train'}
        if (train_seeds|replay_train).intersection(heldout):raise ValueError('Replay held-out leakage')
        if train_seeds.intersection(replay_train):raise ValueError('Replay duplicates base training seeds')
        replay_info=dict(trainingGames=len(rg),trainingExamples=len(rx),validationGames=len(rvg),weight=args.replay_weight)
        train_seeds|=replay_train
        x=np.concatenate([x]+[rx]*args.replay_weight);y=np.concatenate([y]+[ry]*args.replay_weight)
    base_epoch=0;optimizer_restored=False
    if args.resume:
        migration=artifact['featureVersion']=='brass-value-v1' and version=='brass-value-v2'
        if not migration and (artifact['featureVersion']!=version or artifact['inputDim']!=x.shape[1]):raise ValueError('Value schema mismatch')
        model=MLP.load(artifact);base_epoch=artifact['epoch']
        if (source/'value-optimizer.npz').exists() and artifact.get('residualScore',False)==args.residual_score:model.load_optimizer(source/'value-optimizer.npz');optimizer_restored=True
        if migration:
            # New geography inputs are appended. Preserve the old predictor and
            # optimizer exactly; only the new input connections start at zero.
            for values in [model.params,model.m,model.v]:values[0]=np.pad(values[0],((0,x.shape[1]-artifact['inputDim']),(0,0)))
    else:model=MLP(x.shape[1],1,args.hidden,args.seed)
    def validate():
        base=float(np.square(model.predict(vx)-vy).mean())
        replay=float(np.square(model.predict(rvx)-rvy).mean()) if rvx is not None else None
        return (base+replay)/2 if replay is not None else base,base,replay
    rng=np.random.default_rng(args.seed);initial_mse=validate()[0] if args.resume else None
    best=initial_mse if args.resume else float('inf');history=[];started=time.time()
    def save(epoch):
        metadata=dict(schemaVersion=ds.schema['version'],residualScore=args.residual_score,kind='terminal-vp' if args.opponent_weight==0 else 'terminal-vp-margin',featureVersion=version,inputDim=x.shape[1],
            hidden=len(model.params[1]),epoch=epoch,trainingExamples=len(x),trainingGames=len(train_seeds),seed=args.seed,opponentWeight=args.opponent_weight,
            incomeBonusWeight=args.income_bonus_weight,trainingSeats='active-only',
            target=f'score[player] - {args.opponent_weight} * max(score[opponents]); score = finalVP - {args.income_bonus_weight} * finalIncomeBonus',scale=100,replay=replay_info)
        if args.resume and epoch==base_epoch:metadata={k:v for k,v in artifact.items() if k!='layers'};metadata.update(featureVersion=version,inputDim=x.shape[1])
        write_json(out/'value.json',model.export(**metadata))
        fixture=features(vs[:1],ds.schema,version)[0]
        expected=model.predict(fixture).reshape(-1)+(score_anchors(vs[:1],ds.schema)[0]/100 if args.residual_score else 0)
        write_json(out/'value-fixture.json',dict(state=vs[0].tolist(),features=fixture.tolist(),expected=expected.tolist()))
        model.save_optimizer(out/'value-optimizer.npz')
    # An objective change needs at least one fitted checkpoint with the new target.
    if args.resume and artifact.get('opponentWeight',.25)==args.opponent_weight and artifact.get('incomeBonusWeight',0)==args.income_bonus_weight and artifact.get('residualScore',False)==args.residual_score:save(base_epoch)
    else:best=float('inf')
    for epoch in range(args.epochs):
        lr=args.lr*(.1+.9*(1+np.cos(np.pi*epoch/max(1,args.epochs-1)))/2);indices=rng.permutation(len(x));loss=[]
        for i in range(0,len(x),args.batch_size):
            batch=indices[i:i+args.batch_size];loss.append(model.update(x[batch],y[batch],lr=lr))
        mse,base_mse,replay_mse=validate();record=dict(epoch=base_epoch+epoch+1,validationMSE=mse,baseValidationMSE=base_mse,replayValidationMSE=replay_mse,
            validationRMSEVP=float(np.sqrt(mse)*100),trainLoss=float(np.mean(loss)),seconds=round(time.time()-started,2))
        history.append(record);print(record,flush=True)
        if mse<best:
            best=mse;save(base_epoch+epoch+1)
        write_json(out/'value-training.json',dict(config=vars(args),history=history,bestValidationMSE=best,
            initialValidationMSE=initial_mse,baseEpoch=base_epoch,optimizerRestored=optimizer_restored,replay=replay_info,
            selectionMetric='Equal mean of base and replay validation MSE' if args.replay else 'Base validation MSE',
            baselineConstantMSE=float(np.square(vy-y.mean()).mean()),notes='Final outcomes are labels only; whole held-out games are excluded from fitting.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='world_model/data-10k');p.add_argument('--validation-data',default='world_model/data')
    p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=32);p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=1024);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=73)
    p.add_argument('--stride',type=int,default=16);p.add_argument('--opponent-weight',type=float,help='0 for own VP; defaults to the resumed objective, or 0.25 for a new legacy model')
    p.add_argument('--resume',help='Value checkpoint directory or composed world-model directory')
    p.add_argument('--feature-version',choices=['brass-value-v1','brass-value-v2'],help='Defaults to checkpoint version; v2 adds observable network and persistent-industry value')
    p.add_argument('--income-bonus-weight',type=float,help='1 excludes the engine-only final income bonus; defaults to the resumed objective or 0')
    p.add_argument('--replay',help='Additional real games collected with collect-value')
    p.add_argument('--replay-weight',type=int,default=4,help='Training repetitions for each replay sample')
    p.add_argument('--residual-score',action='store_true',help='Learn remaining VP beyond observable current-era score; requires opponent-weight 0')
    train(p.parse_args())
