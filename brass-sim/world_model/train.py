import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import time
from pathlib import Path
import numpy as np
from dataset import Dataset
from model import MLP, read_json, write_json


def target_scale(ds, ids):
    # Bounded memory even with millions of transitions; training rows only.
    total = np.zeros(ds.state_dim, 'float64'); square = total.copy()
    for start in range(0,len(ids),4096):
        _,d,_ = ds.batch(ids[start:start+4096])
        total += d.sum(0,dtype='float64')
        square += np.square(d,dtype='float64').sum(0)
    variance = np.maximum(square/len(ids)-(total/len(ids))**2,0)
    return np.maximum(np.sqrt(variance),0.03).astype('float32')


def validation(ds, ids, wm, policy, scale):
    world_error = 0.; policy_error = 0.
    for start in range(0,len(ids),2048):
        x,d,p = ds.batch(ids[start:start+2048])
        world_error += np.square(wm.predict(x)*scale-d).sum(dtype='float64')
        policy_error += np.abs(policy.predict(x)-p).sum(dtype='float64')
    return world_error/(len(ids)*ds.state_dim), policy_error/len(ids)*100


def train(args):
    ds = Dataset(args.data)
    vds = Dataset(args.validation_data) if args.validation_data else ds
    if ds.schema['version'] != vds.schema['version'] or ds.input_dim != vds.input_dim:
        raise ValueError('Training/validation schema mismatch')
    out = Path(args.out)
    if (out/'training.json').exists():
        raise ValueError('Output already contains a run; choose a new --out directory')
    if args.resume and out.resolve() == Path(args.resume).resolve():
        raise ValueError('Resume into a new output directory to preserve the previous model')
    out.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(args.seed)
    training_games = [g for g in ds.games if g['split']=='train']
    if args.max_games: training_games = training_games[:args.max_games]
    ids = np.concatenate([np.arange(g['start'],g['end']) for g in training_games])
    valid = np.array(vds.splits['validation'])
    held_out = {g['seed'] for g in vds.games if g['split']!='train'}
    if held_out.intersection(g['seed'] for g in training_games):
        raise ValueError('Training games overlap external validation/test seeds')
    parents = {}; optimizer_restored = {}
    if args.resume:
        for name in ['world-model','neural-policy']:
            parents[name] = read_json(Path(args.resume)/(name+'.json'))
            if parents[name]['schemaVersion'] != ds.schema['version']:
                raise ValueError('Resume schema mismatch')
        if parents['world-model']['kind']!='delta-world-model':
            raise ValueError('For structured world models use train-structured --resume; dense training cannot preserve categorical/change heads')
        wm = MLP.load(parents['world-model']); policy = MLP.load(parents['neural-policy'])
        scale = np.array(parents['world-model']['deltaScale'],'float32')
        for name,model in [('world-model',wm),('neural-policy',policy)]:
            checkpoint = Path(args.resume)/(name+'-optimizer.npz')
            optimizer_restored[name] = checkpoint.exists()
            if checkpoint.exists(): model.load_optimizer(checkpoint)
            if model.params[0].shape != (ds.input_dim,args.hidden):
                raise ValueError('Resume dimensions differ; keep --hidden equal to the saved model')
    else:
        scale = target_scale(ds,ids)
        wm = MLP(ds.input_dim,ds.state_dim,args.hidden,args.seed)
        policy = MLP(ds.input_dim,1,args.hidden,args.seed+1)
    history = []; started = time.time()
    initial_world,initial_policy = validation(vds,valid,wm,policy,scale)
    initial = dict(validationMSE=initial_world,policyValidationMAE=initial_policy)
    best = initial_world if args.resume else float('inf')
    best_policy = initial_policy if args.resume else float('inf')
    if args.resume:
        for name,model in [('world-model',wm),('neural-policy',policy)]:
            write_json(out/(name+'.json'),parents[name])
            model.save_optimizer(out/(name+'-optimizer.npz'))
    print(dict(initial=initial,trainingGames=len(training_games),trainingRows=len(ids),
               optimizerRestored=optimizer_restored),flush=True)
    for epoch in range(args.epochs):
        fraction = epoch/max(1,args.epochs-1)
        lr = args.lr if args.final_lr is None else args.final_lr+(args.lr-args.final_lr)*(1+np.cos(np.pi*fraction))/2
        shuffled = rng.permutation(ids); losses=[]; policy_losses=[]
        for start in range(0,len(ids),args.batch_size):
            x,d,p = ds.batch(shuffled[start:start+args.batch_size])
            weight = 1 + 4 * (np.abs(d)>1e-7)
            losses.append(wm.update(x,d/scale,weight,lr))
            policy_losses.append(policy.update(x,p,None,lr))
        val,pval = validation(vds,valid,wm,policy,scale)
        record = dict(epoch=epoch+1,learningRate=float(lr),trainLoss=float(np.mean(losses)),
                      policyTrainLoss=float(np.mean(policy_losses)),validationMSE=val,
                      policyValidationMAE=pval,seconds=round(time.time()-started,2))
        history.append(record); print(record,flush=True)
        metadata = dict(schemaVersion=ds.schema['version'],stateDim=ds.state_dim,actionDim=ds.action_dim,
                        observation=ds.schema['observation'],trainingGames=len(training_games),
                        datasetGames=ds.schema['games'],trainingRows=len(ids),seed=args.seed,
                        continuationEpoch=epoch+1,resumedFrom=args.resume,
                        validationDataset=args.validation_data or args.data)
        if val<best:
            best=val
            write_json(out/'world-model.json',wm.export(**metadata,kind='delta-world-model',
                epoch=parents.get('world-model',{}).get('epoch',0)+epoch+1,deltaScale=scale.tolist()))
            wm.save_optimizer(out/'world-model-optimizer.npz')
        if pval<best_policy:
            best_policy=pval
            write_json(out/'neural-policy.json',policy.export(**metadata,kind='heuristic-distillation-policy',
                epoch=parents.get('neural-policy',{}).get('epoch',0)+epoch+1))
            policy.save_optimizer(out/'neural-policy-optimizer.npz')
        write_json(out/'training.json',dict(config=vars(args),initial=initial,optimizerRestored=optimizer_restored,
            history=history,bestValidationMSE=best,bestPolicyValidationMAE=best_policy,
            checkpointPolicy='Best validation checkpoint for each network; optimizer saved with the matching weights.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='world_model/data');p.add_argument('--out',default='world_model/models')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--final-lr',type=float);p.add_argument('--resume');p.add_argument('--validation-data')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-games',type=int,default=0)
    args=p.parse_args()
    if args.epochs<1 or args.hidden<1 or args.batch_size<1 or args.max_games<0: p.error('Invalid training dimensions or epochs')
    if not np.isfinite(args.lr) or args.lr<=0 or args.final_lr is not None and (not np.isfinite(args.final_lr) or args.final_lr<=0):
        p.error('Learning rates must be finite and positive')
    train(args)
