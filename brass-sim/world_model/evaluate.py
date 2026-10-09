import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
from pathlib import Path
import numpy as np
from dataset import Dataset
from model import MLP, read_json, write_json
from structured import load_predictor

def evaluate(args):
    if args.test_start<0 or args.test_games<0:raise ValueError('Test offsets and counts must be non-negative')
    ds=Dataset(args.data); artifact=read_json(Path(args.models)/'world-model.json')
    if artifact['schemaVersion']!=ds.schema['version'] or artifact['stateDim']!=ds.state_dim:
        raise ValueError('Model/dataset schema mismatch')
    predict=load_predictor(artifact)
    test_games=[g for g in ds.games if g['split']=='test'][args.test_start:]
    if args.test_games:test_games=test_games[:args.test_games]
    if not test_games:raise ValueError('No held-out games selected')
    ids=np.concatenate([np.arange(g['start'],g['end']) for g in test_games]); x,d,_=ds.batch(ids)
    s=x[:,:ds.state_dim]; truth=s+d; pred=predict(x)
    units=np.array([f['scale'] for f in ds.schema['fields']],'float32')
    fields=ds.schema['fields']; actual=truth*units; predicted=pred*units; before=s*units
    changed=np.abs(actual-before)>0.1; claimed=np.abs(predicted-before)>0.5
    def metric(p):
        delta=np.abs(p-actual); tc=changed; pc=np.abs(p-before)>0.5
        tp=int(np.sum(tc&pc)); fp=int(np.sum(~tc&pc)); fn=int(np.sum(tc&~pc))
        return dict(rawMAE=float(delta.mean()),roundedFieldAccuracy=float((np.rint(p)==np.rint(actual)).mean()),
            changedFieldMAE=float(delta[tc].mean()),changePrecision=tp/max(1,tp+fp),changeRecall=tp/max(1,tp+fn),
            changeF1=2*tp/max(1,2*tp+fp+fn),trueChangedFields=int(tc.sum()),predictedChangedFields=int(pc.sum()))
    groups={}
    for group in sorted(set(f['group'] for f in fields)):
        cols=[i for i,f in enumerate(fields) if f['group']==group]
        error=np.abs(predicted[:,cols]-actual[:,cols]); mask=changed[:,cols]
        groups[group]=dict(mae=float(error.mean()),roundedAccuracy=float((np.rint(predicted[:,cols])==np.rint(actual[:,cols])).mean()),
            changedMAE=float(error[mask].mean()) if mask.any() else None,changedCount=int(mask.sum()))
    current=[i for i,f in enumerate(fields) if f['group']=='currentPlayer']
    current_accuracy=float((pred[:,current].argmax(1)==truth[:,current].argmax(1)).mean())
    # Fixed real action tapes, all intermediate state inputs are model outputs.
    # Windows never cross game boundaries. Include normal moves and era switches.
    starts=[]
    for g in test_games:
        starts.extend(range(g['start'],g['end']-4,8))
    starts=np.array(starts,dtype=int)
    if len(starts)>1000: starts=np.random.default_rng(73).choice(starts,1000,replace=False)
    imagined=np.array(ds.data[starts,:ds.state_dim]); initial=imagined.copy(); rollout=[]
    for depth in range(1,6):
        rows=ds.data[starts+depth-1]
        act=rows[:,ds.state_dim:ds.input_dim]
        imagined = predict(np.concatenate([imagined,act],axis=1))
        expected=rows[:,ds.input_dim:-1]
        rollout.append(dict(depth=depth,normalizedMAE=float(np.abs(imagined-expected).mean()),
            normalizedRMSE=float(np.sqrt(np.mean((imagined-expected)**2))),
            persistenceMAE=float(np.abs(initial-expected).mean()),windows=len(starts)))
    # Action-stratified metrics and examples from held-out complete games only.
    action_records={}; samples=[]
    for local,row_id in enumerate(ids):
        ai=int(x[local,ds.state_dim:ds.state_dim+7].argmax())
        action_records.setdefault(ds.schema['actions'][ai],[]).append(local)
    per_action={name:dict(count=len(ii),normalizedMAE=float(np.abs(pred[ii]-truth[ii]).mean()),
                        changedFieldMAE=float(np.abs(predicted[ii]-actual[ii])[changed[ii]].mean()))
                for name,ii in action_records.items()}
    for i in np.random.default_rng(89).choice(len(ids),min(8,len(ids)),replace=False):
        differences=[]
        relevant=np.where(changed[i]|claimed[i])[0]
        for j in sorted(relevant,key=lambda j:abs(predicted[i,j]-actual[i,j]),reverse=True)[:20]:
            differences.append(dict(field=fields[j]['name'],before=float(before[i,j]),predicted=float(predicted[i,j]),
                                    actual=float(actual[i,j]),correct=bool(round(float(predicted[i,j]))==round(float(actual[i,j])))))
        samples.append(dict(row=int(ids[i]),fields=differences))
    report=dict(datasetGames=ds.schema['games'],testGames=len(test_games),testTransitions=len(ids),
        testGameSeeds=[g['seed'] for g in test_games],testStart=args.test_start,
        modelTraining=dict(datasetGames=artifact['datasetGames'],trainingGames=artifact['trainingGames'],trainingRows=artifact['trainingRows']),
        observation=ds.schema['observation'],modelEpoch=artifact['epoch'],currentPlayerAccuracy=current_accuracy,
        normalizedMSE=float(np.mean((pred-truth)**2)),overall=metric(predicted),persistenceBaseline=metric(before),
        groups=groups,byAction=per_action,rollout=rollout,samples=samples,
        notes=['MVP metrics measure agreement with the existing implementation, not conformance to the official rulebook.',
               'Rollouts are conditioned on recorded legal action descriptors; intermediate predicted states are never replaced with truth.',
               'Discrete accuracy can look high because most fields do not change. Inspect change recall/F1 and the persistence baseline.',
               'All hands are visible in this experiment. Future deck order and random-generator state are not encoded.'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);write_json(out/'evaluation.json',report)
    lines=['# World Model Evaluation','',f"Dataset: {ds.schema['games']} complete games; held out: {report['testGames']} games / {len(ids)} transitions.",'',
           f'Current player accuracy: {current_accuracy:.2%}',f"Change precision / recall / F1: {report['overall']['changePrecision']:.2%} / {report['overall']['changeRecall']:.2%} / {report['overall']['changeF1']:.2%}",'',
           '| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |','|---|---:|---:|---:|']
    for name,m in groups.items():lines.append(f"| {name} | {m['mae']:.3f} | {m['roundedAccuracy']:.2%} | {m['changedMAE'] if m['changedMAE'] is not None else 'n/a'} |")
    lines+=['','| Rollout depth | Normalized MAE | Persistence MAE |','|---|---:|---:|']
    for r in rollout:lines.append(f"| {r['depth']} | {r['normalizedMAE']:.5f} | {r['persistenceMAE']:.5f} |")
    lines+=['','See evaluation.json for action breakdowns, sample field differences and limitations.']
    (out/'evaluation.md').write_text('\n'.join(lines),encoding='utf-8')
    # A cross-language fixture verifies the exported browser inference, not just training.
    write_json(out/'inference-fixture.json',dict(input=x[0].tolist(),expected=pred[0].tolist()))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='world_model/data');p.add_argument('--models',default='world_model/models');p.add_argument('--out',default='world_model/reports')
    p.add_argument('--test-start',type=int,default=0);p.add_argument('--test-games',type=int,default=0)
    evaluate(p.parse_args())
