"""Compare incumbent/candidate on whole held-out teacher games, including doubles."""
import argparse, hashlib
from pathlib import Path
import numpy as np
from dataset import Dataset
from model import MLP, read_json, write_json
from structured import StructuredWorld
from train_value import active_samples


def evaluate(data, models, split='test'):
    ds=Dataset(data)
    games=[g for g in ds.games if g['split']==split]
    if not games:raise ValueError('No complete held-out games')
    ids=np.concatenate([np.arange(g['start'],g['end']) for g in games])
    states=np.asarray(ds.data[ids,:ds.state_dim])
    labels=np.concatenate([np.repeat([[p['vp'] for p in g['finalScores']]],g['end']-g['start'],axis=0) for g in games])
    action_start=ds.state_dim+len(ds.schema['actions'])+4+len(ds.schema['slots'])
    doubles=np.flatnonzero(np.asarray(ds.data[ids,action_start:action_start+len(ds.schema['links'])]).sum(1)>1.5)
    reports={}
    for name,directory in models.items():
        path=Path(directory)/'world-model.json';artifact=read_json(path)
        if artifact['schemaVersion']!=ds.schema['version']:raise ValueError('Rules mismatch')
        critic=artifact['valueModel'];value=MLP.load(critic)
        x,y=active_samples(states,labels,ds.schema,critic.get('opponentWeight',0),critic['featureVersion'],critic.get('residualScore',False))
        error=(value.predict(x)-y)*100
        dynamics=StructuredWorld.load(artifact)
        def model_input(inp):
            version=artifact.get('actionEncoding','legacy');data_version=ds.schema.get('actionEncoding','legacy')
            if version==data_version:return inp
            if data_version!='resource-network-v2' or version!='legacy':raise ValueError('Cannot infer missing action encoding information')
            inp=inp.copy();action=inp[:,ds.state_dim:ds.input_dim]
            slot_start=len(ds.schema['actions'])+4
            type_start=slot_start+len(ds.schema['slots'])+len(ds.schema['links'])
            double=(action[:,slot_start+len(ds.schema['slots']):type_start].sum(1)>1.5)
            action[double,slot_start:slot_start+len(ds.schema['slots'])]=0
            action[double,type_start+len(ds.schema['types']):type_start+2*len(ds.schema['types'])]=0
            action[double,-4]=0;action[double,-1]=0
            return inp
        def transition_metrics(indices):
            if not len(indices):return dict(rows=0)
            squared=0.;link_correct=0;money_error=0.;changed_squared=0.;changed_count=0
            scales=np.array([f['scale'] for f in ds.schema['fields']])
            link_ids=[i for i,f in enumerate(ds.schema['fields']) if f['group']=='network' and f['name'].startswith('link.')]
            money_ids=[i for i,f in enumerate(ds.schema['fields']) if f['name'].endswith('.money')]
            for start in range(0,len(indices),512):
                inp,delta,_=ds.batch(ids[indices[start:start+512]])
                truth=inp[:,:ds.state_dim]+delta;pred=dynamics.predict(model_input(inp))
                diff=pred-truth;squared+=np.square(diff).sum()
                changed=np.abs(delta)>1e-7;changed_squared+=np.square(diff[changed]).sum();changed_count+=changed.sum()
                link_correct+=(np.rint(pred[:,link_ids]*scales[link_ids])==np.rint(truth[:,link_ids]*scales[link_ids])).all(1).sum()
                money_error+=np.abs(diff[:,money_ids]*scales[money_ids]).sum()
            return dict(rows=len(indices),mse=float(squared/(len(indices)*ds.state_dim)),
                changedFieldMSE=float(changed_squared/max(1,changed_count)),
                allLinkFieldsExact=float(link_correct/len(indices)),moneyMAE=float(money_error/(len(indices)*len(money_ids))))
        report=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            value=dict(examples=len(error),rmseVP=float(np.sqrt(np.square(error).mean())),maeVP=float(np.abs(error).mean())),
            transitions=transition_metrics(np.arange(len(ids))),doubleRails=transition_metrics(doubles))
        starts=np.array([g['start']+k for g in games for k in range(0,g['end']-g['start']-2,16)])
        state=np.asarray(ds.data[starts,:ds.state_dim]).copy();rollout=[]
        for depth in range(3):
            rows=ds.data[starts+depth];state=dynamics.predict(model_input(np.concatenate([state,rows[:,ds.state_dim:ds.input_dim]],1)))
            rollout.append(float(np.square(state-rows[:,ds.input_dim:-1]).mean()))
        report['threeStepMSE']=rollout;reports[name]=report
    return dict(data=str(data),split=split,games=len(games),seeds=[g['seed'] for g in games],
        provenance=ds.schema.get('provenance'),models=reports,
        note='Synthetic teacher test games, never used for checkpoint selection; prediction metrics do not certify playing strength.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True)
    parser.add_argument('--incumbent',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();result=evaluate(args.data,dict(incumbent=args.incumbent,candidate=args.candidate))
    write_json(args.out,result);print(result)
