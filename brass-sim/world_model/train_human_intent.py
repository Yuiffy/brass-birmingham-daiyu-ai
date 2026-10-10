"""Fit public action-intent preferences from audited human seats, never JS value targets."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from model import MLP, read_json, write_json


def load_rows(data):
    root=Path(data);audit=read_json(root/'audit.json')
    rows=[json.loads(line) for line in (root/'human-intents.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    accepted={r['sha256']:r for r in audit['reports'] if r.get('terminal') and r.get('nativeLegalityAudited') and r['split']!='diagnostic'}
    reports=[r for r in audit['reports'] if r.get('terminal') and r.get('nativeLegalityAudited') and r['split']!='diagnostic']
    if len(accepted)!=len(reports) or len({r['seed'] for r in reports})!=len(reports):raise ValueError('Duplicate audited game')
    if len(rows)!=audit['humanIntentRows'] or any(r.get('accepted') is False for r in audit['reports']):raise ValueError('Incomplete corpus audit')
    for game,report in accepted.items():
        if sum(r['gameSHA256']==game for r in rows)!=report['humanActions']:raise ValueError('Missing human decisions')
    seen=set();splits={k:set() for k in ['train','validation','test']}
    for r in rows:
        game=accepted.get(r['gameSHA256'])
        if not game or r['actor'] not in game['humanSeats'] or r['source']!=game['source'] or r['split']!=game['split']:
            raise ValueError('Unaudited or non-human sample')
        key=(r['gameSHA256'],r['seq'])
        if not isinstance(r['seq'],int) or not 0<=r['seq']<game['actions'] or r['seed']!=game['seed']:raise ValueError('Invalid decision metadata')
        if r.get('provenance')!='public-anonymized-human-vs-AI-replay':raise ValueError('Invalid human provenance')
        if key in seen:raise ValueError('Duplicate replay decision')
        seen.add(key);splits[r['split']].add(r['gameSHA256'])
        if r.get('valueLabelUsableForJS') is not False or len(r['features'])!=len(audit['featureNames']):raise ValueError('Incompatible human data')
        if set(r)!=set(['source','gameSHA256','seed','seq','actor','split','features','intents','provenance','valueLabelUsableForJS']):raise ValueError('Unexpected human labels')
        if not all(math.isfinite(x) for x in r['features']):raise ValueError('Nonfinite human feature')
        if not r['intents'] or any(x not in audit['classes'] for x in r['intents']):raise ValueError('Unknown human intent')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if splits[a]&splits[b]:raise ValueError('Whole-game leakage')
    for k in splits:
        if not splits[k]:raise ValueError('Missing complete game split: '+k)
    return audit,rows


def softmax(z):
    z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)


def metrics(model,x,y,frequency):
    probability=softmax(model.predict(x))
    return dict(rows=len(x),crossEntropy=float(-(y*np.log(np.maximum(probability,1e-8))).sum(1).mean()),
        constantCrossEntropy=float(-(y*np.log(frequency)).sum(1).mean()),
        top1IntentHit=float((y[np.arange(len(y)),probability.argmax(1)]>0).mean()))


def train(args):
    audit,rows=load_rows(args.data);out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose fresh intent model output')
    if args.epochs<1 or args.hidden<1 or args.lr<=0:raise ValueError('Invalid training configuration')
    out.mkdir(parents=True,exist_ok=True)
    x=np.array([r['features'] for r in rows],'float32');y=np.zeros((len(rows),len(audit['classes'])),'float32')
    for i,r in enumerate(rows):
        for name in r['intents']:y[i,audit['classes'].index(name)]=1/len(r['intents'])
    ids={k:np.array([i for i,r in enumerate(rows) if r['split']==k]) for k in ['train','validation','test']}
    mean=x[ids['train']].mean(0);scale=np.maximum(x[ids['train']].std(0),.05);x=(x-mean)/scale
    frequency=(y[ids['train']].sum(0)+1)/(len(ids['train'])+y.shape[1])
    model=MLP(x.shape[1],y.shape[1],args.hidden,args.seed);model.params[3][:]=np.log(frequency)
    history=[];best=float('inf');rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order=rng.permutation(ids['train']);lr=args.lr*(.1+.9*(1+np.cos(np.pi*epoch/max(1,args.epochs-1)))/2)
        for start in range(0,len(order),64):
            ix=order[start:start+64];prob=softmax(model.predict(x[ix]));model.update_gradient(x[ix],(prob-y[ix])/len(ix),lr)
        result=metrics(model,x[ids['validation']],y[ids['validation']],frequency);history.append(dict(epoch=epoch+1,**result))
        if result['crossEntropy']<best:
            best=result['crossEntropy'];artifact=model.export(featureVersion='public-intent-v1',featureNames=audit['featureNames'],inputDim=x.shape[1],
                classes=audit['classes'],inputMean=mean.tolist(),inputScale=scale.tolist(),classFrequency=frequency.tolist(),epoch=epoch+1,
                trainingGames=len({rows[i]['gameSHA256'] for i in ids['train']}),trainingRows=len(ids['train']),
                provenance='audited-public-human-action-intent; no JS value labels',expertStatusVerified=False)
            write_json(out/'human-intent.json',artifact)
    chosen=MLP.load(read_json(out/'human-intent.json'))
    result=dict(config=vars(args),history=history,validation=metrics(chosen,x[ids['validation']],y[ids['validation']],frequency),
        test=metrics(chosen,x[ids['test']],y[ids['test']],frequency),
        games={k:sorted({rows[i]['gameSHA256'] for i in ids[k]}) for k in ids},
        dataSHA256=hashlib.sha256((Path(args.data)/'human-intents.jsonl').read_bytes()).hexdigest(),
        limitations=['Only six four-player games; one validation and one test game.',
          'Anonymous players may repeat across games; no claim of unseen-player generalization.',
          'Native engine validates complete legality. Cross-engine transfer covers public intent, not exact actions or terminal values.',
          'Published human-seat flags accepted as provenance; expert strength unverified.'])
    write_json(out/'training.json',result)
    i=ids['test'][0];write_json(out/'fixture.json',dict(features=rows[i]['features'],probability=softmax(chosen.predict(x[i:i+1]))[0].tolist()))
    print(dict(validation=result['validation'],test=result['test'],epoch=artifact['epoch']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=160);p.add_argument('--hidden',type=int,default=32);p.add_argument('--lr',type=float,default=.003);p.add_argument('--seed',type=int,default=8123)
    train(p.parse_args())
