"""Role-symmetric observable features for learning terminal VP margin."""
import numpy as np
import json
from pathlib import Path

PLAYER_KEYS=['money','income','vp','handSize','spent','canal','rail','wildLocation','wildIndustry']
GLOBAL_KEYS=['era','round','players','deckSize','actionsThisTurn','actionsPerTurn','gameOver','coalMarket','ironMarket']

def features(states,schema,version='brass-value-v1'):
    if version not in ['brass-value-v1','brass-value-v2']:raise ValueError('Unknown value feature version')
    ix={f['name']:i for i,f in enumerate(schema['fields'])};types=schema['types'];slots=schema['slots'];links=schema['links']
    col=lambda name:states[:,ix[name]]
    slot=lambda key:states[:,[ix[f'slot.{s}.{key}'] for s in slots]]
    owner=np.rint(slot('owner')*4);kind=np.rint(slot('type')*6);persistent=(slot('level')*8>=1.5)
    flip=np.clip(slot('flipped'),0,1);vp=np.maximum(slot('vp')*20,0);cubes=np.maximum(slot('cubes')*8,0)
    link_owner=np.rint(states[:,[ix[f'link.{s}.owner'] for s in links]]*4)
    rail=np.clip(states[:,[ix[f'link.{s}.rail'] for s in links]],0,1)
    roles=[]
    for p in range(4):
        owned=(owner==p+1);columns=[col(f'p{p}.{k}') for k in PLAYER_KEYS]+[col(f'p{p}.used.{t}') for t in types]
        for t in range(1,len(types)+1):
            mask=owned*(kind==t)
            for keep in [False,True]:
                for sold in [False,True]:
                    columns.append((mask*(persistent==keep)*(flip if sold else 1-flip)).sum(1)/4)
        columns.extend([(owned*vp*flip).sum(1)/100,(owned*vp*(1-flip)).sum(1)/100])
        columns.extend((owned*(kind==t)*cubes).sum(1)/10 for t in range(1,len(types)+1))
        columns.extend([((link_owner==p+1)*(1-rail)).sum(1)/14,((link_owner==p+1)*rail).sum(1)/14])
        roles.append(np.stack(columns,1))
    roles=np.stack(roles,1);total=roles.sum(1);players=np.maximum(np.rint(col('players')*4)-1,1)
    glob=np.stack([col(k) for k in GLOBAL_KEYS],1)
    base=np.stack([np.concatenate([glob,roles[:,p],(total-roles[:,p])/players[:,None],col(f'current.{p}')[:,None],
        np.stack([col(f'p{k}.vp') for k in range(4) if k!=p],1).max(1,keepdims=True)],1) for p in range(4)],1).astype('float32')
    if version=='brass-value-v1':return base
    topology=json.loads((Path(__file__).parent/'value_topology.json').read_text(encoding='utf-8'))
    if topology['links']!=links or topology['slots']!=slots:raise ValueError('Value topology mismatch')
    icon=np.maximum(slot('linkVP')*3,0)*((owner>=1)&(owner<=4))
    adjacency=np.zeros((len(slots),len(links)),dtype='float32')
    for k,t in enumerate(topology['topology']):adjacency[t['slots'],k]=1
    scored=(icon*flip)@adjacency+np.array([t['merchantVP'] for t in topology['topology']])
    potential=(icon*(1-flip))@adjacency
    sellable=np.isin(kind,[i+1 for i,t in enumerate(types) if t in ['cottonMill','manufacturer','pottery']])
    extra=[]
    for p in range(4):
        owned=owner==p+1;network=link_owner==p+1
        extra.append(np.stack([(network*scored).sum(1),(network*potential).sum(1),
            (owned*persistent*vp*flip).sum(1),(owned*persistent*vp*(1-flip)).sum(1),
            (network*((owned*icon*flip)@adjacency)).sum(1),(network*((owned*icon*(1-flip))@adjacency)).sum(1),
            (owned*sellable*vp*(1-flip)).sum(1),(owned*sellable*persistent*vp*(1-flip)).sum(1)],1)/100)
    extra=np.stack(extra,1);total_extra=extra.sum(1)
    return np.concatenate([base,extra,np.stack([(total_extra-extra[:,p])/players[:,None] for p in range(4)],1)],2).astype('float32')


def score_anchors(states,schema):
    """Observable VP already earned plus VP if the current era ended now."""
    ix={f['name']:i for i,f in enumerate(schema['fields'])}
    scale={f['name']:f['scale'] for f in schema['fields']}
    raw=lambda key:states[:,ix[key]]*scale[key]
    values=features(states,schema,'brass-value-v2')
    anchors=np.stack([raw(f'p{p}.vp') for p in range(4)],1)
    for p in range(4):
        industry=sum((np.rint(raw(f'slot.{s}.owner'))==p+1)*np.clip(raw(f'slot.{s}.flipped'),0,1)*np.maximum(raw(f'slot.{s}.vp'),0) for s in schema['slots'])
        anchors[:,p]+=np.where(raw('gameOver')>=.5,0,industry+values[:,p,109]*100)
    return anchors
