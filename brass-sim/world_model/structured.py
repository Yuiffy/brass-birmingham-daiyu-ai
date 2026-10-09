"""Learned change gates, conditional deltas and categorical turn dynamics.

No game-engine transitions are used here. The schema supplies field types/scales.
"""
import numpy as np
from model import MLP


def control_spec(schema):
    fields=schema['fields']; ix={f['name']:i for i,f in enumerate(fields)}
    inputs=list(range(20))
    for p in range(4):
        inputs.extend(ix[f'p{p}.{k}'] for k in ['money','income','handSize','spent'])
    inputs.extend(range(len(fields),len(fields)+11))
    inputs.append(len(fields)+schema['actionDim']-8)  # Public action cost.
    heads=[dict(name='currentPlayer',indices=[ix[f'current.{p}'] for p in range(4)],classes=4,oneHot=True)]
    for name,classes in [('era',2),('round',17),('actionsThisTurn',3),('actionsPerTurn',3),
                         ('isFirstRound',2),('gameOver',2),('deckSize',65)]:
        heads.append(dict(name=name,indices=[ix[name]],classes=classes,scale=fields[ix[name]]['scale']))
    for p in range(4):
        name=f'order.{p}'
        heads.append(dict(name=name,indices=[ix[name]],classes=5,scale=4))
    return dict(inputIndices=inputs,orderIndices=[ix[f'order.{p}'] for p in range(4)],heads=heads)


def control_input(x,spec):
    # One-hot representation of the current ordering, not its successor.
    order=np.rint(x[:,spec['orderIndices']]*4).astype(int)
    order_onehot=(order[:,:,None]==np.arange(1,5)[None,None,:]).reshape(len(x),16)
    return np.concatenate([x[:,spec['inputIndices']],order_onehot.astype('float32')],axis=1)


class StructuredWorld:
    def __init__(self,schema,scale,frequency,hidden=128,seed=42):
        self.schema=schema;self.state_dim=schema['stateDim'];self.action_dim=schema['actionDim']
        self.spec=control_spec(schema);self.scale=np.asarray(scale,'float32')
        self.positive_weight=np.minimum(np.sqrt((1-frequency)/(frequency+1e-4)),20).clip(1,20).astype('float32')
        self.change_weight=(1/np.maximum(frequency,.02)).astype('float32')
        self.delta=MLP(self.state_dim+self.action_dim,self.state_dim,hidden,seed)
        self.gate=MLP(self.state_dim+self.action_dim,self.state_dim,hidden,seed+1)
        self.gate.params[3][:]=np.log((frequency+.0001)*self.positive_weight/(1-frequency+.0001))
        self.control=MLP(len(self.spec['inputIndices'])+16,sum(h['classes'] for h in self.spec['heads']),96,seed+2)
        self.gate_threshold=.35

    def predict(self,x):
        state=x[:,:self.state_dim]
        threshold=np.log(self.positive_weight*self.gate_threshold/(1-self.gate_threshold))
        mask=self.gate.predict(x)>threshold
        predicted=state+self.delta.predict(x)*self.scale*mask
        logits=self.control.predict(control_input(x,self.spec)); start=0
        for h in self.spec['heads']:
            chosen=logits[:,start:start+h['classes']].argmax(1);start+=h['classes']
            if h.get('oneHot'):predicted[:,h['indices']]=(chosen[:,None]==np.arange(h['classes'])).astype('float32')
            else:predicted[:,h['indices'][0]]=chosen/h['scale']
        return predicted

    def update(self,x,delta,lr):
        changed=(np.abs(delta)>1e-7).astype('float32')
        regression=self.delta.update(x,delta/self.scale,.02+changed*self.change_weight,lr)
        logits=self.gate.predict(x)
        probability=1/(1+np.exp(-np.clip(logits,-40,40)))
        weight=1+changed*(self.positive_weight-1)
        grad=(probability-changed)*weight/probability.size
        self.gate.update_gradient(x,grad,lr)
        gate_loss=float((weight*(np.maximum(logits,0)-logits*changed+np.log1p(np.exp(-np.abs(logits))))).mean())
        cx=control_input(x,self.spec);logits=self.control.predict(cx);gradient=np.zeros_like(logits)
        target=x[:,:self.state_dim]+delta;control_loss=0.;start=0
        for h in self.spec['heads']:
            stop=start+h['classes'];z=logits[:,start:stop];z=z-z.max(1,keepdims=True)
            prob=np.exp(z);prob/=prob.sum(1,keepdims=True)
            if h.get('oneHot'):label=target[:,h['indices']].argmax(1)
            else:label=np.rint(target[:,h['indices'][0]]*h['scale']).astype(int)
            if np.any(label<0) or np.any(label>=h['classes']):raise ValueError(f"Category outside schema: {h['name']}")
            importance=2 if h['name']=='currentPlayer' else 1
            control_loss+=importance*float(-np.log(np.maximum(prob[np.arange(len(x)),label],1e-20)).mean())
            prob[np.arange(len(x)),label]-=1
            gradient[:,start:stop]=importance*prob/(len(x)*(len(self.spec['heads'])+1));start=stop
        self.control.update_gradient(cx,gradient,lr)
        return regression,gate_loss,control_loss/(len(self.spec['heads'])+1)

    def export(self,**metadata):
        return dict(metadata,kind='gated-world-model',schemaVersion=self.schema['version'],stateDim=self.state_dim,
            actionDim=self.action_dim,layers=self.delta.export()['layers'],deltaScale=self.scale.tolist(),
            gateLayers=self.gate.export()['layers'],gatePositiveWeight=self.positive_weight.tolist(),
            gateThreshold=self.gate_threshold,changeWeight=self.change_weight.tolist(),controlLayers=self.control.export()['layers'],controlSpec=self.spec)

    @classmethod
    def load(cls,data):
        m=object.__new__(cls);m.state_dim=data['stateDim'];m.action_dim=data['actionDim']
        m.spec=data['controlSpec'];m.scale=np.array(data['deltaScale'],'float32');m.gate_threshold=data['gateThreshold']
        m.positive_weight=np.array(data['gatePositiveWeight'],'float32')
        m.change_weight=np.array(data['changeWeight'],'float32') if 'changeWeight' in data else None
        m.delta=MLP.load(data);m.gate=MLP.load({'layers':data['gateLayers']});m.control=MLP.load({'layers':data['controlLayers']})
        return m


def load_predictor(artifact):
    if artifact['kind']=='gated-world-model':return StructuredWorld.load(artifact).predict
    model=MLP.load(artifact);scale=np.asarray(artifact['deltaScale'],'float32')
    return lambda x:x[:,:artifact['stateDim']]+model.predict(x)*scale
