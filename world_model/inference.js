(function(root){
    'use strict';
    const node=typeof module!=='undefined'&&module.exports;
    const E=node?require('./encoding'):root.BrassEncoding;
    const V=node?require('./value_features'):root.BrassValueFeatures;
    function layers(raw,inputSize,outputSize){
        if(!Array.isArray(raw)||!raw.length)throw Error('Missing model layers');
        const result=raw.map(l=>({weights:l.weights.map(r=>Float32Array.from(r)),bias:Float32Array.from(l.bias),activation:l.activation}));
        for(const l of result){
            if(l.weights.length!==inputSize||l.weights.some(r=>r.length!==l.bias.length)||!l.bias.length)throw Error('Invalid model dimensions');
            if(!['relu','linear'].includes(l.activation)||l.weights.some(r=>r.some(v=>!Number.isFinite(v)))||l.bias.some(v=>!Number.isFinite(v)))throw Error('Invalid model weights or activation');
            inputSize=l.bias.length;
        }
        if(outputSize!==undefined&&inputSize!==outputSize)throw Error('Invalid output dimensions');
        return result;
    }
    function run(network,input){
        let x=input;
        for(const l of network){
            const y=Float64Array.from(l.bias);
            for(let i=0;i<x.length;i++){const w=l.weights[i],v=x[i];if(v===0)continue;for(let j=0;j<y.length;j++)y[j]+=v*w[j];}
            if(l.activation==='relu')for(let j=0;j<y.length;j++)y[j]=Math.max(0,y[j]);
            x=y;
        }
        return Array.from(x);
    }
    class Network {
        constructor(data) {
            if(data.schemaVersion!==E.schema.version||data.stateDim!==E.fields.length) throw Error('Incompatible model schema');
            this.data=data;
            const inputSize=data.stateDim+data.actionDim;
            this.layers=layers(data.layers,inputSize);
            if(['delta-world-model','gated-world-model'].includes(data.kind)&&
                (this.layers.at(-1).bias.length!==data.stateDim||data.deltaScale?.length!==data.stateDim||data.deltaScale.some(v=>!Number.isFinite(v)||v<=0)))throw Error('Invalid dynamics output');
            if(data.kind==='gated-world-model'){
                const spec=data.controlSpec;
                if(!spec||!Array.isArray(spec.inputIndices)||spec.inputIndices.some(i=>!Number.isInteger(i)||i<0||i>=inputSize)||
                    spec.orderIndices?.length!==4||spec.orderIndices.some(i=>!Number.isInteger(i)||i<0||i>=data.stateDim)||!Array.isArray(spec.heads))throw Error('Invalid control inputs');
                for(const h of spec.heads){
                    if(!Number.isInteger(h.classes)||h.classes<2||!Array.isArray(h.indices)||h.indices.some(i=>!Number.isInteger(i)||i<0||i>=data.stateDim)||
                        (h.oneHot?h.indices.length!==h.classes:h.indices.length!==1||!Number.isFinite(h.scale)||h.scale<=0))throw Error('Invalid control head');
                }
                if(data.gatePositiveWeight?.length!==data.stateDim||data.gatePositiveWeight.some(v=>!Number.isFinite(v)||v<=0)||!(data.gateThreshold>0&&data.gateThreshold<1))throw Error('Invalid change gate');
                this.gateLayers=layers(data.gateLayers,inputSize,data.stateDim);
                this.controlLayers=layers(data.controlLayers,spec.inputIndices.length+16,spec.heads.reduce((n,h)=>n+h.classes,0));
                this.thresholds=data.gatePositiveWeight.map(w=>Math.log(w*data.gateThreshold/(1-data.gateThreshold)));
            }
            if(data.valueModel){
                const dimensions={'brass-value-v1':109,'brass-value-v2':125};
                if(!dimensions[data.valueModel.featureVersion]||data.valueModel.inputDim!==dimensions[data.valueModel.featureVersion]||data.valueModel.scale!==100)throw Error('Invalid learned value model');
                this.opponentWeight=data.valueModel.opponentWeight??.25;
                this.incomeBonusWeight=data.valueModel.incomeBonusWeight??0;
                if(!Number.isFinite(this.incomeBonusWeight)||this.incomeBonusWeight<0||this.incomeBonusWeight>1)throw Error('Invalid income bonus objective');
                if(!Number.isFinite(this.opponentWeight)||this.opponentWeight<0||this.opponentWeight>1)throw Error('Invalid value objective');
                this.valueLayers=layers(data.valueModel.layers,data.valueModel.inputDim,1);
            }
        }
        forward(input) {
            if(input.length!==this.data.stateDim+this.data.actionDim||input.some(v=>!Number.isFinite(v))) throw Error('Invalid model input');
            return run(this.layers,input);
        }
        predict(stateVector,actionVector) {
            if(!['delta-world-model','gated-world-model'].includes(this.data.kind))throw Error('This is not a dynamics model');
            const input=[...stateVector,...actionVector],delta=this.forward(input);
            if(this.data.kind==='delta-world-model')return stateVector.map((v,i)=>v+delta[i]*this.data.deltaScale[i]);
            const gates=run(this.gateLayers,input),spec=this.data.controlSpec;
            const predicted=stateVector.map((v,i)=>gates[i]>this.thresholds[i]?v+delta[i]*this.data.deltaScale[i]:v);
            const cx=spec.inputIndices.map(i=>input[i]);
            for(const i of spec.orderIndices)for(let p=1;p<=4;p++)cx.push(+(Math.round(input[i]*4)===p));
            const logits=run(this.controlLayers,cx);let start=0;
            for(const h of spec.heads){
                let chosen=0;for(let k=1;k<h.classes;k++)if(logits[start+k]>logits[start+chosen])chosen=k;
                if(h.oneHot)h.indices.forEach((i,k)=>predicted[i]=+(k===chosen));else predicted[h.indices[0]]=chosen/h.scale;
                start+=h.classes;
            }
            return predicted;
        }
        imagine(stateVector,actionVectors) {
            let v=stateVector.slice();
            return actionVectors.map(a=>v=this.predict(v,a));
        }
        estimateValue(stateVector,player){
            if(!this.valueLayers)throw Error('局面估值权重未加载');
            if(!Number.isInteger(player)||player<0||player>3)throw Error('Invalid value perspective');
            if(E.raw(stateVector,'gameOver')>=.5){
                const scores=[0,1,2,3].map(p=>E.raw(stateVector,`p${p}.vp`)-this.incomeBonusWeight*E.raw(stateVector,`p${p}.income`));
                return scores[player]-this.opponentWeight*Math.max(...scores.filter((_,p)=>p!==player));
            }
            return run(this.valueLayers,V.features(stateVector,player,this.data.valueModel.featureVersion))[0]*100;
        }
    }
    if(node)module.exports={Network};else root.BrassWorldModel={Network};
})(globalThis);
