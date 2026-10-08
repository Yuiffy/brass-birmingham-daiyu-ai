(function(root){
    'use strict';
    const node=typeof module!=='undefined'&&module.exports;
    const Sim=node?require('./simulator'):root.BrassSimulator;
    const E=node?require('./encoding'):root.BrassEncoding;
    const TYPES={human:'人类玩家',heuristic:'原启发式 AI',search:'搜索树逻辑型',neural:'神经网络型',world:'世界模型型',guided:'学习增强搜索'};
    function shortlist(state,width=8,version='legacy'){
        const all=Sim.candidates(state).sort((a,b)=>b.score-a.score);
        const diverse=[];
        for(const kind of E.schema.actions){const a=all.find(a=>a.action===kind);if(a)diverse.push(a);}
        if(version==='score-diverse-v2')diverse.sort((a,b)=>b.score-a.score);
        else if(version!=='legacy')throw Error('Unknown shortlist version');
        const used=new Set(diverse.map(Sim.key));
        return diverse.concat(all.filter(a=>!used.has(Sim.key(a)))).slice(0,width);
    }
    function informationLimitedCopy(state) {
        const copy=Sim.clone(state),v=E.encodeState(state);
        const seed=v.reduce((a,x,i)=>(a+Math.round(x*997)*(i+1))|0,197);
        const rng=Sim.rng(seed);
        // Sample hidden future cards from the fixed vocabulary. Never inspect
        // the actual draw order or use the real game's RNG state in search.
        const ordinary=E.schema.cards.filter(c=>c.type==='location'||c.type==='industry');
        copy.drawDeck=Array.from({length:state.drawDeck.length},()=>({...rng.pick(ordinary)}));
        copy._rngState=seed;
        return copy;
    }
    function candidatePool(state,limit=32){
        const buckets=new Map();
        for(const a of Sim.candidates(state).sort((a,b)=>b.score-a.score)){
            const key=a.action==='build'?`build:${a.target.industryType}`:a.action;
            if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(a);
        }
        const pool=[];
        while(pool.length<limit){let added=false;for(const bucket of buckets.values()){
            if(bucket.length){pool.push(bucket.shift());added=true;if(pool.length===limit)break;}
        }if(!added)break;}
        return pool;
    }
    function plan(state,{type='world',world,policy,depth=3,width=8}={}) {
        if(!['heuristic','search','neural','world','guided'].includes(type))throw Error('Invalid AI type');
        if(type==='world'&&!world)throw Error('世界模型权重未加载');
        if(type==='neural'&&!policy)throw Error('神经网络权重未加载');
        if(type==='guided'&&!world?.valueLayers)throw Error('局面估值权重未加载');
        if(!Number.isInteger(depth)||depth<1||depth>5||!Number.isInteger(width)||width<1||width>64)throw Error('Use depth 1–5 and width 1–64');
        const shortlistVersion=['world','guided'].includes(type)?world.data?.planningShortlist||'legacy':'legacy';
        const continuationVersion=['world','guided'].includes(type)?world.data?.planningContinuation||'legacy':'legacy';
        const started=performance.now(),actor=state.currentPlayerId,initial=E.encodeState(state);
        let list=shortlist(state,width,shortlistVersion);
        const branches=[];
        const engine=type==='search'||type==='guided';
        const learned=type==='guided'||type==='world'&&world.data?.planningValue==='learned';
        const evaluate=(v,p)=>learned?world.estimateValue(v,p):E.value(v,p);
        const searchRoot=engine?informationLimitedCopy(state):null;
        const transition=(state,vector,action)=>{
            if(engine){const next=Sim.step(state,action,{validate:false}).state;return {state:next,vector:E.encodeState(next)};}
            const v=world.predict(vector,E.encodeAction(action,state));
            return {state:E.decodeState(v,state,{structured:world.data?.kind==='gated-world-model'}),vector:v};
        };
        const rootCache=new Map();
        if((['world','guided'].includes(type)&&world.data?.planningCandidates==='learned-pool-v1')||(type==='search'&&state.rulesVersion==='economy-v2')){
            const pool=candidatePool(state,32);
            for(const a of pool)rootCache.set(Sim.key(a),transition(engine?searchRoot:state,initial,a));
            list=pool.sort((a,b)=>evaluate(rootCache.get(Sim.key(b)).vector,actor)-evaluate(rootCache.get(Sim.key(a)).vector,actor)).slice(0,width);
        }
        for(const action of list) {
            if(type==='heuristic'||type==='neural') {
                const score=type==='heuristic'?action.score:policy.forward([...initial,...E.encodeAction(action,state)])[0]*100;
                branches.push({action,label:Sim.label(action),score,steps:[]});continue;
            }
            let beam=[{state:engine?searchRoot:state,vector:initial,steps:[],score:evaluate(initial,actor)}];
            for(let d=0;d<depth;d++) {
                const expanded=[];
                for(const path of beam) {
                    if(path.state.gameOver){expanded.push(path);continue;}
                    const options=d===0?[action]:continuationVersion==='diverse-pool-12'
                        ?candidatePool(path.state,12)
                        :continuationVersion==='own-diverse-pool-12'&&path.state.currentPlayerId===actor
                            ?candidatePool(path.state,12)
                            :shortlist(path.state,Math.min(3,width),shortlistVersion);
                    if(!options.length){expanded.push(path);continue;}
                    const alternatives=[];
                    for(const candidate of options) {
                        try {
                            const projected=d===0&&rootCache.has(Sim.key(candidate))?rootCache.get(Sim.key(candidate)):transition(path.state,path.vector,candidate);
                            const next=projected.state,vector=projected.vector;
                            const score=evaluate(vector,actor);
                            alternatives.push({state:next,vector,score,steps:path.steps.concat({label:Sim.label(candidate),actor:path.state.currentPlayerId,vector,score})});
                        }catch(error){if(d===0)throw error;}
                    }
                    // Opponents select the branch with highest own estimated value.
                    // Our own successive actions retain a width-2 beam.
                    if(path.state.currentPlayerId!==actor&&alternatives.length) {
                        alternatives.sort((a,b)=>evaluate(b.vector,path.state.currentPlayerId)-evaluate(a.vector,path.state.currentPlayerId));
                        expanded.push(alternatives[0]);
                    }else expanded.push(...alternatives);
                }
                if(!expanded.length)break;
                beam=expanded.sort((a,b)=>b.score-a.score).slice(0,2);
            }
            const best=beam.sort((a,b)=>b.score-a.score)[0];
            branches.push({action,label:Sim.label(action),score:best.score,steps:best.steps});
        }
        branches.sort((a,b)=>b.score-a.score);
        return {type,depth,width,candidatePoolSize:rootCache.size||list.length,branches,selected:branches[0]?.action,elapsedMs:performance.now()-started};
    }
    const api={TYPES,plan,shortlist,candidatePool,informationLimitedCopy};
    if(node)module.exports=api;else root.BrassPlanner=api;
})(globalThis);
