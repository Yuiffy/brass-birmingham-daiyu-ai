const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const Sim=require('../world_model/simulator'),E=require('../world_model/encoding'),P=require('../world_model/planner');
const State=require('../js/gameState'),Logic=require('../js/gameLogic'),Baseline=require('../scripts/autorun');
const {Network}=require('../world_model/inference'),{tournament}=require('../world_model/tournament');
const root=path.join(__dirname,'..');
const modelDir=path.resolve(root,process.env.BRASS_TEST_MODELS||'world_model/models');
const reportDir=path.resolve(root,process.env.BRASS_TEST_REPORTS||'world_model/reports');
const load=name=>new Network(JSON.parse(fs.readFileSync(path.join(modelDir,name),'utf8')));
const world=load('world-model.json'),policy=load('neural-policy.json');
const V=require('../world_model/value_features');

test('original AI results unchanged for three recorded baseline seeds',()=>{
    const results=[1,9974,19947].map(seed=>Baseline.runGame({players:4,seed}).state.players.map(p=>p.vp));
    assert.deepEqual(results,[[64,52,48,60],[80,42,96,73],[21,79,43,9]]);
});
test('browser script scope imports coexist, including worker imports',()=>{
    const ctx=vm.createContext({console,structuredClone,performance,setTimeout});
    for(const p of ['js/gameData.js','js/gameState.js','js/gameLogic.js','scripts/autorun.js',
        'world_model/simulator.js','world_model/encoding.js','world_model/value_features.js','world_model/inference.js','world_model/planner.js','world_model/tournament.js'])vm.runInContext(fs.readFileSync(path.join(root,p),'utf8'),ctx,{filename:p});
    assert.equal(vm.runInContext('BrassSimulator.create(4,1).players.length',ctx),4);
});
test('snapshot clones full state instead of lossy GameState.toJSON',()=>{
    const s=Sim.create(4,99),c=Sim.clone(s);
    assert.ok(c instanceof State);assert.deepEqual(c.players[0].hand,s.players[0].hand);assert.deepEqual(c.drawDeck,s.drawDeck);
    c.players[0].hand.pop();assert.notEqual(c.players[0].hand.length,s.players[0].hand.length);
});
test('adapter delegates an action and turn advancement without mutating original',()=>{
    const s=Sim.create(4,117),before=Sim.snapshot(s),a=Sim.candidates(s).find(a=>a.action==='loan');
    const expected=Sim.clone(s),logic=new Logic(expected);
    logic.executeLoan(expected.currentPlayerId,a.cardIndex);Baseline.applyTurnResult(expected,expected.advanceTurn());
    const next=Sim.step(s,a).state;
    assert.deepEqual(Sim.snapshot(s),before);assert.deepEqual(Sim.snapshot(next),Sim.snapshot(expected));
    assert.throws(()=>Sim.step(s,{action:'build',cardIndex:99,target:{cityId:'invalid'}}));
    assert.deepEqual(Sim.snapshot(s),before);
});
test('same-seed games reproduce and all seven actions, era boundaries and terminal scoring encode',()=>{
    const seen=new Set(),events=new Set();
    for(const players of [2,3,4]){
        let s=Sim.create(players,311),turns=0;const rng=Sim.rng(991);
        assert.deepEqual(Sim.snapshot(s),Sim.snapshot(Sim.create(players,311)));
        while(!s.gameOver){
            const v=E.encodeState(s);assert.equal(v.length,689);assert.ok(v.every(Number.isFinite));
            const list=Sim.candidates(s),a=rng.pick(list);assert.ok(a);seen.add(a.action);
            const next=Sim.step(s,a),repeat=Sim.step(s,a);events.add(next.event.type);
            assert.deepEqual(E.encodeState(next.state),E.encodeState(repeat.state));
            assert.equal(E.encodeAction(a,s).length,147);s=next.state;assert.ok(++turns<1000);
        }
        assert.ok(s.players.every(p=>Number.isFinite(p.vp)));
    }
    assert.equal(seen.size,7);assert.ok(events.has('era')&&events.has('game'));
});
test('exported JavaScript inference matches the Python evaluation fixture',()=>{
    const fixture=JSON.parse(fs.readFileSync(path.join(reportDir,'inference-fixture.json'),'utf8'));
    const output=world.predict(fixture.input.slice(0,689),fixture.input.slice(689));
    const max=Math.max(...output.map((v,i)=>Math.abs(v-fixture.expected[i])));
    assert.ok(max<0.00002,`Maximum cross-language deviation ${max}`);
});
test('world planning never calls real step and both learned types require weights',()=>{
    const s=Sim.create(4,377),before=Sim.snapshot(s),step=Sim.step;
    Sim.step=()=>{throw Error('Ground truth was called from world planner');};
    try{
        const r=P.plan(s,{type:'world',world,depth:3,width:8});assert.ok(r.selected);assert.ok(r.branches.some(b=>b.steps.length>=2));
        const n=P.plan(s,{type:'neural',policy});assert.ok(n.selected);
        assert.throws(()=>P.plan(s,{type:'world'}),/权重/);assert.throws(()=>P.plan(s,{type:'neural'}),/权重/);
    }finally{Sim.step=step;}
    assert.deepEqual(Sim.snapshot(s),before);
});
test('learned rollout recursively consumes the previous prediction',()=>{
    const s=Sim.create(4,183),v=E.encodeState(s),a=E.encodeAction(Sim.candidates(s)[0],s);
    const imagined=world.imagine(v,[a,a,a]);
    assert.deepEqual(imagined[1],world.predict(imagined[0],a));assert.notDeepEqual(imagined[1],imagined[0]);
});
test('search hides actual future deck order and random state',()=>{
    const s=Sim.create(4,112),other=Sim.clone(s);other.drawDeck.reverse();other._rngState=999;
    assert.deepEqual(Sim.snapshot(P.informationLimitedCopy(s)),Sim.snapshot(P.informationLimitedCopy(other)));
});
test('model metadata rejects mismatched schema and malformed input',()=>{
    assert.throws(()=>new Network({...world.data,schemaVersion:'wrong'}));
    assert.throws(()=>world.predict([1],[2]));assert.throws(()=>world.forward(Array(836).fill(NaN)));
});
test('comparison rotates seats and counts only completed games',async()=>{
    const r=await tournament({games:4,depth:1,width:7,world,policy});
    assert.equal(r.completedGames,4);assert.equal(r.balancedSeats,true);
    assert.equal(r.rows.reduce((n,r)=>n+r.wins,0),4);
    for(const kind of r.config.types)assert.equal(new Set(r.games.map(g=>g.seats.indexOf(kind))).size,4);
});

test('repeated AI seats aggregate player appearances and conserve win credit',async()=>{
    const report=await tournament({games:4,seed:4411,depth:1,width:1,types:['heuristic','search','heuristic','search']});
    assert.equal(report.completedGames,4);
    assert.deepEqual(report.rows.map(r=>r.type),['heuristic','search']);
    assert.equal(report.rows.find(r=>r.type==='heuristic').games,8);
    assert.equal(report.rows.find(r=>r.type==='search').games,8);
    assert.equal(report.rows.reduce((sum,r)=>sum+r.wins,0),report.completedGames);
});

test('parallel two/three-player rotations match serial results with repeated entrants',async()=>{
    const {parallelTournament}=require('../world_model/benchmark_parallel');
    for(const types of [['heuristic','search'],['heuristic','search','heuristic']]){
        const config={types,games:types.length*2,seed:5177,depth:1,width:1};
        const serial=await tournament(config);
        const parallel=await parallelTournament({...config,models:modelDir,workers:2});
        assert.deepEqual(parallel.games,serial.games);
        assert.deepEqual(parallel.rows.map(({type,games,wins,vp})=>({type,games,wins,vp})),serial.rows.map(({type,games,wins,vp})=>({type,games,wins,vp})));
    }
});

test('parallel seed groups preserve the same games and aggregate scores as serial play',async()=>{
    const config={games:8,seed:55101,depth:1,width:7};
    const serial=await tournament({...config,world,policy});
    const {parallelTournament}=require('../world_model/benchmark_parallel');
    const parallel=await parallelTournament({...config,workers:2,models:modelDir});
    assert.deepEqual(parallel.games,serial.games);
    for(const row of serial.rows){
        const other=parallel.rows.find(r=>r.type===row.type);
        for(const key of ['games','wins','vp','moves','averageVP','winRate'])assert.equal(other[key],row[key]);
    }
});

test('structured projection reconstructs observable state and predicted turn order',()=>{
    let s=Sim.create(4,381);
    for(let i=0;i<7;i++)s=Sim.step(s,Sim.candidates(s).find(a=>a.action==='loan')||Sim.candidates(s)[0]).state;
    const v=E.encodeState(s),p=E.decodeState(v,s,{structured:true});
    assert.deepEqual(E.encodeState(p),v);
    for(let i=0;i<4;i++)v[E.indices[`order.${i}`]]=(4-i)/4;
    for(let i=0;i<4;i++)v[E.indices[`current.${i}`]]=+(i===2);
    v[E.indices.deckSize]=10/64;
    const predicted=E.decodeState(v,s,{structured:true});
    assert.deepEqual(predicted.turnOrder,[3,2,1,0]);assert.equal(predicted.currentPlayerId,2);assert.equal(predicted.drawDeck.length,10);
    assert.deepEqual(Sim.snapshot(s),Sim.snapshot(Sim.clone(s)));
});

test('learned change gates preserve untouched fields exactly and categorical control is one-hot',()=>{
    if(world.data.kind!=='gated-world-model')return;
    const data=structuredClone(world.data);
    data.gateLayers.at(-1).bias.fill(-1000);
    const gated=new Network(data),state=Sim.create(4,91),v=E.encodeState(state),a=E.encodeAction(Sim.candidates(state)[0],state);
    const output=gated.predict(v,a),controlled=new Set(data.controlSpec.heads.flatMap(h=>h.indices));
    for(let i=0;i<v.length;i++)if(!controlled.has(i))assert.equal(output[i],v[i]);
    assert.equal([0,1,2,3].reduce((n,i)=>n+output[E.indices[`current.${i}`]],0),1);
    assert.throws(()=>new Network({...data,gateThreshold:0}));
    assert.throws(()=>new Network({...data,gatePositiveWeight:[1]}));
});

test('terminal value features and inference match Python; known terminal VP is exact',()=>{
    if(!world.valueLayers)return;
    const f=JSON.parse(fs.readFileSync(path.join(modelDir,'value-fixture.json'),'utf8'));
    for(let player=0;player<4;player++){
        const features=V.features(f.state,player,world.data.valueModel.featureVersion);
        assert.equal(features.length,f.features[player].length);
        assert.ok(Math.max(...features.map((x,i)=>Math.abs(x-f.features[player][i])))<2e-6);
        assert.ok(Math.abs(world.estimateValue(f.state,player)/100-f.expected[player])<2e-5);
    }
    const v=f.state.slice();v[E.indices.gameOver]=1;
    [90,80,70,60].forEach((vp,p)=>v[E.indices[`p${p}.vp`]]=vp/E.fields[E.indices[`p${p}.vp`]].scale);
    for(let p=0;p<4;p++)v[E.indices[`p${p}.income`]]=0;
    const weight=world.data.valueModel.opponentWeight??.25;
    assert.equal(world.estimateValue(v,0),90-weight*80);assert.equal(world.estimateValue(v,1),80-weight*90);
    const ownData=structuredClone(world.data);ownData.valueModel.opponentWeight=0;
    const own=new Network(ownData);assert.equal(own.estimateValue(v,0),90);assert.equal(own.estimateValue(v,1),80);
    ownData.valueModel.incomeBonusWeight=1;
    v[E.indices['p0.income']]=.5;
    assert.equal(new Network(ownData).estimateValue(v,0),75);
    ownData.valueModel.opponentWeight=-1;assert.throws(()=>new Network(ownData),/objective/);
});

test('guided search uses learned value and engine transitions, without learned dynamics or hidden future',()=>{
    if(!world.valueLayers)return;
    const state=Sim.create(4,70217),before=Sim.snapshot(state),other=Sim.clone(state);
    other.drawDeck.reverse();other._rngState=713;
    const predict=world.predict,estimate=world.estimateValue,step=Sim.step;let values=0,steps=0;
    world.predict=()=>{throw Error('Guided search should use engine transitions');};
    world.estimateValue=function(...args){values++;return estimate.apply(this,args);};
    Sim.step=(...args)=>{steps++;return step(...args);};
    try{
        const a=P.plan(state,{type:'guided',world,depth:2,width:8}),b=P.plan(other,{type:'guided',world,depth:2,width:8});
        assert.ok(values>0&&steps>0);assert.deepEqual(a.branches,b.branches);assert.ok(a.selected);
        assert.throws(()=>P.plan(state,{type:'guided'}),/权重/);
    }finally{world.predict=predict;world.estimateValue=estimate;Sim.step=step;}
    assert.deepEqual(Sim.snapshot(state),before);
});

test('classic dense inference remains compatible with the archived Python fixture',()=>{
    const old=new Network(JSON.parse(fs.readFileSync(path.join(root,'world_model/experiments/expanded-10k/models/world-model.json'),'utf8')));
    const f=JSON.parse(fs.readFileSync(path.join(root,'world_model/experiments/expanded-10k/reports/inference-fixture.json'),'utf8'));
    assert.ok(Math.max(...old.predict(f.input.slice(0,689),f.input.slice(689)).map((x,i)=>Math.abs(x-f.expected[i])))<2e-5);
});

test('score-value features and inference match Python across different game phases',()=>{
    const file=path.join(reportDir,'value-cases.json');
    if(!fs.existsSync(file)||!world.valueLayers)return;
    const {cases}=JSON.parse(fs.readFileSync(file,'utf8'));
    assert.ok(cases.length>=6);
    for(const c of cases)for(let player=0;player<4;player++){
        const features=V.features(c.state,player,world.data.valueModel.featureVersion);
        assert.equal(features.length,c.features[player].length);
        assert.ok(Math.max(...features.map((x,i)=>Math.abs(x-c.features[player][i])))<2e-6);
        assert.ok(Math.abs(world.estimateValue(c.state,player)/100-c.expected[player])<2e-5);
    }
});

test('geographic features count actual adjacent industry icons, including opponents and farms',()=>{
    const D=require('../js/gameData'),s=Sim.create(4,3);
    s.boardLinks={'birmingham-oxford':{playerId:0,type:'rail'},'cannock-northern':{playerId:0,type:'rail'}};
    s.boardIndustries={birmingham_0:{playerId:1,type:'cottonMill',industryType:'cottonMill',flipped:true,resourceCubes:0,tileData:D.INDUSTRY_DATA.cottonMill[1]}};
    s.breweryFarmTiles={northern:{playerId:0,type:'brewery',industryType:'brewery',flipped:true,resourceCubes:0,tileData:D.INDUSTRY_DATA.brewery[1]}};
    const v=E.encodeState(s),f=V.features(v,0,'brass-value-v2');
    assert.equal(f.length,125);assert.deepEqual(f.slice(0,109),V.features(v,0));
    const scored=Sim.clone(s).calculateEraScore()[0];
    assert.ok(Math.abs(f[109]*100-scored.linkVP)<1e-6);
    const other=Sim.clone(s);other.boardIndustries.birmingham_0.flipped=false;
    const changed=V.features(E.encodeState(other),0,'brass-value-v2');
    assert.ok(changed[109]<f[109]);assert.ok(changed[110]>f[110]);
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(root,'world_model/value_topology.json'),'utf8')).topology,V.topology);
});

test('versioned continuation shortlist can consider a high-value sale with builds available',()=>{
    const original=Sim.candidates;
    Sim.candidates=()=>['build','network','develop','sell','loan','pass'].map((action,i)=>({action,cardIndex:i,score:action==='sell'?100:10-i,target:{}}));
    try{
        assert.deepEqual(P.shortlist({},3).map(a=>a.action),['build','network','develop']);
        assert.equal(P.shortlist({},3,'score-diverse-v2')[0].action,'sell');
    }finally{Sim.candidates=original;}
});

test('own-diverse continuation is accepted as a separate planning mode',()=>{
    const state=Sim.create(4,3319),model=new Network(structuredClone(world.data));
    model.data.planningContinuation='own-diverse-pool-12';
    const result=P.plan(state,{type:'guided',world:model,depth:2,width:4});
    assert.equal(result.branches.length,4);
    assert.ok(result.branches.every(branch=>branch.steps.length>0));
});

test('broader learned candidate selection stays legal and never uses engine transitions for world planning',()=>{
    const data=structuredClone(world.data);data.planningCandidates='learned-pool-v1';data.planningShortlist='score-diverse-v2';
    const model=new Network(data),state=Sim.create(4,3318),valid=new Set(Sim.candidates(state).map(Sim.key)),step=Sim.step;
    Sim.step=()=>{throw Error('World pre-ranking called the engine');};
    try{
        const result=P.plan(state,{type:'world',world:model,depth:2,width:8});
        assert.ok(result.candidatePoolSize>8&&result.candidatePoolSize<=32);
        assert.equal(result.branches.length,8);assert.ok(valid.has(Sim.key(result.selected)));
    }finally{Sim.step=step;}
});
