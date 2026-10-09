const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..');
const registry=require('../world_model/runtime_models'),Sim=require('../world_model/simulator');
const E=require('../world_model/encoding'),P=require('../world_model/planner'),{Network}=require('../world_model/inference');
const read=file=>JSON.parse(fs.readFileSync(path.join(root,file),'utf8'));
function gameContext(fail=()=>false){
    const nodes=new Map(),requests=[],plans=[];
    const get=id=>{if(!nodes.has(id))nodes.set(id,{checked:true,value:id==='ai-depth'?'1':'',textContent:'',innerHTML:''});return nodes.get(id);};
    const context=vm.createContext({console,BrassSimulator:Sim,BrassEncoding:E,BrassRuntimeModels:registry,
        BrassPlanner:{...P,plan:(state,config)=>{plans.push(config);return P.plan(state,config);}},BrassWorldModel:{Network},
        document:{getElementById:get,querySelectorAll:()=>[]},
        fetch:async url=>{requests.push(url);return fail(url)?{ok:false,status:404}:{ok:true,json:async()=>read(url)};}});
    vm.runInContext(fs.readFileSync(path.join(root,'js/aiController.js'),'utf8'),context);
    return {api:context.BrassAI,requests,plans,nodes};
}

test('runtime routing uses the evaluated 2P/3P artifacts and preserves the 4P checkpoint',()=>{
    const report=read('world_model/reports/small-player-dynamics.json');
    for(const players of [2,3,4]){
        const file=path.join(root,'world_model',registry.directoryForPlayers(players),'world-model.json');
        const actual=crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
        const expected=players===4?report.protocol.baselineSHA256:report.runs.find(r=>r.players===players).worldModelSHA256;
        assert.equal(actual,expected);
    }
    assert.throws(()=>registry.directoryForPlayers(1));assert.throws(()=>registry.directoryForPlayers(5));
});

test('game moves and inspection use the selected player model; loading is shared and cached',async()=>{
    const {api,requests,plans}=gameContext();
    assert.equal(api.readyFor(['world'],2),false);
    await Promise.all([api.load(),api.load()]);assert.equal(requests.length,4);
    await api.load();assert.equal(requests.length,4);
    for(const players of [2,3,4]){
        assert.equal(api.readyFor(['guided','world'],players),true);
        const world=api.getWorld(players),controller=Object.create(api.Controller.prototype);
        controller.state=Sim.create(players,7721);controller.kinds=Array(players).fill('world');
        controller.ui={showToast:message=>assert.fail(message)};controller.refresh=()=>{};
        controller.apply=action=>assert.ok(Sim.candidates(controller.state).some(a=>Sim.key(a)===Sim.key(action)));
        controller.move();assert.equal(plans.at(-1).world,world);
        let predictions=0;const predict=world.predict.bind(world);
        world.predict=(...args)=>{predictions++;return predict(...args);};
        controller.inspect();
        assert.equal(plans.at(-1).world,world);
        assert.ok(controller.preview.branches.length);assert.ok(predictions>0);
        world.predict=predict;
    }
    delete api.worldByPlayers[2];
    assert.equal(api.getWorld(2),null);assert.equal(api.readyFor(['world'],2),false);
    assert.equal(api.readyFor(['world'],4),true);
});

test('a failed specialized model load can retry and never silently falls back to 4P',async()=>{
    let failed=true;
    const {api}=gameContext(url=>failed&&url.includes('dynamics-3p-active'));
    assert.equal(await api.load(),false);assert.match(api.error,/HTTP 404/);
    assert.equal(api.getWorld(3),null);assert.equal(api.readyFor(['guided'],3),false);
    failed=false;assert.equal(await api.load(),true);assert.equal(api.error,null);
    assert.equal(api.readyFor(['world'],3),true);
});

test('arena worker resolves current models by seat count and retains explicit historical versions',async()=>{
    const messages=[],requests=[];let selected;
    const context=vm.createContext({console,performance,structuredClone,setTimeout,
        fetch:async url=>{requests.push(url);return {ok:true,json:async()=>read(`world_model/${url}`)};},
        postMessage:m=>messages.push(m)});
    context.importScripts=(...files)=>files.forEach(file=>vm.runInContext(fs.readFileSync(path.resolve(root,'world_model',file),'utf8'),context));
    vm.runInContext(fs.readFileSync(path.join(root,'world_model/arena-worker.js'),'utf8'),context);
    context.BrassTournament.tournament=async config=>{selected=config;return {completedGames:config.games};};
    for(const players of [2,3,4]){
        await context.onmessage({data:{type:'run',config:{currentModels:true,models:'models',types:Array.from({length:players},(_,i)=>i%2?'world':'guided'),games:players}}});
        const directory=registry.directoryForPlayers(players);
        assert.equal(messages.at(-1).type,'result');assert.equal(messages.at(-1).result.runtimeModels.directory,directory);
        assert.equal(selected.world.data.epoch,read(`world_model/${directory}/world-model.json`).epoch);
        assert.ok(requests.includes(`${directory}/world-model.json`));
    }
    await context.onmessage({data:{type:'run',config:{models:'experiments/score-v2/models',types:['heuristic','search','neural','world'],games:4}}});
    assert.equal(messages.at(-1).type,'result');
    assert.equal(messages.at(-1).result.runtimeModels.directory,'experiments/score-v2/models');
});

test('arena shows reports for the selected player count and dispatches the same lineup',async()=>{
    const nodes=new Map(),sent=[];
    const get=id=>{if(!nodes.has(id))nodes.set(id,{value:({'arena-version':'current','arena-lineup':'strong','arena-players':'4','arena-games':'12','arena-seed':'1','arena-depth':'2'})[id]||'',
        innerHTML:'',textContent:'',querySelector:selector=>get(id+selector)});return nodes.get(id);};
    const context=vm.createContext({console,setTimeout,document:{getElementById:get},
        fetch:async url=>({ok:true,json:async()=>read(url)}),Worker:class {postMessage(message){sent.push(message);}terminate(){}}});
    vm.runInContext(fs.readFileSync(path.join(root,'world_model/arena.js'),'utf8'),context);
    for(const players of [2,3,4]){
        get('arena-players').value=String(players);await vm.runInContext('loadVersion()',context);
        assert.equal(vm.runInContext('lastReport.config.types.length',context),players);
        get('arena-start').onclick();
        assert.equal(sent.at(-1).config.types.length,players);assert.equal(sent.at(-1).config.currentModels,true);
        assert.ok(sent.at(-1).config.types.every(t=>t==='guided'||t==='world'));
    }
    get('arena-version').value='pre-score';await vm.runInContext('loadVersion()',context);
    assert.equal(get('arena-players').value,'4');assert.equal(get('arena-players').disabled,true);
    assert.match(get('arena-status').textContent,/还没有保存/);
});
