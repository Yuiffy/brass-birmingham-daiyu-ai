const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
process.env.BRASS_RULES='economy-v2';
const root=path.resolve(__dirname,'..'),Runtime=require('../world_model/runtime_models'),E=require('../world_model/encoding'),Sim=require('../world_model/simulator'),P=require('../world_model/planner'),{Network}=require('../world_model/inference');
const read=f=>JSON.parse(fs.readFileSync(path.join(root,f),'utf8'));
test('calibrated runtime adopts only independently accepted world and guided heads',()=>{
 const summary=read('world_model/experiments/economy-20260920/summary.json');assert.equal(summary.complete,true);
 for(const run of summary.runs){
  assert.equal('world_model/'+Runtime.directoryForPlayers(run.players),run.runtimeWorld);
  assert.equal('world_model/'+Runtime.guidedDirectoryForPlayers(run.players),run.runtimeGuided);
  for(const [type,directory] of [['world',run.runtimeWorld],['guided',run.runtimeGuided]]){
   const net=new Network(read(directory+'/world-model.json'));assert.equal(net.data.schemaVersion,E.schema.version);
   const state=Sim.create(run.players,1990000021),before=Sim.snapshot(state);
   const action=P.plan(state,{type,world:net,depth:1,width:2}).selected;assert.ok(action);assert.deepEqual(Sim.snapshot(state),before);assert.doesNotThrow(()=>Sim.step(state,action));
  }
  const saved=read(`world_model/experiments/economy-20260920/${run.players}p/runtime-tournament.json`);
  assert.equal(saved.config.rulesVersion,'economy-v2');assert.equal(saved.config.types.length,run.players);
  assert.ok(saved.games.every(g=>g.scores.every(p=>p.incomeBonus===0&&p.vpWithoutIncomeBonus===p.vp)));
 }
});
test('calibrated game loads correct policy schema and routes guided moves independently',async()=>{
 const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,{checked:true,value:'1',textContent:'',innerHTML:''});return nodes.get(id);},plans=[];
 const ctx=vm.createContext({console,BRASS_RULES:'economy-v2',BrassSimulator:Sim,BrassEncoding:E,BrassRuntimeModels:Runtime,BrassWorldModel:{Network},BrassPlanner:{...P,plan:(s,c)=>{plans.push(c);return P.plan(s,c);}},document:{getElementById:get,querySelectorAll:()=>[]},fetch:async url=>({ok:true,json:async()=>read(url)})});
 vm.runInContext(fs.readFileSync(path.join(root,'js/aiController.js'),'utf8'),ctx);const api=ctx.BrassAI;
 assert.equal(await api.load(),true);assert.equal(api.policy.data.schemaVersion,E.schema.version);
 for(const players of [2,3,4])assert.deepEqual(api.getPolicy(players).data,read('world_model/'+Runtime.directoryForPlayers(players)+'/neural-policy.json'));
 for(const players of [2,3,4])for(const type of ['guided','world']){
  assert.equal(api.readyFor([type],players),true);const controller=Object.create(api.Controller.prototype);
  controller.state=Sim.create(players,2);controller.kinds=Array(players).fill(type);controller.refresh=()=>{};controller.ui={showToast:message=>assert.fail(message)};controller.apply=a=>assert.ok(a);
  controller.move();assert.equal(plans.at(-1).world,type==='guided'?api.getGuided(players):api.getWorld(players));
 }
});
test('calibrated worker uses the same accepted per-type models as the game',async()=>{
 const messages=[];let selected;
 const ctx=vm.createContext({console,performance,structuredClone,setTimeout,URL,location:{href:'http://localhost/world_model/arena-worker.js?rules=economy-v2'},fetch:async url=>({ok:true,json:async()=>read('world_model/'+url)}),postMessage:m=>messages.push(m)});
 ctx.importScripts=(...files)=>files.forEach(f=>vm.runInContext(fs.readFileSync(path.resolve(root,'world_model',f),'utf8'),ctx));
 vm.runInContext(fs.readFileSync(path.join(root,'world_model/arena-worker.js'),'utf8'),ctx);
 ctx.BrassTournament.tournament=async config=>{selected=config;return {completedGames:config.games};};
 for(const p of [2,3,4]){
  await ctx.onmessage({data:{type:'run',config:{currentModels:true,games:p,types:Array.from({length:p},(_,i)=>i%2?'world':'guided')}}});
  assert.equal(messages.at(-1).type,'result');assert.equal(selected.world.data.schemaVersion,'brass-wm-economy-v2');
  assert.equal(messages.at(-1).result.runtimeModels.directory,Runtime.directoryForPlayers(p));assert.equal(messages.at(-1).result.runtimeModels.guidedDirectory,Runtime.guidedDirectoryForPlayers(p));
 }
});
