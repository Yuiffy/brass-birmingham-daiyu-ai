const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
process.env.BRASS_RULES='economy-v2';
const root=path.resolve(__dirname,'..'),Sim=require('../world_model/simulator'),E=require('../world_model/encoding'),P=require('../world_model/planner'),{Network}=require('../world_model/inference');
const read=f=>JSON.parse(fs.readFileSync(path.join(root,f)));
test('teacher weights load once, reject unsupported seats and route moves/inspection consistently',async()=>{
 const nodes=new Map(),plans=[],requests=[];
 const get=id=>{if(!nodes.has(id))nodes.set(id,{checked:true,value:id==='guided-strategy'?'teacher-trained-v2':'1',textContent:'',innerHTML:''});return nodes.get(id);};
 const ctx=vm.createContext({console,BrassSimulator:Sim,BrassEncoding:E,BrassRuntimeModels:require('../world_model/runtime_models'),BrassWorldModel:{Network},
  BrassPlanner:{...P,plan:(s,c)=>{plans.push(c);return P.plan(s,c);}},document:{getElementById:get,querySelectorAll:()=>[]},
  fetch:async url=>{requests.push(url);return {ok:true,json:async()=>read(url)};}});
 vm.runInContext(fs.readFileSync(path.join(root,'js/aiController.js'),'utf8'),ctx);const api=ctx.BrassAI;
 await api.load();await Promise.all([api.loadTeacher(),api.loadTeacher()]);await api.loadTeacher();
 assert.equal(requests.filter(u=>u.includes('human-teacher')).length,1);
 for(const count of [2,3]){assert.equal(api.getWorld(count,'teacher-trained-v2'),null);assert.equal(api.readyFor(['guided'],count),false);assert.equal(api.readyFor(['human'],count),true);}
 assert.equal(api.readyFor(['guided','world'],4),true);
 for(const kind of ['guided','world']){
  const c=Object.create(api.Controller.prototype);c.state=Sim.create(4,4423);c.kinds=Array(4).fill(kind);
  c.profile='teacher-trained-v2';c.strategy='human-card-v2';c.refresh=()=>{};c.ui={showToast:m=>assert.fail(m)};c.apply=a=>assert.doesNotThrow(()=>Sim.step(c.state,a));
  c.move();assert.equal(plans.at(-1).world,api.teacherWorld);
  if(kind==='guided')assert.equal(plans.at(-1).strategy,'human-card-v2');
  c.inspect();assert.equal(plans.at(-1).world,api.teacherWorld);assert.ok(c.preview.branches.length);
 }
});
test('exported teacher value predictions agree with the Python fixture and frozen protocol',()=>{
 const file='world_model/experiments/human-teacher-20261009/models/world-model.json';
 const net=new Network(read(file)),fixture=read('world_model/experiments/human-teacher-20261009/models/value-fixture.json');
 const protocol=read('world_model/experiments/human-teacher-20261009/protocol.json');
 assert.equal(require('node:crypto').createHash('sha256').update(fs.readFileSync(path.join(root,file))).digest('hex'),protocol.candidateSHA256);
 assert.equal(net.data.actionEncoding,'resource-network-v2');
 for(let p=0;p<4;p++)assert.ok(Math.abs(net.estimateValue(fixture.state,p)/100-fixture.expected[p])<2e-5);
});
