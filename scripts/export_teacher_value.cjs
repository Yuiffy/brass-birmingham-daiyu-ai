// Run from repo root. Freeze only the value head used by guided search.
process.env.BRASS_RULES = 'economy-v2';
const fs = require('node:fs'), crypto = require('node:crypto');
const source = 'brass-sim/world_model/experiments/human-teacher-20261009/models/world-model.json';
const raw = fs.readFileSync(source), data = JSON.parse(raw);
const Sim = require('../brass-sim/world_model/simulator');
const E = require('../brass-sim/world_model/encoding');
const V = require('../brass-sim/world_model/value_features');
const {Network} = require('../brass-sim/world_model/inference');
const {plan} = require('../brass-sim/world_model/planner');
const world = new Network(data);
fs.writeFileSync('src/game/models/teacher-value.json', JSON.stringify({source, sourceSha256: crypto.createHash('sha256').update(raw).digest('hex'), ...data.valueModel}) + '\n');
const fixture = [];
let state = Sim.create(4, 9197);
for (let step = 0; step < 32 && !state.gameOver; step++) {
  if (step % 8 === 0) {
    const vector = E.encodeState(state);
    for (let p = 0; p < 4; p++) fixture.push({input: V.features(vector, p, 'brass-value-v2'), expected: world.estimateValue(vector, p)});
  }
  state = Sim.step(state, plan(state, {type:'guided', world, depth:2, width:8, strategy:'human-card-v2'}).selected).state;
}
fs.writeFileSync('src/game/models/teacher-value-fixture.json', JSON.stringify(fixture) + '\n');
// Independent semantic projection oracle: normalized schema fields -> JS features.
// Matches the deliberately synthetic Rust fixture, not a foreign game replay.
const rawFields = {era:0, round:3, players:2, deckSize:12, actionsThisTurn:1, actionsPerTurn:2, gameOver:0, coalMarket:7, ironMarket:5, 'current.0':1};
for (const p of [0,1]) for (const [k,v] of Object.entries({money:p===0?42:17,income:p===0?35:10,vp:p===0?13:0,handSize:p===0?6:8,spent:p===0?8:0,canal:13,rail:14,wildLocation:0,wildIndustry:0})) rawFields[`p${p}.${k}`]=v;
rawFields['p0.used.brewery']=2;
rawFields['p0.used.cottonMill']=4;
rawFields['p1.used.cottonMill']=1;
for (const [slot,owner,type,level,flipped,cubes,vp,linkVP] of [
  ['stafford_0',1,'brewery',2,0,2,5,2],
  ['tamworth_0',1,'cottonMill',2,1,0,5,2],
  ['stokeOnTrent_0',2,'cottonMill',1,1,0,5,1]
]) for (const [k,v] of Object.entries({owner,type:E.schema.types.indexOf(type)+1,level,flipped,cubes,vp,linkVP})) rawFields[`slot.${slot}.${k}`]=v;
const link = cities => E.schema.links.find(id => {
  const c=require('../brass-sim/js/gameData').CONNECTIONS.find(c=>c.id===id);
  return cities.every(x=>c.cities.includes(x));
});
rawFields[`link.${link(['stafford','stone'])}.owner`]=1;
rawFields[`link.${link(['warrington','stokeOnTrent'])}.owner`]=2;
const vector=E.fields.map(f=>(rawFields[f.name]||0)/f.scale);
fs.writeFileSync('src/game/models/teacher-native-feature-fixture.json',JSON.stringify(V.features(vector,0,'brass-value-v2'))+'\n');
console.log(`Exported frozen teacher value and ${fixture.length} JS inference fixtures.`);
