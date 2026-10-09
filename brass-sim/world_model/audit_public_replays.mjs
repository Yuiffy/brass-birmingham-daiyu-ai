// Audit public native-engine replays; export human public features/action intent only.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url),Intent=require('./human_intent.js');
const typeMap={brewery:'brewery',coal:'coalMine',cotton:'cottonMill',iron:'ironWorks',manufacturer:'manufacturer',pottery:'pottery'};
export function publicView(s,engine){
 const tiles=Object.values(s.board.slots).flat().filter(Boolean).map(t=>({owner:t.player,type:typeMap[t.tile.industry],flipped:t.flipped,cubes:t.resources,vp:t.tile.vp}));
 return {era:s.era,round:s.round,deckSize:s.deck.length,actionsThisTurn:s.actionsThisTurn,coalMarket:s.coalMarket,ironMarket:s.ironMarket,
  players:s.players.map(p=>({money:p.money,income:engine.incomeLevelAt(p.incomeSpace),vp:p.vp,spent:p.spentThisRound,handSize:p.hand.length,
   wildLocation:p.hand.some(c=>c.kind==='wild-location'),wildIndustry:p.hand.some(c=>c.kind==='wild-industry'),
   nextLevels:Object.fromEntries(Object.entries(typeMap).map(([source,dest])=>[dest,p.tiles.find(t=>t.industry===source)?.level||0]))})),
  tiles,links:s.board.links.map(l=>({owner:l.player,era:l.era})),
  demands:['cotton','manufacturer','pottery'].filter(t=>Object.values(s.merchants).some(m=>m.tiles.includes(t)||m.tiles.includes('any'))).map(t=>typeMap[t])};
}
export function actionIntents(a){
 if(a.type==='build')return ['build:'+typeMap[a.industry]];
 if(a.type==='develop')return [...new Set(a.removals.map(t=>'develop:'+typeMap[t]))];
 if(a.type==='network')return ['network:'+a.links.length];
 return [a.type];
}
export function auditRecord(record,engine,{source,split,sha256}){
 if(![2,3,4].includes(record.playerCount)||!Number.isInteger(record.seed)||!Array.isArray(record.seats)||record.seats.length!==record.playerCount||!Array.isArray(record.actions)||!record.actions.length)throw Error('Invalid replay metadata');
 if(record.seats.some(s=>!Number.isInteger(s.seat)||s.seat<0||s.seat>=record.playerCount||
  s.isAI!==undefined&&s.is_ai!==undefined&&(typeof s.isAI!=='boolean'||![0,1].includes(s.is_ai)||s.isAI!==(s.is_ai===1))))throw Error('Contradictory or invalid seat provenance');
 const humans=record.seats.filter(s=>s.isAI===false||s.is_ai===0).map(s=>s.seat);
 const flags=record.seats.map(s=>s.isAI??(s.is_ai===0?false:s.is_ai===1?true:undefined));
 if(flags.some(x=>typeof x!=='boolean')||new Set(record.seats.map(s=>s.seat)).size!==record.playerCount)throw Error('Ambiguous player provenance');
 let state=engine.newGame(record.playerCount,record.seed),rows=[],canalScores=null;
 for(let i=0;i<record.actions.length;i++){
  const row=record.actions[i],actor=state.turnOrder[state.currentPlayerIdx];
  if(state.phase==='game-over'||row.seq!==i||row.player!==actor)throw Error('Replay turn/order mismatch at '+i);
  const next=engine.applyAction(state,row.action); // Full native legality validation; no assumeLegal.
  if(humans.includes(actor)){
   const intents=actionIntents(row.action);if(intents.some(t=>!Intent.classes.includes(t)))throw Error('Unsupported human action intent');
   rows.push({source,gameSHA256:sha256,seed:record.seed,seq:i,actor,split,features:Intent.features(publicView(state,engine),actor),intents,
    provenance:'public-anonymized-human-vs-AI-replay',valueLabelUsableForJS:false});
  }
  if(state.era==='canal'&&next.era==='rail')canalScores=next.players.map(p=>p.vp);
  state=next;
 }
 if(state.phase!=='game-over')throw Error('Incomplete replay');
 return {rows,report:{source,sha256,seed:record.seed,players:record.playerCount,humanSeats:humans,actions:record.actions.length,
  humanActions:rows.length,split,terminal:true,nativeLegalityAudited:true,canalScores,finalScores:state.players.map(p=>p.vp),
  expertStatusVerified:false,jsValueLabels:false}};
}
export async function audit({referenceRoot,enginePath,engineRevision,out}){
 if(!/^[0-9a-f]{40}$/.test(engineRevision||''))throw Error('Pin the exact reference engine commit');
 if(execFileSync('git',['-C',referenceRoot,'rev-parse','HEAD'],{encoding:'utf8'}).trim()!==engineRevision)throw Error('Reference engine revision mismatch');
 if(execFileSync('git',['-C',referenceRoot,'status','--porcelain','--untracked-files=no'],{encoding:'utf8'}).trim())throw Error('Reference tracked sources were modified');
 const engine=await import(pathToFileURL(path.resolve(enginePath)));
 const dirs=['packages/server/replays','packages/server/test/fixtures/games'];
 const files=dirs.flatMap(d=>fs.readdirSync(path.join(referenceRoot,d)).filter(n=>n.endsWith('.json')).map(n=>path.join(d,n))).sort();
 if(fs.existsSync(out)&&fs.readdirSync(out).length)throw Error('Choose fresh audit output');
 fs.mkdirSync(out,{recursive:true});let rows=[],reports=[],seen=new Set(),human4=0;
 for(const file of files){
  const bytes=fs.readFileSync(path.join(referenceRoot,file)),record=JSON.parse(bytes),sha256=crypto.createHash('sha256').update(bytes).digest('hex');
  if(seen.has(record.seed))throw Error('Duplicate replay seed');seen.add(record.seed);
  const human=record.seats.some(s=>s.isAI===false||s.is_ai===0);
  const split=human&&record.playerCount===4?(['train','train','train','train','validation','test'][human4++]||'test'):'diagnostic';
  const source='https://github.com/Quasrain-Coder/BrassBirmingham/blob/'+engineRevision+'/'+file.replaceAll(path.sep,'/');
  try{const result=auditRecord(record,engine,{source,split,sha256});reports.push(result.report);if(split!=='diagnostic')rows.push(...result.rows);}
  catch(error){throw Error('Replay audit failed for '+file+': '+error.message);}
 }
 if(human4!==6||rows.length!==372||reports.filter(r=>r.players===4&&r.humanActions>0).some(r=>r.humanActions!==62))throw Error('Pinned corpus integrity mismatch');
 fs.writeFileSync(path.join(out,'human-intents.jsonl'),rows.map(r=>JSON.stringify(r)).join('\n')+'\n');
 const bundleRoot=path.dirname(path.resolve(enginePath));
 const hashes={};
 function hashBundle(dir){for(const entry of fs.readdirSync(dir,{withFileTypes:true})){
  const full=path.join(dir,entry.name);if(entry.isDirectory())hashBundle(full);
  else if(entry.name.endsWith('.js'))hashes[path.relative(bundleRoot,full).replaceAll(path.sep,'/')]=crypto.createHash('sha256').update(fs.readFileSync(full)).digest('hex');
 }}
 hashBundle(bundleRoot);
 fs.writeFileSync(path.join(out,'audit.json'),JSON.stringify({engineRevision,humanIntentRows:rows.length,featureVersion:'public-intent-v1',
  engineBundleSHA256:crypto.createHash('sha256').update(fs.readFileSync(enginePath)).digest('hex'),
  engineSourceTree:execFileSync('git',['-C',referenceRoot,'rev-parse',engineRevision+':packages/engine'],{encoding:'utf8'}).trim(),
  compiledModuleHashes:hashes,compileCommand:'tsc -p packages/engine/tsconfig.json --rootDir packages/engine --noEmit false --outDir audited-engine',
  featureNames:Intent.featureNames,classes:Intent.classes,fullNativeReplayRequired:true,expertStatusVerified:false,jsValueTrainingRows:0,reports},null,2));
 console.log(JSON.stringify({humanIntentRows:rows.length,reports},null,2));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].slice(2)]=process.argv[i+1];await audit(args);
}
