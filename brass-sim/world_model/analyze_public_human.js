const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),zlib=require('node:zlib');
const {analyze}=require('./analyze_teacher_league'),{average,grouped,interval}=require('./human_analysis');
function analyzeRound(root){
 const read=n=>{const file=path.join(root,n+'.json');return JSON.parse(fs.existsSync(file)?fs.readFileSync(file):zlib.gunzipSync(fs.readFileSync(file+'.gz')));};
 const protocol=read('protocol'),mixed=read('mixed'),baseline=read('baseline'),selfplay=read('selfplay'),world=read('world');
 for(const [name,report] of Object.entries({mixed,baseline,selfplay,world})){
  if(report.config.seed!==(name==='world'?protocol.worldSeed:protocol.seed)||report.modelSHA256!==protocol.incumbentSHA256||
   report.intentPriorSHA256!==protocol.priorSHA256||report.config.intentWeight!==protocol.intentWeight)throw Error('Final protocol mismatch');
  for(const [file,hash] of Object.entries(report.sourceSHA256))if(protocol.sourceSHA256['world_model/'+file]!==hash&&protocol.sourceSHA256[path.posix.normalize('world_model/'+file)]!==hash)throw Error('Unfrozen source');
 }
 if(world.completedGames!==40||world.independentSeedGroups!==10||world.candidateModelSHA256!==protocol.worldSHA256)throw Error('World protocol mismatch');
 const result=analyze(mixed,baseline,selfplay,{candidate:'intent',incumbent:'trainedcards'});
 result.intentPriorSHA256=protocol.priorSHA256;result.intentWeight=protocol.intentWeight;
 result.notes=['Current teacher-trained-v2 + retained cards is the incumbent.',
  '372 native human intent rows, 248 for training; foreign scores are never JS value labels.',
  'Bootstrap resamples whole seed groups; identical selfplay rotations remain correlated.',
  'All hands public in playing-strength tests; no tabletop expert or hidden-information certification.'];
 const old=grouped(world,'oldworld',s=>s.vp),candidate=grouped(world,'world',s=>s.vp),delta=candidate.map((v,i)=>v-old[i]);
 result.world={groups:delta.length,incumbentMeanVP:average(old),candidateMeanVP:average(candidate),
  withinMixedGroupGain:average(delta),withinMixedGroupGain95CI:interval(delta),
  limitation:'One candidate and one incumbent share each mixed game. This is a seat-balanced matchup, not paired counterfactual selfplay.'};
 function diagnostics(report,profile){
  const entries=report.games.flatMap(g=>g.scores.filter(s=>s.profile===profile));
  return {appearances:entries.length,
   meanUnsoldAtCanalSettlement:average(entries.map(s=>s.settlements.find(r=>r.era==='canal').unsoldIndustries)),
   meanUnsoldAtRailSettlement:average(entries.map(s=>s.settlements.find(r=>r.era==='rail').unsoldIndustries)),
   meanUnusedDevelopedTiles:average(entries.map(s=>s.developmentWithoutLaterBuild.length)),
   railOpeningConstraints:entries.flatMap(s=>s.trace.filter(t=>t.railOpening).map(t=>t.railOpening)).reduce((o,r)=>{
    o.decisions++;for(const k of ['cashBelowBaseDoubleCost','linksBelowTwo','noBoardBeer','unresolvedConnectivityCoalOrResourceCost'])o[k]=(o[k]||0)+Number(r[k]);
    o.legalDoubleAvailable=(o.legalDoubleAvailable||0)+Number(r.legalDoubleCount>0);return o;
   },{decisions:0}),
   limitation:'Constraints can overlap; availability and later builds do not establish a causal explanation for a final score.'};
 }
 result.diagnostics={incumbent:diagnostics(baseline,'trainedcards'),intentSelfplay:diagnostics(selfplay,'intent'),world:diagnostics(world,'world')};
 result.defaultPromoted=false;
 fs.writeFileSync(path.join(root,'summary.json'),JSON.stringify(result,null,2));return result;
}
if(require.main===module)console.log(JSON.stringify(analyzeRound(process.argv[2]),null,2));
module.exports={analyzeRound};
