// Observed outcomes, never causal labels inferred from a low terminal score.
const Intent=require('./human_intent');
function settlement(state){
 const view=Intent.fromState(state);
 return view.players.map((_,player)=>{
  const tiles=view.tiles.filter(t=>t.owner===player),unsold=tiles.filter(t=>!t.flipped&&['cottonMill','manufacturer','pottery'].includes(t.type));
  return {player,era:state.era,unsoldIndustries:unsold.length,unsoldVP:unsold.reduce((n,t)=>n+t.vp,0),
   unflippedResources:tiles.filter(t=>!t.flipped&&!['cottonMill','manufacturer','pottery'].includes(t.type)).length};
 });
}
function railConstraints(state,actor,legalDoubleCount){
 if(state.era!=='rail'||state.round!==1)return null;
 const p=state.players[actor],view=Intent.fromState(state),beer=view.tiles.filter(t=>t.type==='brewery').reduce((n,t)=>n+t.cubes,0);
 return {money:p.money,cashBelowBaseDoubleCost:p.money<15,linksBelowTwo:p.linksRemaining.rail<2,
  noBoardBeer:beer===0,legalDoubleCount,
  unresolvedConnectivityCoalOrResourceCost:legalDoubleCount===0&&p.money>=15&&p.linksRemaining.rail>=2&&beer>0};
}
function unconvertedDevelopment(trace){
 const developed=trace.filter(r=>r.rawAction.action==='develop').flatMap(r=>r.developed.map(d=>({...d,seq:r.seq})));
 return developed.filter(d=>!trace.some(r=>r.seq>d.seq&&r.rawAction.action==='build'&&r.rawAction.target.industryType===d.type&&r.builtLevel>d.level));
}
module.exports={settlement,railConstraints,unconvertedDevelopment};
