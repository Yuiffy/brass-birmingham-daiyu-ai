// Public economic/board features for transferable action intent, never terminal labels.
(function(root){
 'use strict';
 const node=typeof module!=='undefined'&&module.exports;
 const types=['brewery','coalMine','cottonMill','ironWorks','manufacturer','pottery'];
 const classes=types.map(t=>'build:'+t).concat(types.map(t=>'develop:'+t),['network:1','network:2','loan','sell','scout','pass']);
 const featureNames=['era','round','players','deckSize','actionsThisTurn','ownHandSize','coalMarket','ironMarket',
  ...['own','rivals'].flatMap(p=>['money','income','vp','spent'].map(k=>p+'.'+k)),
  ...['own','rivals'].flatMap(p=>[...types.map(t=>p+'.built.'+t),...types.map(t=>p+'.unflipped.'+t),
   ...['coal','iron','beer','canalLinks','railLinks','unsoldVP'].map(k=>p+'.'+k)]),
  ...types.map(t=>'nextLevel.'+t),'merchant.cottonMill','merchant.manufacturer','merchant.pottery','ownWildLocation','ownWildIndustry'];
 function features(view,actor){
  if(!Number.isInteger(actor)||actor<0||actor>=view.players.length)throw Error('Invalid human-intent perspective');
  const own=view.players[actor],other=view.players.filter((_,p)=>p!==actor);
  const economy=p=>[p.money/100,p.income/10,p.vp/100,p.spent/50];
  const board=p=>{
   const owned=view.tiles.filter(t=>t.owner===p),links=view.links.filter(l=>l.owner===p);
   return types.map(type=>owned.filter(t=>t.type===type).length/4)
    .concat(types.map(type=>owned.filter(t=>t.type===type&&!t.flipped).length/4),
     ['coalMine','ironWorks','brewery'].map(type=>owned.filter(t=>t.type===type).reduce((n,t)=>n+t.cubes,0)/8),
     ['canal','rail'].map(era=>links.filter(l=>l.era===era).length/14),
     [owned.filter(t=>['cottonMill','manufacturer','pottery'].includes(t.type)&&!t.flipped).reduce((n,t)=>n+t.vp,0)/100]);
  };
  const mean=rows=>rows[0].map((_,i)=>rows.reduce((n,row)=>n+row[i],0)/rows.length);
  const v=[+(view.era==='rail'),view.round/16,view.players.length/4,view.deckSize/64,view.actionsThisTurn/2,
   own.handSize/8,view.coalMarket/15,view.ironMarket/11,...economy(own),...mean(other.map(economy)),
   ...board(actor),...mean(view.players.map((_,p)=>p).filter(p=>p!==actor).map(board)),
   ...types.map(t=>(own.nextLevels[t]||0)/8),...['cottonMill','manufacturer','pottery'].map(t=>+view.demands.includes(t)),
   +own.wildLocation,+own.wildIndustry];
  if(v.length!==featureNames.length||v.some(x=>!Number.isFinite(x)))throw Error('Invalid human-intent features');
  return v;
 }
 function fromState(s){
  const tiles=Object.values(s.boardIndustries).concat(Object.values(s.breweryFarmTiles)).filter(Boolean)
   .map(t=>({owner:t.playerId,type:t.type,flipped:!!t.flipped,cubes:t.resourceCubes||0,vp:t.tileData.vp}));
  return {era:s.era,round:s.round,deckSize:s.drawDeck.length,actionsThisTurn:s.actionsThisTurn,coalMarket:s.coalMarket,ironMarket:s.ironMarket,
   players:s.players.map((p,i)=>({money:p.money,income:s.getIncomeAmount(p.income),vp:p.vp,spent:s.moneySpentThisRound[i]||0,
    handSize:p.hand.length,wildLocation:!!p.hasWildLocation,wildIndustry:!!p.hasWildIndustry,
    nextLevels:Object.fromEntries(types.map(t=>[t,s.getNextTile(i,t)?.level||0]))})),tiles,
   links:Object.values(s.boardLinks).map(l=>({owner:l.playerId,era:l.type})),
   demands:['cottonMill','manufacturer','pottery'].filter(t=>s.merchantTiles.some(m=>m.buys===null||m.buys===t))};
 }
 function intents(a){
  const t=a.target||{};
  if(a.action==='build')return ['build:'+t.industryType];
  if(a.action==='develop')return [...new Set([t.type1,t.type2].filter(Boolean).map(x=>'develop:'+x))];
  if(a.action==='network')return ['network:'+(t.connectionIds?2:1)];
  return [a.action];
 }
 function forward(model,input){
  let v=input;
  for(const l of model.layers){const out=l.bias.slice();for(let i=0;i<v.length;i++)for(let j=0;j<out.length;j++)out[j]+=v[i]*l.weights[i][j];v=l.activation==='relu'?out.map(x=>Math.max(0,x)):out;}
  const max=Math.max(...v),exp=v.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(x=>x/sum);
 }
 const validated=new WeakSet();
 function validate(model){
  if(!model||model.featureVersion!=='public-intent-v1'||JSON.stringify(model.classes)!==JSON.stringify(classes)||
   JSON.stringify(model.featureNames)!==JSON.stringify(featureNames)||model.inputDim!==featureNames.length)throw Error('Incompatible human intent prior');
  if(validated.has(model))return model;
  for(const [name,size,positive] of [['inputMean',featureNames.length,false],['inputScale',featureNames.length,true],['classFrequency',classes.length,true]]){
   if(!Array.isArray(model[name])||model[name].length!==size||model[name].some(x=>!Number.isFinite(x)||positive&&x<=0))throw Error('Invalid intent normalization');
  }
  if(Math.abs(model.classFrequency.reduce((a,b)=>a+b,0)-1)>1e-5)throw Error('Invalid intent frequencies');
  let size=featureNames.length;
  if(!Array.isArray(model.layers)||!model.layers.length)throw Error('Missing intent layers');
  for(const l of model.layers){
   if(!Array.isArray(l.bias)||!l.bias.length||l.bias.some(x=>!Number.isFinite(x))||!['relu','linear'].includes(l.activation)||
    !Array.isArray(l.weights)||l.weights.length!==size||l.weights.some(r=>!Array.isArray(r)||r.length!==l.bias.length||r.some(x=>!Number.isFinite(x))))throw Error('Invalid intent layers');
   size=l.bias.length;
  }
  if(size!==classes.length)throw Error('Invalid intent output');
  validated.add(model);return model;
 }
 function predictFeatures(model,v){
  validate(model);
  if(v.length!==featureNames.length||v.some(x=>!Number.isFinite(x)))throw Error('Invalid intent input');
  return forward(model,v.map((x,i)=>(x-model.inputMean[i])/model.inputScale[i]));
 }
 function probabilities(model,s,actor=s.currentPlayerId){
  return predictFeatures(model,features(fromState(s),actor));
 }
 function bonus(model,probability,a,weight){
  const ids=intents(a).map(k=>classes.indexOf(k));if(ids.some(i=>i<0))throw Error('Unknown action intent');
  return weight*ids.reduce((n,i)=>n+Math.max(-3,Math.min(3,Math.log(Math.max(.0001,probability[i])/Math.max(.005,model.classFrequency[i])))),0)/ids.length;
 }
 const api={types,classes,featureNames,features,fromState,intents,validate,predictFeatures,probabilities,bonus};
 if(node)module.exports=api;else root.BrassHumanIntent=api;
})(globalThis);
