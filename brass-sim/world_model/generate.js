const fs = require('node:fs');
const path = require('node:path');
const Sim = require('./simulator');
const E = require('./encoding');
const Legacy = require('../scripts/autorun');
const Logic = require('../js/gameLogic');

function generate({games=1000,players=4,seed=1701,out='world_model/data',ai='mixed',exploration=0.3,depth=1}={}) {
    if(!Number.isInteger(games)||games<10) throw Error('Use at least 10 complete games for train/validation/test splitting');
    if(![2,3,4].includes(players)||!Number.isInteger(seed)) throw Error('Invalid players or seed');
    if(!['mixed','heuristic','random','search'].includes(ai)) throw Error('AI must be mixed, heuristic or random');
    if(!Number.isFinite(exploration)||exploration<0||exploration>1)throw Error('Exploration must be between 0 and 1');
    fs.mkdirSync(out,{recursive:true});
    if(fs.existsSync(path.join(out,'schema.json'))) throw Error('Dataset exists; choose a new --out directory');
    const dataFd=fs.openSync(path.join(out,'transitions.f32'),'w');
    const metaFd=fs.openSync(path.join(out,'transitions.jsonl'),'w');
    const splits={train:[],validation:[],test:[]}, counts={}, gameRows=[];
    const sampleState=Sim.create(players,seed), sampleAction=Sim.candidates(sampleState)[0];
    const stateDim=E.fields.length, actionDim=E.encodeAction(sampleAction,sampleState).length;
    let row=0;
    try {
        for(let g=0;g<games;g++) {
            const gameSeed=seed+g*9973, rng=Sim.rng(gameSeed^0x72ab39);
            const split=g%10===8?'validation':g%10===9?'test':'train';
            let state=Sim.create(players,gameSeed), turn=0;
            const start=row;
            while(!state.gameOver) {
                if(++turn>1000) throw Error(`Game ${g} failed to terminate`);
                // Retain the original bot's retry behavior for reproducible data.
                let list=state.rulesVersion==='economy-v2'?Sim.candidates(state):Legacy.collectCandidates(state,new Logic(state),state.currentPlayerId), action, next;
                while(list.length) {
                    if(ai==='random'||((ai==='mixed'||ai==='search')&&rng.next()<exploration)) {
                        const kinds=[...new Set(list.map(a=>a.action))];
                        const kind=rng.pick(kinds);
                        action=rng.pick(list.filter(a=>a.action===kind));
                        if(!action) action=rng.pick(list);
                    } else if(ai==='search')action=require('./planner').plan(state,{type:'search',depth,width:8}).selected;
                    else action=list.slice().sort((a,b)=>b.score-a.score)[0];
                    try {next=Sim.step(state,action,{validate:false});break;}
                    catch {list=list.filter(a=>Sim.key(a)!==Sim.key(action));}
                }
                if(!next) throw Error(`No executable action in game ${g}`);
                const s=E.encodeState(state), a=E.encodeAction(action,state), n=E.encodeState(next.state);
                const values=Float32Array.from([...s,...a,...n,action.score/100]);
                fs.writeSync(dataFd,Buffer.from(values.buffer));
                fs.writeSync(metaFd,JSON.stringify({row,game:g,seed:gameSeed,split,turn,player:state.currentPlayerId,
                    era:state.era,action:action.action,label:Sim.label(action),move:action,
                    reward:next.state.players[state.currentPlayerId].vp-state.currentPlayer.vp,event:next.event.type})+'\n');
                counts[action.action]=(counts[action.action]||0)+1; splits[split].push(row++); state=next.state;
            }
            gameRows.push({game:g,seed:gameSeed,split,start,end:row,finalScores:state.players.map(p=>({id:p.id,vp:p.vp,income:p.income,money:p.money}))});
            if((g+1)%50===0||g+1===games) console.log(`Generated ${g+1}/${games} games, ${row} transitions`);
        }
    } finally {fs.closeSync(dataFd);fs.closeSync(metaFd);}
    const schema={...E.schema,stateDim,actionDim,rowWidth:2*stateDim+actionDim+1,rows:row,
        games,players,seed,ai,exploration,depth,counts,format:'little-endian float32 [state,action,next_state,heuristic_score/100]',generatedAt:new Date().toISOString()};
    fs.writeFileSync(path.join(out,'schema.json'),JSON.stringify(schema,null,2));
    fs.writeFileSync(path.join(out,'splits.json'),JSON.stringify(splits));
    fs.writeFileSync(path.join(out,'games.json'),JSON.stringify(gameRows,null,2));
    console.log(JSON.stringify({games,transitions:row,stateDim,actionDim,counts,out},null,2));
    return schema;
}
module.exports={generate};
