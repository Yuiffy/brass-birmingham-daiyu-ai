// Audit exact teacher replays, then add legal counterfactual double rails only to
// training games. Held-out state/successor labels stay intact; actions are reencoded.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
function augment({data,out,teacherRoot,maxPerState=6}) {
    if(fs.existsSync(out)&&fs.readdirSync(out).length)throw Error('Choose fresh augmentation output');
    if(!Number.isInteger(maxPerState)||maxPerState<0)throw Error('Invalid augmentation limit');
    const meta=JSON.parse(fs.readFileSync(path.join(data,'schema.json'))),games=JSON.parse(fs.readFileSync(path.join(data,'games.json')));
    if(meta.rulesVersion!=='economy-v2')throw Error('Requires calibrated economy-v2 teacher data');
    process.env.BRASS_RULES=meta.rulesVersion;
    const Sim=require(path.resolve(teacherRoot,'world_model/simulator'));
    const E=require(path.resolve(teacherRoot,'world_model/encoding'));
    const UpdatedEncoding=require('./encoding');
    const buffer=fs.readFileSync(path.join(data,'transitions.f32'));
    const raw=new Float32Array(buffer.buffer,buffer.byteOffset,buffer.byteLength/4);
    if(raw.length!==meta.rows*meta.rowWidth)throw Error('Incomplete teacher data');
    const {stateDim:sd,actionDim:ad,rowWidth:rw}=meta;
    const close=(a,b)=>a.length===b.length&&a.every((v,i)=>Math.abs(v-b[i])<2e-6);
    const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    fs.mkdirSync(out,{recursive:true});const fd=fs.openSync(path.join(out,'transitions.f32'),'wx');
    const records=[],splits={train:[],validation:[],test:[]},counts={...meta.counts};let rows=0,added=0;
    const write=values=>{fs.writeSync(fd,Buffer.from(Float32Array.from(values).buffer));rows++;};
    try{for(const g of games){
        let state=Sim.create(meta.players,g.seed,g.seats),start=rows;
        for(let index=g.start;index<g.end;index++) {
            const row=raw.subarray(index*rw,(index+1)*rw),observable=E.encodeState(state);
            if(!close(observable,row.subarray(0,sd)))throw Error(`Replay state mismatch game ${g.game} row ${index}`);
            const candidates=Sim.candidates(state,{doubleRail:true});
            const matching=candidates.filter(a=>close(E.encodeAction(a,state),row.subarray(sd,sd+ad)));
            if(!matching.length)throw Error(`Recorded action unavailable ${g.game}:${index}`);
            // Beer-source identity is not encoded in the existing action schema.
            // Audit against the recorded successor to resolve identical encodings;
            // never use that successor as a model input or a planning oracle.
            let next=null,recordedAction=null;
            for(const action of matching) {
                const trial=Sim.step(state,action,{validate:false}).state;
                if(close(E.encodeState(trial),row.subarray(sd+ad,sd+ad+sd))){next=trial;recordedAction=action;break;}
            }
            if(!next)throw Error(`Replay successor mismatch ${g.game}:${index}`);
            write([...row.subarray(0,sd),...UpdatedEncoding.encodeAction(recordedAction,state,{version:'resource-network-v2'}),...row.subarray(sd+ad)]);
            if(g.split==='train'&&state.era==='rail') {
                const doubles=candidates.filter(a=>a.target?.connectionIds);
                const rng=Sim.rng(g.seed^index);
                for(let n=0;n<Math.min(maxPerState,doubles.length);n++) {
                    const i=rng.int(doubles.length),counter=doubles.splice(i,1)[0];
                    const imagined=Sim.step(state,counter,{validate:false}).state;
                    write([...observable,...UpdatedEncoding.encodeAction(counter,state,{version:'resource-network-v2'}),...E.encodeState(imagined),counter.score/100]);
                    added++;counts.network=(counts.network||0)+1;counts.doubleRail=(counts.doubleRail||0)+1;
                }
            }
            state=next;
        }
        if(!state.gameOver)throw Error('Incomplete audited game');
        records.push({...g,start,end:rows});for(let i=start;i<rows;i++)splits[g.split].push(i);
        if(records.length%20===0)console.log(`Audited ${records.length}/${games.length} games, added ${added} legal double rails`);
    }}finally{fs.closeSync(fd);}
    const provenance={...meta.provenance,augmentation:'legal-engine-double-rail-counterfactuals-training-only',
        addedRows:added,parentSHA256:sha(path.join(data,'transitions.f32')),maxPerState,
        trainingRowsContiguous:false,heldoutSuccessorsUnchanged:true,heldoutActionsReencoded:true};
    fs.writeFileSync(path.join(out,'schema.json'),JSON.stringify({...meta,rows,counts,provenance,actionEncoding:'resource-network-v2'}));
    fs.writeFileSync(path.join(out,'games.json'),JSON.stringify(records));fs.writeFileSync(path.join(out,'splits.json'),JSON.stringify(splits));
    console.log(JSON.stringify({rows,added,games:records.length,counts}));
}
if(require.main===module){const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].slice(2)]=process.argv[i+1];
    augment({...args,maxPerState:Number(args.maxPerState||6)});}
module.exports={augment};
